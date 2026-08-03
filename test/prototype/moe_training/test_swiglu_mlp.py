# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the composite MXFP8 SwiGLU MLP (swiglu_mlp.py).

The composite's two modes must agree with each other (only the activation
boundary differs: unified kernel vs standalone BF16 + cast kernels), track the
existing per-GEMM mx_mm MLP, compile without graph breaks, and never launch
standalone activation-cast kernels in fused mode.
"""

import pytest
import torch
import torch.nn.functional as F

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10):
    pytest.skip("Requires CUDA SM 10.x (Blackwell)", allow_module_level=True)

from torch.profiler import ProfilerActivity, profile

from torchao.prototype.moe_training.swiglu_mlp import (
    _unfused_grouped_mlp,
    _unfused_mlp,
    mxfp8_swiglu_grouped_mlp_w13,
    mxfp8_swiglu_mlp_w13,
)
from torchao.quantization.utils import compute_error

# (M, D, H): llama3-debugmodel-like plus one wider shape.
_SHAPES = [
    (256, 256, 768),
    (512, 512, 1024),
]

_SWIGLU_OPS = {
    "torchao::swiglu_mxfp8_forward",
    "torchao::swiglu_mxfp8_backward",
}
_STANDALONE_CAST_OPS = {
    "torchao::mxfp8_quantize_2d_1x32_cutedsl",
    "torchao::mxfp8_quantize_2d_32x1_cutedsl",
}


def _make_inputs(m, d, h, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(m, d, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w13 = torch.randn(h, 2, d, dtype=torch.bfloat16, device="cuda") * 0.02
    w2 = torch.randn(d, h, dtype=torch.bfloat16, device="cuda") * 0.02
    w13.requires_grad_(True)
    w2.requires_grad_(True)
    return x, w13, w2


def _bf16_reference(x, w13, w2):
    gate, up = torch.einsum("...d,hgd->...hg", x, w13).unbind(-1)
    return F.linear(F.silu(gate.float()).to(x.dtype) * up, w2)


def _run(fn, x, w13, w2, **kwargs):
    x = x.clone().detach().requires_grad_(True)
    w13 = w13.clone().detach().requires_grad_(True)
    w2 = w2.clone().detach().requires_grad_(True)
    out = fn(x, w13, w2, **kwargs)
    torch.manual_seed(1234)
    out.backward(torch.randn_like(out))
    return out, x.grad, w13.grad, w2.grad


@pytest.mark.parametrize("shape", _SHAPES)
def test_fused_matches_unfused_mode(shape):
    m, d, h = shape
    x, w13, w2 = _make_inputs(m, d, h)
    out_f, dx_f, dw13_f, dw2_f = _run(
        mxfp8_swiglu_mlp_w13, x, w13, w2, fuse_activation=True
    )
    out_u, dx_u, dw13_u, dw2_u = _run(
        mxfp8_swiglu_mlp_w13, x, w13, w2, fuse_activation=False
    )
    # Forward h quantization is bitwise identical between the unified kernel
    # and the standalone casts, so the outputs must match exactly.
    torch.testing.assert_close(out_f, out_u, rtol=0, atol=0)
    # Backward may differ by one E4M3 code in <=1e-5 of [dGate | dUp] elements
    # (hardware ex2.approx sigmoid), so gradients are near- but not bitwise-equal.
    for got, ref, name in [
        (dx_f, dx_u, "dx"),
        (dw13_f, dw13_u, "dw13"),
        (dw2_f, dw2_u, "dw2"),
    ]:
        sqnr = compute_error(ref.float(), got.float())
        assert sqnr >= 50.0, f"{name} SQNR between modes {sqnr} < 50"


@pytest.mark.parametrize("fuse_activation", [True, False])
@pytest.mark.parametrize("shape", _SHAPES)
def test_tracks_mx_mm_reference(shape, fuse_activation):
    m, d, h = shape
    x, w13, w2 = _make_inputs(m, d, h)
    out, dx, dw13, dw2 = _run(
        mxfp8_swiglu_mlp_w13, x, w13, w2, fuse_activation=fuse_activation
    )
    ref_out, ref_dx, ref_dw13, ref_dw2 = _run(
        _unfused_mlp, x, w13, w2, wgrad_with_hp=False
    )
    for got, ref, name, min_sqnr in [
        (out, ref_out, "out", 25.0),
        (dx, ref_dx, "dx", 22.0),
        (dw13, ref_dw13, "dw13", 22.0),
        (dw2, ref_dw2, "dw2", 22.0),
    ]:
        sqnr = compute_error(ref.float(), got.float())
        assert sqnr >= min_sqnr, f"{name} SQNR vs mx_mm {sqnr} < {min_sqnr}"
        assert torch.isfinite(got).all(), f"{name} contains non-finite values"


@pytest.mark.parametrize("fuse_activation", [True, False])
def test_tracks_bf16_reference(fuse_activation):
    m, d, h = _SHAPES[0]
    x, w13, w2 = _make_inputs(m, d, h)
    out, dx, dw13, dw2 = _run(
        mxfp8_swiglu_mlp_w13, x, w13, w2, fuse_activation=fuse_activation
    )
    ref_out, ref_dx, ref_dw13, ref_dw2 = _run(_bf16_reference, x, w13, w2)
    for got, ref, name, min_sqnr in [
        (out, ref_out, "out", 20.0),
        (dx, ref_dx, "dx", 18.0),
        (dw13, ref_dw13, "dw13", 18.0),
        (dw2, ref_dw2, "dw2", 18.0),
    ]:
        sqnr = compute_error(ref.float(), got.float())
        assert sqnr >= min_sqnr, f"{name} SQNR vs bf16 {sqnr} < {min_sqnr}"


@pytest.mark.parametrize("fuse_activation", [True, False])
def test_compile(fuse_activation):
    m, d, h = _SHAPES[0]
    x, w13, w2 = _make_inputs(m, d, h)

    def fn(x, w13, w2):
        return mxfp8_swiglu_mlp_w13(x, w13, w2, fuse_activation=fuse_activation)

    eager_out, eager_dx, eager_dw13, eager_dw2 = _run(fn, x, w13, w2)
    compiled = torch.compile(fn, fullgraph=True)
    comp_out, comp_dx, comp_dw13, comp_dw2 = _run(compiled, x, w13, w2)

    torch.testing.assert_close(comp_out, eager_out, rtol=0, atol=0)
    for got, ref, name in [
        (comp_dx, eager_dx, "dx"),
        (comp_dw13, eager_dw13, "dw13"),
        (comp_dw2, eager_dw2, "dw2"),
    ]:
        # Inductor may fuse the BF16 elementwise math differently; quantized
        # GEMM inputs are identical custom-op outputs, so keep this tight.
        sqnr = compute_error(ref.float(), got.float())
        assert sqnr >= 50.0, f"compiled {name} SQNR {sqnr} < 50"


def _op_counts(fuse_activation):
    m, d, h = _SHAPES[0]
    x, w13, w2 = _make_inputs(m, d, h)
    # Warm up kernel JIT outside the profiled region.
    _run(mxfp8_swiglu_mlp_w13, x, w13, w2, fuse_activation=fuse_activation)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        _run(mxfp8_swiglu_mlp_w13, x, w13, w2, fuse_activation=fuse_activation)
    counts = {}
    for evt in prof.key_averages():
        if evt.key in _SWIGLU_OPS or evt.key in _STANDALONE_CAST_OPS:
            counts[evt.key] = evt.count
    return counts


def test_no_standalone_activation_casts_in_fused_mode():
    fused = _op_counts(fuse_activation=True)
    unfused = _op_counts(fuse_activation=False)

    # Fused mode: one unified kernel per direction, and the only standalone
    # casts are the GEMM-operand casts (fwd: x, w13, w2 rowwise; bwd: go
    # rowwise plus w2, w13, x, go colwise).
    assert fused.get("torchao::swiglu_mxfp8_forward", 0) == 1
    assert fused.get("torchao::swiglu_mxfp8_backward", 0) == 1
    assert fused.get("torchao::mxfp8_quantize_2d_1x32_cutedsl", 0) == 4
    assert fused.get("torchao::mxfp8_quantize_2d_32x1_cutedsl", 0) == 4

    # Unfused mode: no unified kernel; the SwiGLU boundary adds exactly one
    # rowwise + one colwise standalone cast per direction (h and [dGate|dUp]).
    assert unfused.get("torchao::swiglu_mxfp8_forward", 0) == 0
    assert unfused.get("torchao::swiglu_mxfp8_backward", 0) == 0
    assert unfused.get("torchao::mxfp8_quantize_2d_1x32_cutedsl", 0) == 6
    assert unfused.get("torchao::mxfp8_quantize_2d_32x1_cutedsl", 0) == 6


def test_wgrad_with_hp_falls_back_to_mx_mm_path():
    m, d, h = _SHAPES[0]
    x, w13, w2 = _make_inputs(m, d, h)
    got = _run(mxfp8_swiglu_mlp_w13, x, w13, w2, wgrad_with_hp=True)
    ref = _run(_unfused_mlp, x, w13, w2, wgrad_with_hp=True)
    for g, r, name in zip(got, ref, ["out", "dx", "dw13", "dw2"]):
        torch.testing.assert_close(g, r, rtol=0, atol=0, msg=name)
    # And the fused kernel must not have been used.
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        _run(mxfp8_swiglu_mlp_w13, x, w13, w2, wgrad_with_hp=True)
    keys = {evt.key for evt in prof.key_averages()}
    assert not (keys & _SWIGLU_OPS)


@pytest.mark.parametrize(
    "shape",
    [
        (100, 256, 768),  # M not a multiple of 32: BF16 fallback
        (96, 256, 768),  # M multiple of 32 but not 128: mx_mm fallback
        (256, 256, 704),  # H not a multiple of 128: mx_mm fallback
    ],
)
def test_unsupported_shapes_fall_back(shape):
    m, d, h = shape
    x, w13, w2 = _make_inputs(m, d, h)
    out, dx, dw13, dw2 = _run(mxfp8_swiglu_mlp_w13, x, w13, w2)
    for t, name in [(out, "out"), (dx, "dx"), (dw13, "dw13"), (dw2, "dw2")]:
        assert torch.isfinite(t).all(), f"{name} contains non-finite values"
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        _run(mxfp8_swiglu_mlp_w13, x, w13, w2)
    keys = {evt.key for evt in prof.key_averages()}
    assert not (keys & _SWIGLU_OPS)


@pytest.mark.parametrize("fuse_activation", [True, False])
def test_no_nans_with_large_inputs(fuse_activation):
    m, d, h = _SHAPES[0]
    x, w13, w2 = _make_inputs(m, d, h)
    with torch.no_grad():
        x.mul_(100.0)
    out, dx, dw13, dw2 = _run(
        mxfp8_swiglu_mlp_w13, x, w13, w2, fuse_activation=fuse_activation
    )
    for t, name in [(out, "out"), (dx, "dx"), (dw13, "dw13"), (dw2, "dw2")]:
        assert torch.isfinite(t).all(), f"{name} contains non-finite values"


# ---------------------------------------------------------------------------
# Grouped (MoE) composite
# ---------------------------------------------------------------------------

# Unequal per-expert token groups, all 128-row aligned as the token dispatcher
# guarantees (pad_multiple=128).
_GROUP_SIZES = [256, 128, 384, 256]
_GROUPED_E, _GROUPED_F, _GROUPED_D = 4, 256, 256


def _make_grouped_inputs(sizes=_GROUP_SIZES, f=_GROUPED_F, d=_GROUPED_D, seed=0):
    torch.manual_seed(seed)
    e = len(sizes)
    m = sum(sizes)
    offs = torch.tensor(sizes, dtype=torch.int32, device="cuda").cumsum(0).int()
    x = torch.randn(m, d, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w13 = torch.randn(e, f, 2, d, dtype=torch.bfloat16, device="cuda") * 0.02
    w2_edf = torch.randn(e, d, f, dtype=torch.bfloat16, device="cuda") * 0.02
    w13.requires_grad_(True)
    w2_edf.requires_grad_(True)
    return x, w13, w2_edf, offs


def _run_grouped(fn, x, w13, w2_edf, offs, **kwargs):
    x = x.clone().detach().requires_grad_(True)
    w13 = w13.clone().detach().requires_grad_(True)
    w2_edf = w2_edf.clone().detach().requires_grad_(True)
    out = fn(x, w13, w2_edf.transpose(-2, -1), offs, **kwargs)
    torch.manual_seed(1234)
    out.backward(torch.randn_like(out))
    return out, x.grad, w13.grad, w2_edf.grad


def _bf16_grouped_reference(x, w13, w2_t, offs):
    e, f, _, d = w13.shape
    w13_packed = w13.transpose(1, 2).reshape(e, 2 * f, d)
    gated = torch._grouped_mm(x, w13_packed.transpose(-2, -1), offs=offs)
    h = (F.silu(gated[:, :f].float()) * gated[:, f:].float()).to(gated.dtype)
    return torch._grouped_mm(h, w2_t, offs=offs)


def test_grouped_fused_matches_unfused_mode():
    x, w13, w2_edf, offs = _make_grouped_inputs()
    out_f, dx_f, dw13_f, dw2_f = _run_grouped(
        mxfp8_swiglu_grouped_mlp_w13, x, w13, w2_edf, offs, fuse_activation=True
    )
    out_u, dx_u, dw13_u, dw2_u = _run_grouped(
        mxfp8_swiglu_grouped_mlp_w13, x, w13, w2_edf, offs, fuse_activation=False
    )
    # Forward h quantization is bitwise identical between the modes.
    torch.testing.assert_close(out_f, out_u, rtol=0, atol=0)
    for got, ref, name in [
        (dx_f, dx_u, "dx"),
        (dw13_f, dw13_u, "dw13"),
        (dw2_f, dw2_u, "dw2"),
    ]:
        sqnr = compute_error(ref.float(), got.float())
        assert sqnr >= 50.0, f"{name} SQNR between modes {sqnr} < 50"


@pytest.mark.parametrize("fuse_activation", [True, False])
def test_grouped_tracks_references(fuse_activation):
    x, w13, w2_edf, offs = _make_grouped_inputs()
    out, dx, dw13, dw2 = _run_grouped(
        mxfp8_swiglu_grouped_mlp_w13, x, w13, w2_edf, offs,
        fuse_activation=fuse_activation,
    )
    ref = _run_grouped(
        lambda *a, **k: _unfused_grouped_mlp(*a, wgrad_with_hp=False),
        x, w13, w2_edf, offs,
    )
    bf16 = _run_grouped(_bf16_grouped_reference, x, w13, w2_edf, offs)
    for got, r, b, name in [
        (out, ref[0], bf16[0], "out"),
        (dx, ref[1], bf16[1], "dx"),
        (dw13, ref[2], bf16[2], "dw13"),
        (dw2, ref[3], bf16[3], "dw2"),
    ]:
        assert torch.isfinite(got).all(), f"{name} contains non-finite values"
        sqnr_q = compute_error(r.float(), got.float())
        assert sqnr_q >= 22.0, f"{name} SQNR vs grouped mx path {sqnr_q} < 22"
        sqnr_b = compute_error(b.float(), got.float())
        assert sqnr_b >= 18.0, f"{name} SQNR vs bf16 {sqnr_b} < 18"


def test_grouped_padded_rows_do_not_affect_results():
    """Appending a 128-row zero pad block to every expert group must not change
    real-row outputs or any gradient (pad rows carry zero upstream grad, as the
    un-permute backward guarantees)."""
    x, w13, w2_edf, offs = _make_grouped_inputs()
    d = x.shape[1]

    xs = list(x.split(_GROUP_SIZES))
    pad = x.new_zeros(128, d)
    x_padded = torch.cat([t for g in xs for t in (g, pad)])
    sizes_padded = [s + 128 for s in _GROUP_SIZES]
    offs_padded = (
        torch.tensor(sizes_padded, dtype=torch.int32, device="cuda").cumsum(0).int()
    )

    def run(xin, offs_in, sizes):
        xin = xin.clone().detach().requires_grad_(True)
        w13_ = w13.clone().detach().requires_grad_(True)
        w2_ = w2_edf.clone().detach().requires_grad_(True)
        out = mxfp8_swiglu_grouped_mlp_w13(
            xin, w13_, w2_.transpose(-2, -1), offs_in, fuse_activation=True
        )
        torch.manual_seed(1234)
        grads = torch.randn(sum(_GROUP_SIZES), out.shape[1], device="cuda").bfloat16()
        gsplit = list(grads.split(_GROUP_SIZES))
        if sizes != _GROUP_SIZES:
            grads = torch.cat([t for g in gsplit for t in (g, pad)])
        out.backward(grads)
        real = torch.cat(
            [t[:s] for t, s in zip(out.split(sizes), _GROUP_SIZES)]
        )
        real_dx = torch.cat(
            [t[:s] for t, s in zip(xin.grad.split(sizes), _GROUP_SIZES)]
        )
        return real, real_dx, w13_.grad, w2_.grad

    out_a, dx_a, dw13_a, dw2_a = run(x, offs, _GROUP_SIZES)
    out_b, dx_b, dw13_b, dw2_b = run(x_padded, offs_padded, sizes_padded)
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
    torch.testing.assert_close(dx_a, dx_b, rtol=0, atol=0)
    torch.testing.assert_close(dw13_a, dw13_b, rtol=0, atol=0)
    torch.testing.assert_close(dw2_a, dw2_b, rtol=0, atol=0)


def test_grouped_global_tail_padding_is_inert():
    """The dispatcher may pad the token buffer globally past offs[-1]; those
    tail rows belong to no expert group and must not affect real-row outputs
    or any gradient (and must not crash the scale-tile gather)."""
    x, w13, w2_edf, offs = _make_grouped_inputs()
    m, d = x.shape

    def run(xin, total_rows):
        xin = xin.clone().detach().requires_grad_(True)
        w13_ = w13.clone().detach().requires_grad_(True)
        w2_ = w2_edf.clone().detach().requires_grad_(True)
        out = mxfp8_swiglu_grouped_mlp_w13(
            xin, w13_, w2_.transpose(-2, -1), offs, fuse_activation=True
        )
        torch.manual_seed(1234)
        grads = torch.randn(m, out.shape[1], device="cuda").bfloat16()
        if total_rows > m:
            grads = torch.cat([grads, grads.new_zeros(total_rows - m, out.shape[1])])
        out.backward(grads)
        return out[:m], xin.grad[:m], w13_.grad, w2_.grad

    ref = run(x, m)
    tail = run(torch.cat([x.detach(), x.new_zeros(128, d)]), m + 128)
    for a, b, name in zip(ref, tail, ["out", "dx", "dw13", "dw2"]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("unbacked_m", [False, True])
def test_grouped_compile(unbacked_m):
    x, w13, w2_edf, offs = _make_grouped_inputs()

    def fn(x, w13, w2_edf):
        return mxfp8_swiglu_grouped_mlp_w13(
            x, w13, w2_edf.transpose(-2, -1), offs, fuse_activation=True
        )

    eager = _run_grouped(
        mxfp8_swiglu_grouped_mlp_w13, x, w13, w2_edf, offs, fuse_activation=True
    )
    compiled_fn = torch.compile(fn, fullgraph=True)

    xc = x.clone().detach().requires_grad_(True)
    if unbacked_m:
        # Routing-dependent M as EP token dispatch produces it: unbacked under
        # compile, so the support checks must defer to runtime asserts instead
        # of failing at trace time.
        torch._dynamo.decorators.mark_unbacked(xc, 0)
    w13c = w13.clone().detach().requires_grad_(True)
    w2c = w2_edf.clone().detach().requires_grad_(True)
    out = compiled_fn(xc, w13c, w2c)
    torch.manual_seed(1234)
    out.backward(torch.randn_like(out))

    torch.testing.assert_close(out, eager[0], rtol=0, atol=0)
    for got, ref, name in [
        (xc.grad, eager[1], "dx"),
        (w13c.grad, eager[2], "dw13"),
        (w2c.grad, eager[3], "dw2"),
    ]:
        sqnr = compute_error(ref.float(), got.float())
        assert sqnr >= 50.0, f"compiled {name} SQNR {sqnr} < 50"


def _grouped_op_counts(fuse_activation):
    x, w13, w2_edf, offs = _make_grouped_inputs()
    _run_grouped(
        mxfp8_swiglu_grouped_mlp_w13, x, w13, w2_edf, offs,
        fuse_activation=fuse_activation,
    )
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        _run_grouped(
            mxfp8_swiglu_grouped_mlp_w13, x, w13, w2_edf, offs,
            fuse_activation=fuse_activation,
        )
    counts = {}
    for evt in prof.key_averages():
        if evt.key in _SWIGLU_OPS or evt.key in _STANDALONE_CAST_OPS:
            counts[evt.key] = evt.count
    return counts


def test_grouped_no_standalone_activation_casts_in_fused_mode():
    fused = _grouped_op_counts(fuse_activation=True)
    unfused = _grouped_op_counts(fuse_activation=False)

    # Fused mode: the only 2D cutedsl casts are the GEMM-operand rowwise casts
    # of x (fwd) and grad_out (bwd); wgrad colwise casts use the CUDA dim1
    # kernel, mirroring the existing grouped wgrad path.
    assert fused.get("torchao::swiglu_mxfp8_forward", 0) == 1
    assert fused.get("torchao::swiglu_mxfp8_backward", 0) == 1
    assert fused.get("torchao::mxfp8_quantize_2d_1x32_cutedsl", 0) == 2
    assert fused.get("torchao::mxfp8_quantize_2d_32x1_cutedsl", 0) == 0

    # Unfused mode: the SwiGLU boundary adds one rowwise + one colwise
    # standalone cast per direction (h and [dGate | dUp]).
    assert unfused.get("torchao::swiglu_mxfp8_forward", 0) == 0
    assert unfused.get("torchao::swiglu_mxfp8_backward", 0) == 0
    assert unfused.get("torchao::mxfp8_quantize_2d_1x32_cutedsl", 0) == 4
    assert unfused.get("torchao::mxfp8_quantize_2d_32x1_cutedsl", 0) == 2


def test_grouped_fallbacks():
    # wgrad_with_hp falls back to the per-GEMM grouped path.
    x, w13, w2_edf, offs = _make_grouped_inputs()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        _run_grouped(
            mxfp8_swiglu_grouped_mlp_w13, x, w13, w2_edf, offs, wgrad_with_hp=True
        )
    keys = {evt.key for evt in prof.key_averages()}
    assert not (keys & _SWIGLU_OPS)

    # F a multiple of 32 but not 128: falls back, still finite.
    x, w13, w2_edf, offs = _make_grouped_inputs(f=192)
    res = _run_grouped(mxfp8_swiglu_grouped_mlp_w13, x, w13, w2_edf, offs)
    for t, name in zip(res, ["out", "dx", "dw13", "dw2"]):
        assert torch.isfinite(t).all(), f"{name} contains non-finite values"
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        _run_grouped(mxfp8_swiglu_grouped_mlp_w13, x, w13, w2_edf, offs)
    keys = {evt.key for evt in prof.key_averages()}
    assert not (keys & _SWIGLU_OPS)
