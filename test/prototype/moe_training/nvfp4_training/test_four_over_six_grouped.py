# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.nvfp4_training.four_over_six import (
    four_over_six_dequantize,
    four_over_six_linear,
    four_over_six_quantize,
)
from torchao.prototype.moe_training.nvfp4_training import four_over_six_grouped
from torchao.prototype.moe_training.nvfp4_training.four_over_six_grouped import (
    four_over_six_grouped_mm,
)
from torchao.utils import is_sm_at_least_100, torch_version_at_least

_skip_no_sm100 = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and is_sm_at_least_100()
        and torch_version_at_least("2.10.0")
    ),
    reason="requires SM100+ and PyTorch 2.10+ (FP4 scaled_grouped_mm)",
)


def _make_grouped_inputs(group_sizes, K, N, seed=0, device="cuda"):
    """Packed activations, stacked expert weights, and end offsets."""
    torch.manual_seed(seed)
    M = sum(group_sizes)
    E = len(group_sizes)
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(E, N, K, dtype=torch.bfloat16, device=device) * 0.1
    offs = torch.tensor(group_sizes, dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )
    return A, B, offs


@_skip_no_sm100
@pytest.mark.parametrize("err_mode", ["mae", "mse"])
@pytest.mark.parametrize("e4m3_scale_bound", [256, 448])
def test_group_expanded_amax_matches_per_split_quantize(err_mode, e4m3_scale_bound):
    """One quantize call with group-expanded amaxes == a per-split loop."""
    group_sizes = [128, 384, 256]
    A, _, offs = _make_grouped_inputs(group_sizes, K=256, N=128)
    group_amax = torch.stack(
        [
            A[start:end].abs().amax().to(torch.float32)
            for start, end in zip([0, *offs.tolist()[:-1]], offs.tolist())
        ]
    )
    expanded = group_amax.repeat_interleave(
        torch.tensor(group_sizes, device=A.device)
    )
    codes, scales = four_over_six_quantize(
        A, expanded, err_mode=err_mode, e4m3_scale_bound=e4m3_scale_bound
    )
    start = 0
    for g, end in enumerate(offs.tolist()):
        split_codes, split_scales = four_over_six_quantize(
            A[start:end].contiguous(),
            group_amax[g],
            err_mode=err_mode,
            e4m3_scale_bound=e4m3_scale_bound,
        )
        torch.testing.assert_close(codes[start:end], split_codes, atol=0, rtol=0)
        torch.testing.assert_close(
            scales[start:end].view(torch.uint8),
            split_scales.view(torch.uint8),
            atol=0,
            rtol=0,
        )
        start = end


@_skip_no_sm100
@pytest.mark.parametrize("weight_block", ["16x16", "1x16"])
def test_per_tensor_grouped_forward_matches_dense_loop(weight_block):
    """Grouped forward vs dense four_over_six GEMMs per 128-aligned group.

    The quantized operands are bitwise-identical by construction (pinned by
    the amax-expansion test above); the GEMM outputs are compared bitwise
    and fall back to an SQNR bound if the grouped and dense kernels reduce
    in different orders.
    """
    group_sizes = [128, 256, 128]
    K, N = 256, 384
    A, B, offs = _make_grouped_inputs(group_sizes, K=K, N=N)
    y = four_over_six_grouped_mm(A, B, offs, weight_block=weight_block)

    start = 0
    refs = []
    for e, end in enumerate(offs.tolist()):
        refs.append(
            four_over_six_linear(
                A[start:end].contiguous(),
                B[e],
                None,
                "mae",
                256,
                False,
                "high_precision",
                weight_block,
            )
        )
        start = end
    y_ref = torch.cat(refs)
    if not torch.equal(y, y_ref):
        sqnr = compute_error(y_ref.float(), y.float())
        assert sqnr > 85.0, f"grouped vs dense-loop forward SQNR {sqnr:.1f} dB"
        print(f"\ngrouped GEMM reduction differs from dense: SQNR {sqnr:.1f} dB")


@_skip_no_sm100
def test_row_scaled_grouped_forward_matches_dense_loop():
    """Row-scaled grouped forward vs dense four_over_six GEMMs per group.

    On builds without an FP32 grouped-GEMM output the default path is the
    per-group dense loop, which matches bitwise; when the fused single-GEMM
    path runs, the grouped kernel may reduce in a different order, so the
    comparison falls back to an SQNR bound.
    """
    group_sizes = [128, 256, 128]
    A, B, offs = _make_grouped_inputs(group_sizes, K=256, N=384)
    y = four_over_six_grouped_mm(A, B, offs, row_scaled_activation=True)

    start = 0
    refs = []
    for e, end in enumerate(offs.tolist()):
        refs.append(
            four_over_six_linear(
                A[start:end].contiguous(), B[e], None, "mae", 256, True
            )
        )
        start = end
    y_ref = torch.cat(refs)
    if not torch.equal(y, y_ref):
        sqnr = compute_error(y_ref.float(), y.float())
        assert sqnr > 85.0, f"row-scaled grouped vs dense-loop SQNR {sqnr:.1f} dB"
        print(f"\nrow-scaled grouped forward differs from dense: SQNR {sqnr:.1f} dB")


@_skip_no_sm100
@pytest.mark.parametrize(
    "group_sizes, pad",
    [
        pytest.param([128] * 64, False, id="uniform-128-rows-per-expert"),
        pytest.param(
            [128 if e % 8 == 0 else 0 for e in range(64)],
            False,
            id="decode-like-8-active-56-empty",
        ),
        pytest.param([1, 220, 77], True, id="ragged-padded"),
    ],
)
def test_row_scaled_fused_matches_loop_oracle(group_sizes, pad, monkeypatch):
    """Fused single-GEMM row-scaled forward vs the per-group loop oracle.

    The bf16-output fused GEMM rounds its output once before the fp32 row
    scale — an unavoidable extra rounding worth ~51 dB — so the exactness
    check compares against the loop oracle with that one rounding emulated
    on its dense fp32 GEMMs (bitwise, with the reduction-order SQNR
    fallback), and the raw loop comparison only bounds the rounding cost.
    The quantized operands are identical by construction.
    """
    A, B, offs = _make_grouped_inputs(group_sizes, K=2048, N=768, seed=5)
    monkeypatch.setattr(
        four_over_six_grouped, "_grouped_mm_fp32_out_supported", lambda _i: False
    )
    monkeypatch.setattr(
        four_over_six_grouped,
        "FOUR_OVER_SIX_GROUPED_ROW_SCALED_FUSED_BF16_OUT",
        False,
    )
    y_loop = four_over_six_grouped_mm(
        A,
        B,
        offs,
        row_scaled_activation=True,
        pad_token_groups_for_grouped_mm=pad,
    )

    # Loop oracle with the fused path's bf16 GEMM-output rounding emulated.
    dense_gemm = four_over_six_grouped._scaled_mm_nvfp4

    def _dense_gemm_bf16_rounded(*args):
        return dense_gemm(*args).to(torch.bfloat16).to(torch.float32)

    monkeypatch.setattr(
        four_over_six_grouped, "_scaled_mm_nvfp4", _dense_gemm_bf16_rounded
    )
    y_emul = four_over_six_grouped_mm(
        A,
        B,
        offs,
        row_scaled_activation=True,
        pad_token_groups_for_grouped_mm=pad,
    )
    monkeypatch.setattr(four_over_six_grouped, "_scaled_mm_nvfp4", dense_gemm)

    monkeypatch.setattr(
        four_over_six_grouped,
        "FOUR_OVER_SIX_GROUPED_ROW_SCALED_FUSED_BF16_OUT",
        True,
    )
    y_fused = four_over_six_grouped_mm(
        A,
        B,
        offs,
        row_scaled_activation=True,
        pad_token_groups_for_grouped_mm=pad,
    )
    assert y_fused.shape == y_loop.shape
    if not torch.equal(y_fused, y_emul):
        sqnr = compute_error(y_emul.float(), y_fused.float())
        assert sqnr > 85.0, f"fused vs rounding-emulated oracle SQNR {sqnr:.1f} dB"
        print(f"\nfused grouped GEMM reduction differs from dense: SQNR {sqnr:.1f} dB")
    rounding_sqnr = compute_error(y_loop.float(), y_fused.float())
    assert rounding_sqnr > 45.0, f"fused vs loop-oracle SQNR {rounding_sqnr:.1f} dB"
    print(f"\nbf16 GEMM-output rounding cost vs loop: SQNR {rounding_sqnr:.1f} dB")


@_skip_no_sm100
@pytest.mark.parametrize("err_mode", ["mae", "mse"])
@pytest.mark.parametrize("e4m3_scale_bound", [256, 448])
def test_batched_weight_quantize_matches_per_expert_loop(
    err_mode, e4m3_scale_bound
):
    """The one-call flattened 1x16 weight quantize == the per-expert loop."""
    torch.manual_seed(2)
    E, N, K = 5, 128, 256
    B = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.1
    weight_amax = B.abs().amax(dim=(1, 2)).to(torch.float32)
    codes, scales = four_over_six_grouped._quantize_expert_weights(
        B, weight_amax, "1x16", err_mode, e4m3_scale_bound
    )
    for e in range(E):
        ref_codes, ref_scales = four_over_six_quantize(
            B[e],
            weight_amax[e],
            block="1x16",
            err_mode=err_mode,
            e4m3_scale_bound=e4m3_scale_bound,
        )
        torch.testing.assert_close(codes[e], ref_codes, atol=0, rtol=0)
        torch.testing.assert_close(
            scales[e].view(torch.uint8),
            ref_scales.view(torch.uint8),
            atol=0,
            rtol=0,
        )


@_skip_no_sm100
def test_row_scaled_dispatch_and_loop_fallback(monkeypatch):
    """The dispatch takes the loop when no fused output mode is available,
    and the fused single GEMM when the bf16-output knob is set."""
    group_sizes = [128, 256, 128]
    A, B, offs = _make_grouped_inputs(group_sizes, K=256, N=384, seed=13)
    fused_gemm = four_over_six_grouped._row_scaled_single_grouped_gemm

    monkeypatch.setattr(
        four_over_six_grouped, "_grouped_mm_fp32_out_supported", lambda _i: False
    )
    monkeypatch.setattr(
        four_over_six_grouped,
        "FOUR_OVER_SIX_GROUPED_ROW_SCALED_FUSED_BF16_OUT",
        False,
    )

    def _reject_fused(*args, **kwargs):
        raise AssertionError("fused single grouped GEMM must not run")

    monkeypatch.setattr(
        four_over_six_grouped, "_row_scaled_single_grouped_gemm", _reject_fused
    )
    y_loop = four_over_six_grouped_mm(A, B, offs, row_scaled_activation=True)

    monkeypatch.setattr(
        four_over_six_grouped, "_row_scaled_single_grouped_gemm", fused_gemm
    )
    monkeypatch.setattr(
        four_over_six_grouped,
        "FOUR_OVER_SIX_GROUPED_ROW_SCALED_FUSED_BF16_OUT",
        True,
    )

    def _reject_loop(*args, **kwargs):
        raise AssertionError("loop fallback must not run")

    monkeypatch.setattr(
        four_over_six_grouped, "_row_scaled_gemm_loop", _reject_loop
    )
    y_fused = four_over_six_grouped_mm(A, B, offs, row_scaled_activation=True)

    # The bf16 GEMM-output rounding costs ~51 dB vs the loop's fp32 output;
    # the exactness contract is pinned by the rounding-emulated oracle test.
    sqnr = compute_error(y_loop.float(), y_fused.float())
    assert sqnr > 45.0, f"fused vs loop SQNR {sqnr:.1f} dB"


@_skip_no_sm100
@pytest.mark.parametrize("row_scaled_activation", [False, True])
def test_grouped_backward_high_precision(row_scaled_activation):
    """dx/dw are bf16 grouped GEMMs on the original operands."""
    group_sizes = [128, 256, 128]
    A, B, offs = _make_grouped_inputs(group_sizes, K=256, N=384)
    A.requires_grad_(True)
    B.requires_grad_(True)
    y = four_over_six_grouped_mm(
        A, B, offs, row_scaled_activation=row_scaled_activation
    )
    dy = torch.randn_like(y)
    y.backward(dy)

    dx_ref = torch._grouped_mm(
        dy, B.detach(), offs=offs, out_dtype=torch.bfloat16
    )
    dw_ref = torch._grouped_mm(
        dy.transpose(-2, -1), A.detach(), offs=offs, out_dtype=torch.bfloat16
    )
    torch.testing.assert_close(A.grad, dx_ref, atol=0, rtol=0)
    torch.testing.assert_close(B.grad, dw_ref, atol=0, rtol=0)


@_skip_no_sm100
@pytest.mark.parametrize("row_scaled_activation", [False, True])
@pytest.mark.parametrize("weight_block", ["16x16", "1x16"])
def test_grouped_backward_dequantized(row_scaled_activation, weight_block):
    """dx/dw are bf16 grouped GEMMs on dequantized fprop operands."""
    group_sizes = [128, 256, 128]
    K, N = 256, 384
    A, B, offs = _make_grouped_inputs(group_sizes, K=K, N=N)
    A.requires_grad_(True)
    B.requires_grad_(True)
    y = four_over_six_grouped_mm(
        A,
        B,
        offs,
        err_mode="mse",
        row_scaled_activation=row_scaled_activation,
        weight_block=weight_block,
        backward_override="dequantized",
    )
    dy = torch.randn_like(y)
    y.backward(dy)

    A_hp, B_hp = A.detach(), B.detach()
    if row_scaled_activation:
        x_amax = A_hp.abs().amax(dim=1).to(torch.float32)
    else:
        group_amax = []
        start = 0
        for end in offs.tolist():
            group_amax.append(A_hp[start:end].abs().amax().to(torch.float32))
            start = end
        x_amax = torch.stack(group_amax).repeat_interleave(
            torch.tensor(group_sizes, device=A.device)
        )
    x_codes, x_scales = four_over_six_quantize(A_hp, x_amax, err_mode="mse")
    x_dq = four_over_six_dequantize(x_codes, x_scales, x_amax)
    w_dq = []
    for e in range(B.shape[0]):
        w_amax = B_hp[e].abs().amax().to(torch.float32)
        w_codes, w_scales = four_over_six_quantize(
            B_hp[e], w_amax, block=weight_block, err_mode="mse"
        )
        w_dq.append(four_over_six_dequantize(w_codes, w_scales, w_amax))
    w_dq = torch.stack(w_dq)

    dx_ref = torch._grouped_mm(dy, w_dq, offs=offs, out_dtype=torch.bfloat16)
    dw_ref = torch._grouped_mm(
        dy.transpose(-2, -1), x_dq, offs=offs, out_dtype=torch.bfloat16
    )
    torch.testing.assert_close(A.grad, dx_ref, atol=0, rtol=0)
    torch.testing.assert_close(B.grad, dw_ref, atol=0, rtol=0)


@_skip_no_sm100
@pytest.mark.parametrize("row_scaled_activation", [False, True])
def test_grouped_padding_matches_aligned(row_scaled_activation):
    """Unaligned groups with padding == an aligned construction, per group."""
    K, N = 256, 384
    aligned_sizes = [128, 256, 128]
    ragged_sizes = [100, 220, 77]
    A_al, B, offs_al = _make_grouped_inputs(aligned_sizes, K=K, N=N, seed=3)
    # Ragged view: the first rows of each aligned group, so every ragged
    # group's rows (and hence its amax and quantization) exist verbatim in
    # the aligned run.
    ragged_rows = []
    start = 0
    for size, ragged in zip(aligned_sizes, ragged_sizes):
        ragged_rows.append(A_al[start : start + ragged])
        start += size
    A_rg = torch.cat(ragged_rows).contiguous()
    offs_rg = torch.tensor(
        ragged_sizes, dtype=torch.int32, device=A_al.device
    ).cumsum(0, dtype=torch.int32)

    y_rg = four_over_six_grouped_mm(
        A_rg,
        B,
        offs_rg,
        row_scaled_activation=row_scaled_activation,
        pad_token_groups_for_grouped_mm=True,
    )
    assert y_rg.shape == (sum(ragged_sizes), N)

    # Reference: dense per-group forward on the ragged rows padded to 128.
    start = 0
    for e, ragged in enumerate(ragged_sizes):
        rows = A_rg[start : start + ragged]
        padded = torch.zeros(128 * ((ragged + 127) // 128), K, dtype=rows.dtype, device=rows.device)
        padded[:ragged] = rows
        if row_scaled_activation:
            ref = four_over_six_linear(padded, B[e], None, "mae", 256, True)
        else:
            # Per-tensor group scale comes from the real rows' amax; the
            # zero padding rows cannot change it.
            ref = four_over_six_linear(
                padded, B[e], None, "mae", 256, False, "high_precision"
            )
        torch.testing.assert_close(
            y_rg[start : start + ragged], ref[:ragged], atol=0, rtol=0
        )
        start += ragged


@_skip_no_sm100
def test_grouped_validation():
    group_sizes = [128, 128]
    A, B, offs = _make_grouped_inputs(group_sizes, K=256, N=128)
    with pytest.raises(ValueError, match="no quantized backward"):
        four_over_six_grouped_mm(A, B, offs, backward_override="quantized")
    with pytest.raises(ValueError, match="1D int32"):
        four_over_six_grouped_mm(A, B, offs.to(torch.int64))
    with pytest.raises(ValueError, match="one group-end offset per expert"):
        four_over_six_grouped_mm(A, B, offs[:1])
    with pytest.raises(ValueError, match="must be 2D"):
        four_over_six_grouped_mm(A.unsqueeze(0), B, offs)
    with pytest.raises(ValueError, match="divisible by 128"):
        four_over_six_grouped_mm(A[:, :144], B[:, :, :144].contiguous(), offs)
    with pytest.raises(ValueError, match="weight_block"):
        four_over_six_grouped_mm(A, B, offs, weight_block="8x8")


@_skip_no_sm100
def test_grouped_miles_recipe_point():
    """The miles NVFP4 RL recipe: row-scaled + MSE + bound 256 + 1x16 weights
    + dequantized backward, on ragged token groups."""
    group_sizes = [100, 220, 77]
    A, B, offs = _make_grouped_inputs(group_sizes, K=256, N=384, seed=7)
    A.requires_grad_(True)
    B.requires_grad_(True)
    y = four_over_six_grouped_mm(
        A,
        B,
        offs,
        err_mode="mse",
        e4m3_scale_bound=256,
        row_scaled_activation=True,
        weight_block="1x16",
        backward_override="dequantized",
        pad_token_groups_for_grouped_mm=True,
    )
    assert y.shape == (sum(group_sizes), 384)
    y.backward(torch.randn_like(y))
    assert A.grad is not None and A.grad.shape == A.shape
    assert B.grad is not None and B.grad.shape == B.shape
    sqnr = compute_error(
        torch._grouped_mm(
            A.detach(), B.detach().transpose(-2, -1), offs=offs
        ).float(),
        y.float(),
    )
    assert sqnr > 14.0, f"quantization noise floor too high: {sqnr:.1f} dB"


@_skip_no_sm100
@pytest.mark.parametrize("row_scaled_activation", [False, True])
@pytest.mark.parametrize("weight_block", ["16x16", "1x16"])
def test_grouped_backward_dequantized_ragged(row_scaled_activation, weight_block):
    """Ragged groups + padding + dequantized backward, value-checked.

    This is the composition the torchtitan grouped-experts hook ships;
    the padded rows quantize to zeros and are unpadded away before the
    backward GEMMs, so the reference can quantize the ragged rows directly.
    """
    group_sizes = [100, 220, 77]
    K, N = 256, 384
    A, B, offs = _make_grouped_inputs(group_sizes, K=K, N=N, seed=7)
    A.requires_grad_(True)
    B.requires_grad_(True)
    y = four_over_six_grouped_mm(
        A,
        B,
        offs,
        err_mode="mse",
        row_scaled_activation=row_scaled_activation,
        weight_block=weight_block,
        backward_override="dequantized",
        pad_token_groups_for_grouped_mm=True,
    )
    dy = torch.randn_like(y)
    y.backward(dy)

    A_hp, B_hp = A.detach(), B.detach()
    if row_scaled_activation:
        x_amax = A_hp.abs().amax(dim=1).to(torch.float32)
    else:
        # Group amaxes come from the real rows; zero padding cannot raise them.
        group_amax = []
        start = 0
        for end in offs.tolist():
            group_amax.append(A_hp[start:end].abs().amax().to(torch.float32))
            start = end
        x_amax = torch.stack(group_amax).repeat_interleave(
            torch.tensor(group_sizes, device=A.device)
        )
    x_codes, x_scales = four_over_six_quantize(A_hp, x_amax, err_mode="mse")
    x_dq = four_over_six_dequantize(x_codes, x_scales, x_amax)
    w_dq = []
    for e in range(B.shape[0]):
        w_amax = B_hp[e].abs().amax().to(torch.float32)
        w_codes, w_scales = four_over_six_quantize(
            B_hp[e], w_amax, block=weight_block, err_mode="mse"
        )
        w_dq.append(four_over_six_dequantize(w_codes, w_scales, w_amax))
    w_dq = torch.stack(w_dq)

    dx_ref = torch._grouped_mm(dy, w_dq, offs=offs, out_dtype=torch.bfloat16)
    dw_ref = torch._grouped_mm(
        dy.transpose(-2, -1), x_dq, offs=offs, out_dtype=torch.bfloat16
    )
    torch.testing.assert_close(A.grad, dx_ref, atol=0, rtol=0)
    torch.testing.assert_close(B.grad, dw_ref, atol=0, rtol=0)


@_skip_no_sm100
@pytest.mark.parametrize(
    "backward_override", ["high_precision", "dequantized"]
)
def test_grouped_compile(backward_override):
    """fullgraph compile of the per-tensor grouped op, forward and backward.

    The op is nonstrict-traced under compile, so eager numerics carry over
    bitwise. Row-scaled mode is exempt: its per-group dense-GEMM loop reads
    the offsets on the host, which fullgraph capture cannot express (the
    titan converter rejects that combination at config time).
    """
    group_sizes = [128, 256, 128]
    A, B, offs = _make_grouped_inputs(group_sizes, K=256, N=384, seed=11)

    # Compile the decorated op directly (the mxfp8 grouped test's pattern);
    # calling a nonstrict-traced function from a compiled frame is rejected.
    A_e = A.clone().requires_grad_(True)
    B_e = B.clone().requires_grad_(True)
    y_eager = four_over_six_grouped_mm(
        A_e, B_e, offs, err_mode="mse", backward_override=backward_override
    )
    dy = torch.randn_like(y_eager)
    y_eager.backward(dy)

    A_c = A.clone().requires_grad_(True)
    B_c = B.clone().requires_grad_(True)
    try:
        y_compiled = torch.compile(four_over_six_grouped_mm, fullgraph=True)(
            A_c, B_c, offs, err_mode="mse", backward_override=backward_override
        )
    except torch._dynamo.exc.Unsupported as e:
        if "nonstrict_trace" in str(e):
            pytest.skip(
                "this torch build rejects autograd.Function outputs from "
                "nonstrict_trace-ed functions (the mxfp8 grouped compile "
                "test's pattern); coverage resumes on builds that accept it"
            )
        raise
    y_compiled.backward(dy)

    torch.testing.assert_close(y_compiled, y_eager, atol=0, rtol=0)
    torch.testing.assert_close(A_c.grad, A_e.grad, atol=0, rtol=0)
    torch.testing.assert_close(B_c.grad, B_e.grad, atol=0, rtol=0)
