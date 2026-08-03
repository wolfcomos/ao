# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Correctness tests for the unified SwiGLU + MXFP8 kernel (cutedsl_swiglu_mxfp8).

Every case goes through the public API -- `swiglu_mxfp8_forward` or
`swiglu_mxfp8_backward` -- and is compared against the equivalent eager sequence:
  forward:  h = silu(gate) * up          -> mxfp8_quantize_2d_{1x32,32x1}_cutedsl(h)
  backward: dGate, dUp via SwiGLU grads  -> the same quantize functions

E8M0 scales and forward E4M3 data must match BITWISE. Backward E4M3 data is
allowed to differ by at most one code in a bounded fraction of elements.

That allowance is not slop: the kernel evaluates sigmoid as
`rcp.rn(1 + ex2.approx(-x * log2e))`, a hardware fast exponential chosen so the
kernel is bitwise consistent with the reference gated-MXFP8 implementation it
has to interoperate with. No PyTorch op reproduces `ex2.approx` bit for bit, so
exact agreement with `torch.sigmoid` is not achievable by construction. The
reference below otherwise mirrors the kernel's evaluation order exactly (the
kernel additionally contracts the d_silu term into one FMA, which eager cannot
reproduce), so those two lowerings are the only remaining sources of
difference. Measured over 6 seeds x 4 shapes: forward 0 of 101.6M codes
differ, backward 75 of 203.3M (3.7e-7), max gap 1 code, scales exact
throughout. _MAX_DIFFERING_FRACTION is set well above the measured rate so a
systematic error still fails.
"""

import pytest
import torch
import torch.nn.functional as F

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10):
    pytest.skip("Requires CUDA SM 10.x (Blackwell)", allow_module_level=True)

from torchao.prototype.moe_training.kernels.mxfp8 import (
    mxfp8_quantize_2d_1x32_cutedsl,
    mxfp8_quantize_2d_32x1_cutedsl,
    swiglu_mxfp8_backward,
    swiglu_mxfp8_forward,
)

_SHAPES = [
    (128, 128),
    (256, 256),
    (512, 2048),
    (1024, 7168),
]

_DIRECTIONS = [False, True]  # is_backward

_LAYOUTS = [
    (True, False),
    (False, True),
    (True, True),
]

# Backward E4M3 only; see the module docstring. Measured rate is 5.7e-7.
_MAX_DIFFERING_FRACTION = 1e-5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eager_reference(gated_input, grad_h):
    """The bf16 tensor the kernel is expected to quantize."""
    K = gated_input.shape[1] // 2
    gate = gated_input[:, :K].float()
    up = gated_input[:, K:].float()
    if grad_h is None:
        return (F.silu(gate) * up).bfloat16()
    grad_h_f = grad_h.float()
    # Mirror the kernel's evaluation order exactly:
    #   silu = gate * s ;  deriv = 1 + gate * (1 - s) ;  silu_grad = s * deriv
    #   dGate = (grad * up) * silu_grad ;  dUp = grad * silu
    # The kernel contracts `deriv` into a single FMA; eager has no fp32 fma,
    # so that term rounds twice here and stays a (tiny) source of divergence.
    sigmoid_gate = torch.sigmoid(gate)
    silu = gate * sigmoid_gate
    deriv = gate * (1.0 - sigmoid_gate) + 1.0
    silu_grad = sigmoid_gate * deriv
    dgate = ((grad_h_f * up) * silu_grad).bfloat16()
    dup = (grad_h_f * silu).bfloat16()
    return torch.cat([dgate, dup], dim=1)


def _assert_layout(actual, ref, msg):
    assert actual.shape == ref.shape, f"{msg}: shape {actual.shape} vs {ref.shape}"
    assert actual.stride() == ref.stride(), (
        f"{msg}: stride {actual.stride()} vs {ref.stride()}"
    )
    assert actual.dtype == ref.dtype, f"{msg}: dtype {actual.dtype} vs {ref.dtype}"


def _assert_identical(actual, ref, msg):
    """Bitwise equality, including shape, stride and dtype."""
    _assert_layout(actual, ref, msg)
    torch.testing.assert_close(actual, ref, rtol=0, atol=0, msg=msg)


def _assert_qdata_matches(actual, ref, msg, exact):
    """Exact for forward; at most a one-code difference for backward."""
    if exact:
        _assert_identical(actual, ref, msg)
        return
    _assert_layout(actual, ref, msg)
    a = actual.contiguous().view(torch.uint8).int()
    r = ref.contiguous().view(torch.uint8).int()
    gap = (a - r).abs()
    max_gap = int(gap.max())
    assert max_gap <= 1, f"{msg}: max E4M3 code gap {max_gap} > 1"
    fraction = int((gap != 0).sum()) / a.numel()
    assert fraction <= _MAX_DIFFERING_FRACTION, (
        f"{msg}: {fraction:.3e} of codes differ, limit {_MAX_DIFFERING_FRACTION:.3e}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M,K", _SHAPES)
@pytest.mark.parametrize("is_backward", _DIRECTIONS)
@pytest.mark.parametrize("rowwise,colwise", _LAYOUTS)
def test_swiglu_mxfp8(M, K, is_backward, rowwise, colwise):
    torch.manual_seed(42)
    gated_input = torch.randn(M, 2 * K, device="cuda", dtype=torch.bfloat16)
    grad_h = (
        torch.randn(M, K, device="cuda", dtype=torch.bfloat16) if is_backward else None
    )
    tag = f"bwd={is_backward} rowwise={rowwise} colwise={colwise} M={M} K={K}"

    if is_backward:
        outputs = swiglu_mxfp8_backward(
            grad_h, gated_input, rowwise=rowwise, colwise=colwise
        )
    else:
        outputs = swiglu_mxfp8_forward(gated_input, rowwise=rowwise, colwise=colwise)

    # Fixed four-output tuple regardless of which directions are enabled.
    assert isinstance(outputs, tuple) and len(outputs) == 4, f"{tag}: arity"
    output_rowwise, output_colwise, scales_rowwise, scales_colwise = outputs

    # Forward emits h (width K); backward emits [dGate | dUp] (width 2K).
    expected_width = 2 * K if is_backward else K
    reference = _eager_reference(gated_input, grad_h)
    assert reference.shape == (M, expected_width)

    if rowwise:
        ref_q, ref_s = mxfp8_quantize_2d_1x32_cutedsl(reference, scaling_mode="rceil")
        assert output_rowwise.shape == (M, expected_width), f"{tag}: rowwise width"
        assert output_rowwise.stride() == (expected_width, 1), (
            f"{tag}: rowwise qdata must be row-major, got {output_rowwise.stride()}"
        )
        _assert_qdata_matches(
            output_rowwise, ref_q, f"{tag}: rowwise qdata", exact=not is_backward
        )
        _assert_identical(scales_rowwise, ref_s, f"{tag}: rowwise scales")
    else:
        assert output_rowwise.numel() == 0, f"{tag}: rowwise output should be empty"
        assert scales_rowwise.numel() == 0, f"{tag}: rowwise scales should be empty"
        assert output_rowwise.dtype == torch.float8_e4m3fn
        assert scales_rowwise.dtype == torch.float8_e8m0fnu
        assert output_rowwise.device == gated_input.device

    if colwise:
        ref_q, ref_s = mxfp8_quantize_2d_32x1_cutedsl(reference, scaling_mode="rceil")
        assert output_colwise.shape == (M, expected_width), f"{tag}: colwise width"
        assert output_colwise.stride() == (1, M), (
            f"{tag}: colwise qdata must have stride (1, M), got {output_colwise.stride()}"
        )
        _assert_qdata_matches(
            output_colwise, ref_q, f"{tag}: colwise qdata", exact=not is_backward
        )
        _assert_identical(scales_colwise, ref_s, f"{tag}: colwise scales")
        # The 32x1 quantizer returns flat scales; ours must match that layout.
        assert scales_colwise.ndim == 1, f"{tag}: colwise scales should be flat"
    else:
        assert output_colwise.numel() == 0, f"{tag}: colwise output should be empty"
        assert scales_colwise.numel() == 0, f"{tag}: colwise scales should be empty"
        assert output_colwise.dtype == torch.float8_e4m3fn
        assert scales_colwise.dtype == torch.float8_e8m0fnu
        assert output_colwise.device == gated_input.device


def test_swiglu_mxfp8_special_values():
    """Zero, tiny, Inf, and NaN scale blocks must match the standalone
    quantizers bitwise: a clamped-to-0 scale byte pairs with the 2^126
    reciprocal (so tiny blocks don't collapse to zero codes), Inf amax
    saturates the scale byte to 254, and NaN elements come through as E4M3
    NaN. Special rows pin gate to 20 (sigmoid saturates identically under
    both lowerings) and use power-of-two `up` values so the h tensor is
    bitwise stable and any difference isolates the scale path."""
    torch.manual_seed(7)
    M, K = 128, 128
    gated_input = torch.randn(M, 2 * K, device="cuda", dtype=torch.bfloat16)
    gate = gated_input[:, :K]
    up = gated_input[:, K:]

    gate[:5, :] = 20.0
    up[:5, :] = 0.5
    up[0, :64] = 0.0  # all-zero rowwise scale blocks
    up[1, :64] = 2.0**-125  # tiny amax: scale byte clamps to 0
    up[2, 0] = float("inf")  # Inf amax rowwise block
    up[3, 0] = float("inf")
    up[3, 1] = -3.0e38  # h overflows f32 to -Inf: mixed-sign Inf block
    gate[4, 0] = float("inf")
    up[4, 0] = 0.0  # silu(inf) * 0 -> NaN element
    gate[:32, 100:103] = 20.0
    up[:32, 100] = 2.0**-125  # tiny colwise block
    up[:32, 101] = 0.0  # zero colwise block
    gate[:32, 102] = float("inf")
    up[:32, 102] = 0.0  # all-NaN colwise block

    reference = _eager_reference(gated_input, None)
    out_r, out_c, s_r, s_c = swiglu_mxfp8_forward(
        gated_input, rowwise=True, colwise=True
    )
    ref_qr, ref_sr = mxfp8_quantize_2d_1x32_cutedsl(reference, scaling_mode="rceil")
    ref_qc, ref_sc = mxfp8_quantize_2d_32x1_cutedsl(reference, scaling_mode="rceil")

    def _assert_bitwise(actual, ref, msg):
        # assert_close treats NaN as unequal even at zero tolerance, and NaN
        # codes are expected content here, so compare raw bytes.
        _assert_layout(actual, ref, msg)
        torch.testing.assert_close(
            actual.contiguous().view(torch.uint8),
            ref.contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
            msg=msg,
        )

    _assert_bitwise(out_r, ref_qr, "special rowwise qdata")
    _assert_bitwise(s_r, ref_sr, "special rowwise scales")
    _assert_bitwise(out_c, ref_qc, "special colwise qdata")
    _assert_bitwise(s_c, ref_sc, "special colwise scales")

    # The tiny blocks must not collapse to zero codes: q = v * 2^126, not v * 1.
    assert bool(out_r.view(torch.uint8)[1, :64].ne(0).all()), (
        "tiny-amax block quantized to zero codes"
    )
    # NaN elements survive as E4M3 NaN.
    assert int(out_r.view(torch.uint8)[4, 0]) & 0x7F == 0x7F, (
        "NaN element did not map to the E4M3 NaN code"
    )


@pytest.mark.parametrize("is_backward", _DIRECTIONS)
def test_swiglu_mxfp8_compile(is_backward):
    """Fullgraph compile exercises the fake impls and the fixed operator schema."""
    torch.manual_seed(42)
    M, K = 512, 2048
    gated_input = torch.randn(M, 2 * K, device="cuda", dtype=torch.bfloat16)
    grad_h = (
        torch.randn(M, K, device="cuda", dtype=torch.bfloat16) if is_backward else None
    )

    if is_backward:

        def fn(gated_input, grad_h):
            return swiglu_mxfp8_backward(
                grad_h, gated_input, rowwise=True, colwise=True
            )
    else:

        def fn(gated_input, grad_h):
            return swiglu_mxfp8_forward(gated_input, rowwise=True, colwise=True)

    try:
        expected = fn(gated_input, grad_h)
        actual = torch.compile(fn, fullgraph=True)(gated_input, grad_h)
        assert len(actual) == 4
        for i, (a, e) in enumerate(zip(actual, expected)):
            # Same kernel on both sides, so this must be bitwise identical; it
            # also pins the fake's shapes and strides to the real ones.
            _assert_layout(a, e, f"compile bwd={is_backward} output {i}")
            torch.testing.assert_close(a, e, rtol=0, atol=0)
    finally:
        torch._dynamo.reset()
