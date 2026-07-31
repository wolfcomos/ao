# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Correctness tests for the unified SwiGLU + MXFP8 kernel (cutedsl_swiglu_mxfp8).

Every case goes through the single public API, `swiglu_mxfp8_quantize`, and is
compared against the equivalent eager sequence:
  forward:  h = silu(gate) * up          -> mxfp8_quantize_2d_{1x32,32x1}_cutedsl(h)
  backward: dGate, dUp via SwiGLU grads  -> the same quantize functions

Agreement is near-exact:

  * E8M0 scales match BITWISE, in every direction and layout.
  * Forward E4M3 data matches BITWISE.
  * Backward E4M3 data may differ in a handful of codes, by at most one code
    each. The kernel evaluates sigmoid with rcp.approx plus two Markstein
    refinement steps and fuses the d_silu product differently than eager
    PyTorch, so a result can land 1 ULP away in bfloat16 and tip a quantized
    code. Measured rate over 4 seeds x 4 shapes is ~3.6e-7 of elements (48 of
    134M, max gap 1 code, SQNR 110 dB); _MAX_DIFFERING_FRACTION is set two
    orders of magnitude above that so a systematic off-by-one still fails.
"""

import pytest
import torch
import torch.nn.functional as F

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10):
    pytest.skip("Requires CUDA SM 10.x (Blackwell)", allow_module_level=True)

from torchao.prototype.moe_training.kernels.mxfp8 import (
    mxfp8_quantize_2d_1x32_cutedsl,
    mxfp8_quantize_2d_32x1_cutedsl,
    swiglu_mxfp8_quantize,
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

# Only backward E4M3 data is allowed to differ at all; see the module docstring.
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
    sigmoid_gate = torch.sigmoid(gate)
    d_silu = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    dgate = (grad_h_f * up * d_silu).bfloat16()
    dup = (grad_h_f * gate * sigmoid_gate).bfloat16()
    return torch.cat([dgate, dup], dim=1)


def _assert_layout(actual, ref, msg):
    assert actual.shape == ref.shape, f"{msg}: shape {actual.shape} vs {ref.shape}"
    assert actual.stride() == ref.stride(), (
        f"{msg}: stride {actual.stride()} vs {ref.stride()}"
    )
    assert actual.dtype == ref.dtype, f"{msg}: dtype {actual.dtype} vs {ref.dtype}"


def _assert_scales_identical(actual, ref, msg):
    """Scale bytes must match exactly."""
    _assert_layout(actual, ref, msg)
    torch.testing.assert_close(actual, ref, rtol=0, atol=0, msg=msg)


def _assert_qdata_matches(actual, ref, msg, exact):
    """Exact FP8-bit comparison, or a bounded 1-code tolerance for backward."""
    _assert_layout(actual, ref, msg)
    if exact:
        torch.testing.assert_close(actual, ref, rtol=0, atol=0, msg=msg)
        return
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
def test_swiglu_mxfp8_quantize(M, K, is_backward, rowwise, colwise):
    torch.manual_seed(42)
    gated_input = torch.randn(M, 2 * K, device="cuda", dtype=torch.bfloat16)
    grad_h = (
        torch.randn(M, K, device="cuda", dtype=torch.bfloat16) if is_backward else None
    )
    tag = f"bwd={is_backward} rowwise={rowwise} colwise={colwise} M={M} K={K}"

    outputs = swiglu_mxfp8_quantize(
        gated_input, grad_h, rowwise=rowwise, colwise=colwise
    )

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
        _assert_scales_identical(scales_rowwise, ref_s, f"{tag}: rowwise scales")
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
        _assert_scales_identical(scales_colwise, ref_s, f"{tag}: colwise scales")
        # The 32x1 quantizer returns flat scales; ours must match that layout.
        assert scales_colwise.ndim == 1, f"{tag}: colwise scales should be flat"
    else:
        assert output_colwise.numel() == 0, f"{tag}: colwise output should be empty"
        assert scales_colwise.numel() == 0, f"{tag}: colwise scales should be empty"
        assert output_colwise.dtype == torch.float8_e4m3fn
        assert scales_colwise.dtype == torch.float8_e8m0fnu
        assert output_colwise.device == gated_input.device


def test_swiglu_mxfp8_quantize_requires_a_direction():
    gated_input = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="rowwise or colwise"):
        swiglu_mxfp8_quantize(gated_input, rowwise=False, colwise=False)


@pytest.mark.parametrize("is_backward", _DIRECTIONS)
def test_swiglu_mxfp8_quantize_compile(is_backward):
    """Fullgraph compile exercises the fake impls and the fixed operator schema."""
    torch.manual_seed(42)
    M, K = 512, 2048
    gated_input = torch.randn(M, 2 * K, device="cuda", dtype=torch.bfloat16)
    grad_h = (
        torch.randn(M, K, device="cuda", dtype=torch.bfloat16) if is_backward else None
    )

    def fn(gated_input, grad_h):
        return swiglu_mxfp8_quantize(gated_input, grad_h, rowwise=True, colwise=True)

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
