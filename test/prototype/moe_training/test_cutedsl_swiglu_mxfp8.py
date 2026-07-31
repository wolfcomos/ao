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

Every output must match BITWISE -- E4M3 data and E8M0 scales, both directions,
all layouts. The reference below mirrors the kernel's multiply order exactly:
dUp is `grad_h * (gate * sigmoid)`, forming silu first, because
`grad_h * gate * sigmoid` associates as `(grad_h * gate) * sigmoid` and rounds
differently in fp32.
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
    # grad_h * (gate * sigmoid), matching the kernel; see the module docstring.
    dup = (grad_h_f * (gate * sigmoid_gate)).bfloat16()
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
        _assert_identical(output_rowwise, ref_q, f"{tag}: rowwise qdata")
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
        _assert_identical(output_colwise, ref_q, f"{tag}: colwise qdata")
        _assert_identical(scales_colwise, ref_s, f"{tag}: colwise scales")
        # The 32x1 quantizer returns flat scales; ours must match that layout.
        assert scales_colwise.ndim == 1, f"{tag}: colwise scales should be flat"
    else:
        assert output_colwise.numel() == 0, f"{tag}: colwise output should be empty"
        assert scales_colwise.numel() == 0, f"{tag}: colwise scales should be empty"
        assert output_colwise.dtype == torch.float8_e4m3fn
        assert scales_colwise.dtype == torch.float8_e8m0fnu
        assert output_colwise.device == gated_input.device


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
