# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Correctness tests for the unified SwiGLU + MXFP8 kernel (cutedsl_swiglu_mxfp8).

Compares the fused kernel against an eager baseline:
  forward:  h = silu(gate) * up  →  mxfp8_quantize_2d_{1x32,32x1}_cutedsl(h)
  backward: dgate/dup via PyTorch grads  →  same quantize functions

Tests cover rowwise (1x32), colwise (32x1), and bidirectional for both
forward and backward paths.
"""

import pytest
import torch
import torch.nn.functional as F

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10):
    pytest.skip("Requires CUDA SM 10.x (Blackwell)", allow_module_level=True)

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.kernels.mxfp8 import (
    mxfp8_quantize_2d_1x32_cutedsl,
    mxfp8_quantize_2d_32x1_cutedsl,
)
from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_swiglu_mxfp8 import (
    swiglu_mxfp8_backward_bidirectional,
    swiglu_mxfp8_backward_col,
    swiglu_mxfp8_backward_row,
    swiglu_mxfp8_forward_bidirectional,
    swiglu_mxfp8_forward_col,
    swiglu_mxfp8_forward_row,
)

# Minimum acceptable SQNR (dB) between fused kernel and eager baseline.
# The fused kernel uses _rcp2 for sigmoid which may differ by <= 1 ULP from
# PyTorch's sigmoid, so we tolerate minor numerical differences.
_MIN_SQNR = 20.0

_SHAPES = [
    (128, 128),
    (256, 256),
    (512, 2048),
    (1024, 7168),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gated_input(M, K):
    return torch.randn(M, 2 * K, device="cuda", dtype=torch.bfloat16)


def _make_grad_h(M, K):
    return torch.randn(M, K, device="cuda", dtype=torch.bfloat16)


def _eager_forward_h(gated_input):
    """Compute SwiGLU in f32 and round to BF16 to match the kernel's _bf16x2 path."""
    M, two_k = gated_input.shape
    K = two_k // 2
    gate = gated_input[:, :K].float()
    up = gated_input[:, K:].float()
    return (F.silu(gate) * up).bfloat16()


def _eager_backward_grads(grad_h, gated_input):
    """Compute dgate and dUp in f32, rounded to BF16."""
    M, two_k = gated_input.shape
    K = two_k // 2
    gate = gated_input[:, :K].float()
    up = gated_input[:, K:].float()
    grad_h_f = grad_h.float()
    sigmoid_gate = torch.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    d_silu = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    dgate = (grad_h_f * up * d_silu).bfloat16()
    dup = (grad_h_f * silu_gate).bfloat16()
    return dgate, dup


def _sqnr(actual, ref, msg):
    sqnr = compute_error(ref.float(), actual.float())
    assert sqnr >= _MIN_SQNR, f"{msg}: SQNR {sqnr:.2f} dB < {_MIN_SQNR:.2f} dB"


def _check_scales(kernel_s, ref_s, msg):
    """Compare scale tensors; fall back to SQNR if shapes differ."""
    if kernel_s.shape == ref_s.shape:
        _sqnr(kernel_s.view(torch.uint8).float(), ref_s.view(torch.uint8).float(), msg)
    else:
        # Shapes may differ due to layout conventions; just verify non-trivial values.
        assert kernel_s.numel() > 0, f"{msg}: scale tensor is empty"


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M,K", _SHAPES)
def test_swiglu_mxfp8_forward_rowwise(M, K):
    torch.manual_seed(42)
    gated = _make_gated_input(M, K)
    h = _eager_forward_h(gated)

    ref_q, ref_s = mxfp8_quantize_2d_1x32_cutedsl(h, scaling_mode="rceil")
    ker_q, ker_s = swiglu_mxfp8_forward_row(gated)

    assert ker_q.shape == ref_q.shape, f"qdata shape mismatch: {ker_q.shape} vs {ref_q.shape}"
    assert ker_s.shape == ref_s.shape, f"scales shape mismatch: {ker_s.shape} vs {ref_s.shape}"

    _sqnr(ker_q, ref_q, f"forward rowwise qdata M={M} K={K}")
    torch.testing.assert_close(ker_s, ref_s, rtol=0, atol=0,
                               msg=f"forward rowwise scales M={M} K={K}")


@pytest.mark.parametrize("M,K", _SHAPES)
def test_swiglu_mxfp8_forward_colwise(M, K):
    torch.manual_seed(42)
    gated = _make_gated_input(M, K)
    h = _eager_forward_h(gated)

    ref_q, ref_s = mxfp8_quantize_2d_32x1_cutedsl(h, scaling_mode="rceil")
    ker_q, ker_s = swiglu_mxfp8_forward_col(gated)

    assert ker_q.shape == ref_q.shape, f"qdata shape mismatch: {ker_q.shape} vs {ref_q.shape}"
    assert ker_s.shape == ref_s.shape, f"scales shape mismatch: {ker_s.shape} vs {ref_s.shape}"

    _sqnr(ker_q, ref_q, f"forward colwise qdata M={M} K={K}")
    torch.testing.assert_close(ker_s, ref_s, rtol=0, atol=0,
                               msg=f"forward colwise scales M={M} K={K}")


@pytest.mark.parametrize("M,K", _SHAPES)
def test_swiglu_mxfp8_forward_bidirectional(M, K):
    torch.manual_seed(42)
    gated = _make_gated_input(M, K)
    h = _eager_forward_h(gated)

    ref_rq, ref_rs = mxfp8_quantize_2d_1x32_cutedsl(h, scaling_mode="rceil")
    ref_cq, ref_cs = mxfp8_quantize_2d_32x1_cutedsl(h, scaling_mode="rceil")

    ker_rq, ker_rs, ker_cq, ker_cs = swiglu_mxfp8_forward_bidirectional(gated)

    _sqnr(ker_rq, ref_rq, f"fwd bidi rowwise qdata M={M} K={K}")
    torch.testing.assert_close(ker_rs, ref_rs, rtol=0, atol=0,
                               msg=f"fwd bidi rowwise scales M={M} K={K}")

    _sqnr(ker_cq, ref_cq, f"fwd bidi colwise qdata M={M} K={K}")
    torch.testing.assert_close(ker_cs, ref_cs, rtol=0, atol=0,
                               msg=f"fwd bidi colwise scales M={M} K={K}")


# ---------------------------------------------------------------------------
# Backward tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M,K", _SHAPES)
def test_swiglu_mxfp8_backward_rowwise(M, K):
    torch.manual_seed(42)
    gated = _make_gated_input(M, K)
    grad_h = _make_grad_h(M, K)

    dgate, dup = _eager_backward_grads(grad_h, gated)
    # The kernel concatenates [dGate | dUp] in a single [M, 2K] output.
    ref_rq, ref_rs = mxfp8_quantize_2d_1x32_cutedsl(
        torch.cat([dgate, dup], dim=1), scaling_mode="rceil"
    )

    ker_rq, ker_rs = swiglu_mxfp8_backward_row(grad_h, gated)

    assert ker_rq.shape == (M, 2 * K), f"qdata shape: {ker_rq.shape}"
    assert ker_rq.shape == ref_rq.shape

    _sqnr(ker_rq[:, :K], ref_rq[:, :K], f"bwd rowwise dGate qdata M={M} K={K}")
    _sqnr(ker_rq[:, K:], ref_rq[:, K:], f"bwd rowwise dUp qdata M={M} K={K}")
    torch.testing.assert_close(ker_rs, ref_rs, rtol=0, atol=0,
                               msg=f"bwd rowwise scales M={M} K={K}")


@pytest.mark.parametrize("M,K", _SHAPES)
def test_swiglu_mxfp8_backward_colwise(M, K):
    torch.manual_seed(42)
    gated = _make_gated_input(M, K)
    grad_h = _make_grad_h(M, K)

    dgate, dup = _eager_backward_grads(grad_h, gated)
    ref_cq, ref_cs = mxfp8_quantize_2d_32x1_cutedsl(
        torch.cat([dgate, dup], dim=1), scaling_mode="rceil"
    )

    ker_cq, ker_cs = swiglu_mxfp8_backward_col(grad_h, gated)

    assert ker_cq.shape == (M, 2 * K)
    _sqnr(ker_cq, ref_cq, f"bwd colwise qdata M={M} K={K}")
    torch.testing.assert_close(ker_cs, ref_cs, rtol=0, atol=0,
                               msg=f"bwd colwise scales M={M} K={K}")


@pytest.mark.parametrize("M,K", _SHAPES)
def test_swiglu_mxfp8_backward_bidirectional(M, K):
    torch.manual_seed(42)
    gated = _make_gated_input(M, K)
    grad_h = _make_grad_h(M, K)

    dgate, dup = _eager_backward_grads(grad_h, gated)
    combined = torch.cat([dgate, dup], dim=1)
    ref_rq, ref_rs = mxfp8_quantize_2d_1x32_cutedsl(combined, scaling_mode="rceil")
    ref_cq, ref_cs = mxfp8_quantize_2d_32x1_cutedsl(combined, scaling_mode="rceil")

    ker_rq, ker_rs, ker_cq, ker_cs = swiglu_mxfp8_backward_bidirectional(grad_h, gated)

    _sqnr(ker_rq[:, :K], ref_rq[:, :K], f"bwd bidi dGate rowwise qdata M={M} K={K}")
    _sqnr(ker_rq[:, K:], ref_rq[:, K:], f"bwd bidi dUp rowwise qdata M={M} K={K}")
    torch.testing.assert_close(ker_rs, ref_rs, rtol=0, atol=0,
                               msg=f"bwd bidi rowwise scales M={M} K={K}")

    _sqnr(ker_cq, ref_cq, f"bwd bidi colwise qdata M={M} K={K}")
    torch.testing.assert_close(ker_cs, ref_cs, rtol=0, atol=0,
                               msg=f"bwd bidi colwise scales M={M} K={K}")
