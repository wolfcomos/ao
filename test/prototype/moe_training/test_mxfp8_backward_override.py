# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchao.utils import is_sm_version

if not (torch.cuda.is_available() and is_sm_version(10, 0)):
    pytest.skip(
        "MXFP8 grouped GEMM backward overrides require SM100",
        allow_module_level=True,
    )

pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.prototype.moe_training.config import MXFP8TrainingOpConfig
from torchao.prototype.moe_training.mxfp8_grouped_mm import (
    _SM100_KERNELS_AVAILABLE,
    _compute_wgrad_sm100,
    _to_mxfp8_then_scaled_grouped_mm,
)
from torchao.prototype.moe_training.utils import _quantize_then_scaled_grouped_mm
from torchao.prototype.mx_formats.config import ScaleCalculationMode

if not _SM100_KERNELS_AVAILABLE:
    pytest.skip(
        "SM100 MXFP8 kernels (CUDA + Triton) unavailable",
        allow_module_level=True,
    )

# Group sizes must be multiples of 128 for the CuTe DSL 1x32 quantizer used
# by the stock forward, so every test allocates 128 rows per expert.
_ROWS_PER_EXPERT = 128


def _make_inputs(num_experts=4, K=256, N=512, seed=0, tail_rows=0):
    torch.manual_seed(seed)
    M = num_experts * _ROWS_PER_EXPERT
    A = torch.randn(M + tail_rows, K, dtype=torch.bfloat16, device="cuda")
    if tail_rows:
        A[M:] = 0
    B = torch.randn(num_experts, N, K, dtype=torch.bfloat16, device="cuda") * 0.1
    offs = torch.arange(
        _ROWS_PER_EXPERT,
        M + 1,
        _ROWS_PER_EXPERT,
        dtype=torch.int32,
        device="cuda",
    )
    return A, B, offs


def _reference_grads(A, B, offs, grad_output):
    """bf16 reference gradients from the same two-grouped-GEMM formulation."""
    grad_output = grad_output.contiguous()
    ref_grad_A = torch._grouped_mm(grad_output, B, offs=offs, out_dtype=torch.bfloat16)
    ref_grad_B = torch._grouped_mm(
        grad_output.transpose(-2, -1), A, offs=offs, out_dtype=torch.bfloat16
    )
    return ref_grad_A, ref_grad_B


def _run_override(A, B, offs, backward_override, grad_output=None):
    """Run forward+backward through the op; returns (out, A.grad, B.grad)."""
    A = A.detach().clone().requires_grad_(True)
    B = B.detach().clone().requires_grad_(True)
    out = _to_mxfp8_then_scaled_grouped_mm(
        A,
        B.transpose(-2, -1),
        offs=offs,
        backward_override=backward_override,
    )
    if grad_output is None:
        grad_output = torch.randn_like(out)
    out.backward(grad_output)
    return out, A.grad, B.grad


def test_forward_bitwise_identical_across_overrides():
    """The override must not change the forward: the Triton dim0 quantizer is
    bitwise identical to the CuTe DSL 1x32 rceil kernel, so all three arms
    consume identical GEMM operands."""
    A, B, offs = _make_inputs()
    outputs = {}
    for override in (None, "high_precision", "dequantized"):
        out = _to_mxfp8_then_scaled_grouped_mm(
            A.detach(),
            B.detach().transpose(-2, -1),
            offs=offs,
            backward_override=override,
        )
        outputs[override] = out
    assert torch.equal(outputs[None], outputs["high_precision"])
    assert torch.equal(outputs[None], outputs["dequantized"])
    # "quantized" is an accepted alias for the default backward.
    out_quantized = _to_mxfp8_then_scaled_grouped_mm(
        A.detach(),
        B.detach().transpose(-2, -1),
        offs=offs,
        backward_override="quantized",
    )
    assert torch.equal(outputs[None], out_quantized)


def test_high_precision_backward_matches_grouped_mm_reference():
    A, B, offs = _make_inputs()
    grad_output = torch.randn(
        A.shape[0], B.shape[1], dtype=torch.bfloat16, device="cuda"
    )
    _, grad_A, grad_B = _run_override(A, B, offs, "high_precision", grad_output)
    ref_grad_A, ref_grad_B = _reference_grads(A, B, offs, grad_output)
    # Same two torch._grouped_mm calls on the same operands: bitwise.
    assert torch.equal(grad_A, ref_grad_A)
    assert torch.equal(grad_B, ref_grad_B)


def test_dequantized_backward_close_to_reference_and_distinct():
    A, B, offs = _make_inputs()
    grad_output = torch.randn(
        A.shape[0], B.shape[1], dtype=torch.bfloat16, device="cuda"
    )
    _, grad_A, grad_B = _run_override(A, B, offs, "dequantized", grad_output)
    assert torch.isfinite(grad_A).all()
    assert torch.isfinite(grad_B).all()

    ref_grad_A, ref_grad_B = _reference_grads(A, B, offs, grad_output)
    # Loose tolerance: the operands are round-tripped through MXFP8 (e4m3
    # data with e8m0 block scales), so gradients carry one quantization
    # error per operand — observed ~5% max relative error at these shapes;
    # 0.15 is a deliberately loose bound to avoid seed sensitivity.
    for grad, ref in ((grad_A, ref_grad_A), (grad_B, ref_grad_B)):
        rel_err = ((grad - ref).abs().max() / ref.abs().max()).item()
        assert rel_err < 0.15, f"relative error too large: {rel_err}"

    # Sanity: the dequantized backward consumed quantized operands, so its
    # gradients differ from the high-precision ones.
    _, hp_grad_A, hp_grad_B = _run_override(A, B, offs, "high_precision", grad_output)
    assert not torch.equal(grad_A, hp_grad_A)
    assert not torch.equal(grad_B, hp_grad_B)


@pytest.mark.parametrize("backward_override", ["high_precision", "dequantized"])
def test_tail_rows_past_final_offset_tolerated(backward_override):
    """Token dispatchers may over-allocate activation rows past offs[-1];
    the grouped GEMMs only read rows covered by the offsets."""
    tail_rows = 128
    A, B, offs = _make_inputs(tail_rows=tail_rows)
    M_logical = int(offs[-1])
    grad_output = torch.randn(
        A.shape[0], B.shape[1], dtype=torch.bfloat16, device="cuda"
    )
    out, grad_A, grad_B = _run_override(A, B, offs, backward_override, grad_output)
    assert out.shape[0] == A.shape[0]
    ref_grad_A, ref_grad_B = _reference_grads(
        A[:M_logical], B, offs, grad_output[:M_logical]
    )
    # Rows past offs[-1] belong to no group and may be uninitialized, so
    # only the logical rows are checked.
    assert torch.isfinite(grad_A[:M_logical]).all()
    assert torch.isfinite(grad_B).all()
    if backward_override == "high_precision":
        assert torch.equal(grad_A[:M_logical], ref_grad_A)
        assert torch.equal(grad_B, ref_grad_B)
    else:
        for grad, ref in ((grad_A[:M_logical], ref_grad_A), (grad_B, ref_grad_B)):
            rel_err = ((grad - ref).abs().max() / ref.abs().max()).item()
            assert rel_err < 0.15, f"relative error too large: {rel_err}"


def test_noncontiguous_grad_output_wgrad():
    """The CUDA dim1 cast in the quantized wgrad requires contiguous inputs;
    expanded (stride-0) and other non-contiguous gradient views must not
    crash it."""
    A, B, offs = _make_inputs()
    N = B.shape[1]
    M = A.shape[0]

    # Direct wgrad call with an expanded (stride-0) grad_output.
    grad_output_expanded = torch.randn(
        M, 1, dtype=torch.bfloat16, device="cuda"
    ).expand(M, N)
    grad_weight_t = _compute_wgrad_sm100(
        grad_output_expanded,
        A,
        offs,
        32,
        torch.bfloat16,
        ScaleCalculationMode.RCEIL,
        wgrad_with_hp=False,
    )
    assert torch.isfinite(grad_weight_t).all()

    # End-to-end: sum().backward() feeds an expanded ones gradient into the
    # default quantized backward.
    A_leaf = A.detach().clone().requires_grad_(True)
    B_leaf = B.detach().clone().requires_grad_(True)
    out = _to_mxfp8_then_scaled_grouped_mm(A_leaf, B_leaf.transpose(-2, -1), offs=offs)
    out.sum().backward()
    assert torch.isfinite(A_leaf.grad).all()
    assert torch.isfinite(B_leaf.grad).all()


def test_invalid_backward_override_rejected():
    A, B, offs = _make_inputs(num_experts=1, K=128, N=128)
    with pytest.raises(AssertionError, match="backward_override"):
        _to_mxfp8_then_scaled_grouped_mm(
            A.detach(),
            B.detach().transpose(-2, -1),
            offs=offs,
            backward_override="hp",
        )


def test_config_threads_backward_override():
    """backward_override must participate in config equality/hashing (no
    silent aliasing) and reach the grouped GEMM through the op config."""
    default_config = MXFP8TrainingOpConfig()
    hp_config = MXFP8TrainingOpConfig(backward_override="high_precision")
    assert default_config != hp_config
    assert hash(default_config) != hash(hp_config)
    assert hp_config == MXFP8TrainingOpConfig(backward_override="high_precision")

    A, B, offs = _make_inputs()
    grad_output = torch.randn(
        A.shape[0], B.shape[1], dtype=torch.bfloat16, device="cuda"
    )
    A_leaf = A.detach().clone().requires_grad_(True)
    B_leaf = B.detach().clone().requires_grad_(True)
    out = _quantize_then_scaled_grouped_mm(
        A_leaf, B_leaf.transpose(-2, -1), config=hp_config, offs=offs
    )
    out.backward(grad_output)
    ref_grad_A, ref_grad_B = _reference_grads(A, B, offs, grad_output)
    assert torch.equal(A_leaf.grad, ref_grad_A)
    assert torch.equal(B_leaf.grad, ref_grad_B)
