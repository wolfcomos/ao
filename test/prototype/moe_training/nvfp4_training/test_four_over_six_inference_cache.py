# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

from torchao.prototype.moe_training.config import NVFP4FourOverSixTrainingOpConfig
from torchao.prototype.moe_training.nvfp4_training.four_over_six_grouped import (
    _blocked_expert_scales,
    _quantize_expert_weights,
    four_over_six_grouped_mm,
    four_over_six_grouped_mm_prequantized,
    four_over_six_quantize_expert_weights,
)
from torchao.prototype.moe_training.utils import _quantize_then_scaled_grouped_mm
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.utils import is_sm_at_least_100, torch_version_at_least

_skip_no_sm100 = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and is_sm_at_least_100()
        and torch_version_at_least("2.10.0")
    ),
    reason="requires SM100+ and PyTorch 2.10+ (FP4 scaled_grouped_mm)",
)

_GROUP_SIZES = [128, 384, 256]


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
@pytest.mark.parametrize("row_scaled_activation", [False, True])
@pytest.mark.parametrize("weight_block", ["16x16", "1x16"])
@pytest.mark.parametrize("err_mode", ["mae", "mse"])
@pytest.mark.parametrize("e4m3_scale_bound", [256, 448])
def test_prequantized_matches_grouped_mm(
    row_scaled_activation, weight_block, err_mode, e4m3_scale_bound
):
    """The prequantized inference forward is bitwise == the differentiable op."""
    A, B, offs = _make_grouped_inputs(_GROUP_SIZES, K=256, N=128)
    reference = four_over_six_grouped_mm(
        A,
        B,
        offs,
        err_mode=err_mode,
        e4m3_scale_bound=e4m3_scale_bound,
        row_scaled_activation=row_scaled_activation,
        weight_block=weight_block,
    )
    with torch.no_grad():
        quantized_weight = four_over_six_quantize_expert_weights(
            B,
            err_mode=err_mode,
            e4m3_scale_bound=e4m3_scale_bound,
            weight_block=weight_block,
        )
        output = four_over_six_grouped_mm_prequantized(
            A,
            quantized_weight,
            offs,
            row_scaled_activation=row_scaled_activation,
        )
    torch.testing.assert_close(output, reference, atol=0, rtol=0)


@_skip_no_sm100
@pytest.mark.parametrize("row_scaled_activation", [False, True])
@pytest.mark.parametrize("tail_rows", [256, 1792])
@pytest.mark.parametrize(
    "group_sizes",
    [_GROUP_SIZES, [128, 384, 0], [128, 0, 384]],
    ids=["full", "empty-last", "empty-middle"],
)
def test_prequantized_over_allocated_matches_dispatcher(
    row_scaled_activation, tail_rows, group_sizes
):
    """Over-allocated zero tails match the eager dispatcher path, bitwise.

    The eager dispatcher slices A at a host read of offs[-1] and zero-extends
    the output; the prequantized forward must reproduce that — including with
    an empty final group, which the on-device extension of the last GEMM
    group resurrects over the zero tail.
    """
    A, B, offs = _make_grouped_inputs(group_sizes, K=256, N=128)
    A_over = torch.cat(
        [A, torch.zeros(tail_rows, A.shape[1], dtype=A.dtype, device=A.device)]
    )
    config = NVFP4FourOverSixTrainingOpConfig(
        row_scaled_activation=row_scaled_activation
    )
    reference = _quantize_then_scaled_grouped_mm(
        A_over, B.transpose(-2, -1), config=config, offs=offs
    )
    with torch.no_grad():
        quantized_weight = four_over_six_quantize_expert_weights(B)
        output = four_over_six_grouped_mm_prequantized(
            A_over, quantized_weight, offs, row_scaled_activation=row_scaled_activation
        )
    assert output.shape == reference.shape
    torch.testing.assert_close(output, reference, atol=0, rtol=0)


@_skip_no_sm100
def test_prequantized_row_scaled_tolerates_garbage_tail():
    """Row-scaled mode masks tail amaxes, so non-zero tail content past
    offs[-1] (a dispatcher that recycles its over-allocation without
    zero-filling) neither perturbs the real rows nor leaks into the output."""
    tail_rows = 256
    A, B, offs = _make_grouped_inputs(_GROUP_SIZES, K=256, N=128)
    garbage = torch.randn(tail_rows, A.shape[1], dtype=A.dtype, device=A.device) * 100
    A_over = torch.cat([A, garbage])
    with torch.no_grad():
        quantized_weight = four_over_six_quantize_expert_weights(B)
        reference = torch.nn.functional.pad(
            four_over_six_grouped_mm_prequantized(
                A, quantized_weight, offs, row_scaled_activation=True
            ),
            (0, 0, 0, tail_rows),
        )
        output = four_over_six_grouped_mm_prequantized(
            A_over, quantized_weight, offs, row_scaled_activation=True
        )
    torch.testing.assert_close(output, reference, atol=0, rtol=0)


@_skip_no_sm100
def test_prequantized_rejects_grad_mode():
    """A forward on cached quantized weights has no weight-gradient path."""
    A, B, offs = _make_grouped_inputs(_GROUP_SIZES, K=256, N=128)
    with torch.no_grad():
        quantized_weight = four_over_six_quantize_expert_weights(B)
    with pytest.raises(RuntimeError, match="inference-only"):
        four_over_six_grouped_mm_prequantized(A, quantized_weight, offs)


@_skip_no_sm100
@pytest.mark.parametrize("weight_block", ["16x16", "1x16"])
def test_blocked_expert_scales_equal_per_expert_stack(weight_block):
    """One to_blocked over row-flattened expert scales == the per-expert stack.

    The per-tensor forward historically built its GEMM scale operand as a
    per-expert to_blocked stack; the cached form uses the flattened single
    call, which must be bitwise identical while N % 128 == 0.
    """
    _, B, _ = _make_grouped_inputs(_GROUP_SIZES, K=256, N=128)
    weight_amax = B.abs().amax(dim=(1, 2)).to(torch.float32)
    _, w_scales = _quantize_expert_weights(B, weight_amax, weight_block, "mae", 256)
    stacked = torch.stack([to_blocked(s) for s in w_scales]).reshape(B.shape[0], -1)
    torch.testing.assert_close(
        _blocked_expert_scales(w_scales).view(torch.uint8),
        stacked.view(torch.uint8),
        atol=0,
        rtol=0,
    )
