# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Differentiable NVFP4 four-over-six grouped GEMM for MoE training.

Grouped counterpart of ``four_over_six_mm`` for routed-expert layers:
``A`` holds token groups packed along dim 0, ``B`` holds one weight matrix
per expert, and ``offs`` marks each group's end row. The recipe follows
TransformerEngine's GroupedLinear semantics, where every group is quantized
as its own tensor:

* per-tensor activations (the default): each token group gets its own
  global scale from that group's amax. The group amaxes are expanded to a
  per-row amax vector so the whole packed tensor quantizes in one
  ``four_over_six_quantize`` call — bitwise identical to quantizing each
  group separately, because the quantizer derives every row's scale chain
  from that row's amax entry. The forward GEMM is one
  ``F.scaled_grouped_mm`` with per-group second-level scales.
* row-scaled activations: one global scale per token row. TransformerEngine
  has no fused row-scaled NVFP4 grouped GEMM — its ``general_grouped_gemm``
  runs a per-group loop of dense GEMMs — so the forward here is the same
  loop over the dense four-over-six GEMM (FP32 output scaled by the raw
  per-row amaxes, then the bf16 cast).

Weights always quantize per expert with per-tensor scales
(``weight_block`` selects 16x16 tiles or 1x16 blocks, as in the dense op).

Gradients never quantize with four-over-six, and TransformerEngine rejects
four-over-six group quantization outright, so the grouped backward supports
only the high-precision and dequantized overrides of ``four_over_six_mm``:

* ``"high_precision"`` (the default): bf16 grouped GEMMs on the saved
  original operands;
* ``"dequantized"``: bf16 grouped GEMMs on dequantizations of the rowwise
  operands the forward consumed — the RL train/inference-consistency mode.

``"quantized"`` raises. Requires K % 128 == 0 and N % 128 == 0; token
groups must be 128-row aligned unless ``pad_token_groups_for_grouped_mm``
is set, which zero-pads each group to the next 128 multiple before
quantization (zero rows quantize to zero codes and are sliced away from the
output).
"""

from typing import Optional

import torch
import torch.nn.functional as F

from torchao.prototype.moe_training.nvfp4_training.four_over_six import (
    FP4_E2M1_MAX,
    _global_decode_scale,
    _scaled_mm_nvfp4,
    four_over_six_dequantize,
    four_over_six_quantize,
)
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
    _DEVICE_ASSERTS,
)
from torchao.prototype.moe_training.utils import (
    conditional_nostrict_trace,
    pad_token_groups,
    unpad_token_groups,
)
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.quantization.quantize_.common import KernelPreference
from torchao.utils import is_sm_at_least_100

_ALIGNMENT = 128
_SCALE_RECIPE = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
_SWIZZLE = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]

__all__ = ["four_over_six_grouped_mm"]


@conditional_nostrict_trace
def four_over_six_grouped_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    offs: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    err_mode: str = "mae",
    e4m3_scale_bound: int = 256,
    row_scaled_activation: bool = False,
    weight_block: str = "16x16",
    backward_override: Optional[str] = None,
    pad_token_groups_for_grouped_mm: bool = False,
) -> torch.Tensor:
    """Quantize grouped activations and expert weights with four-over-six.

    ``A`` has shape ``(M, K)``, ``B`` has shape ``(E, N, K)``, and ``offs``
    contains the cumulative row-end offset for each expert. Knobs match
    ``four_over_six_mm``; see the module docstring for the grouped-specific
    backward and alignment semantics.
    """
    output = _FourOverSixGroupedMM.apply(
        A,
        B,
        offs,
        err_mode,
        e4m3_scale_bound,
        row_scaled_activation,
        weight_block,
        backward_override,
        pad_token_groups_for_grouped_mm,
    )
    if bias is not None:
        output = output + bias.to(output.dtype)
    return output


def _expand_group_amax(
    row_amax: torch.Tensor, group_end_offsets: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """Per-row amax vector holding each row's group amax.

    Rows past the final offset (the pad-helper's over-allocated tail) take
    the last group's amax; they are all-zero and never enter the GEMM.
    """
    group_idx = torch.searchsorted(
        group_end_offsets,
        torch.arange(row_amax.shape[0], device=row_amax.device, dtype=torch.int32),
        right=True,
    ).clamp_(max=num_experts - 1)
    group_amax = torch.zeros(
        num_experts, dtype=torch.float32, device=row_amax.device
    ).scatter_reduce_(
        0, group_idx, row_amax.to(torch.float32), reduce="amax", include_self=True
    )
    return group_amax[group_idx], group_amax


def _quantize_expert_weights(
    weight: torch.Tensor,
    weight_amax: torch.Tensor,
    weight_block: str,
    err_mode: str,
    e4m3_scale_bound: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert four-over-six quantization of a stacked (E, N, K) weight."""
    codes = []
    scales = []
    for e in range(weight.shape[0]):
        expert_codes, expert_scales = four_over_six_quantize(
            weight[e],
            weight_amax[e],
            block=weight_block,
            err_mode=err_mode,
            e4m3_scale_bound=e4m3_scale_bound,
        )
        codes.append(expert_codes)
        scales.append(expert_scales)
    return torch.stack(codes), torch.stack(scales)


def _dequantize_expert_weights(
    codes: torch.Tensor,
    scales: torch.Tensor,
    weight_amax: torch.Tensor,
    e4m3_scale_bound: int,
) -> torch.Tensor:
    """Dequantize stacked per-expert codes back to a bf16 (E, N, K) weight.

    Flattening experts along rows and expanding each expert's amax over its
    rows reproduces the per-expert scalar dequantization exactly — the
    decode chain reads one amax entry per row either way.
    """
    num_experts, N = codes.shape[0], codes.shape[1]
    row_amax = weight_amax.to(torch.float32).repeat_interleave(N)
    flat = four_over_six_dequantize(
        codes.reshape(num_experts * N, -1),
        scales.reshape(num_experts * N, -1),
        row_amax,
        e4m3_scale_bound=e4m3_scale_bound,
    )
    return flat.view(num_experts, N, -1)


class _FourOverSixGroupedMM(torch.autograd.Function):
    """NVFP4 four-over-six grouped forward with override-only backward."""

    @staticmethod
    def forward(
        ctx,
        input_act: torch.Tensor,
        weight: torch.Tensor,
        group_end_offsets: torch.Tensor,
        err_mode: str,
        e4m3_scale_bound: int,
        row_scaled_activation: bool,
        weight_block: str,
        backward_override: Optional[str],
        pad_token_groups_for_grouped_mm: bool,
    ) -> torch.Tensor:
        if input_act.ndim != 2:
            raise ValueError(f"input_act must be 2D, got {input_act.ndim}D")
        if weight.ndim != 3:
            raise ValueError(f"weight must be 3D, got {weight.ndim}D")
        if group_end_offsets is None:
            raise ValueError("offs is required for NVFP4 grouped GEMM")
        if group_end_offsets.ndim != 1 or group_end_offsets.dtype != torch.int32:
            raise ValueError("offs must be a 1D int32 tensor")
        if not group_end_offsets.is_contiguous():
            raise ValueError("offs must be contiguous")
        if group_end_offsets.numel() != weight.shape[0]:
            raise ValueError("offs must contain one group-end offset per expert")
        if not (input_act.is_cuda and weight.is_cuda and group_end_offsets.is_cuda):
            raise ValueError("input_act, weight, and offs must be CUDA tensors")
        if not (input_act.device == weight.device == group_end_offsets.device):
            raise ValueError("all tensor arguments must be on the same device")
        if not is_sm_at_least_100():
            raise NotImplementedError(
                "NVFP4 four-over-six grouped GEMM requires SM100+"
            )
        if backward_override is None:
            backward_override = "high_precision"
        if backward_override not in ("high_precision", "dequantized"):
            if backward_override == "quantized":
                raise ValueError(
                    "grouped four-over-six has no quantized backward; use "
                    "'high_precision' or 'dequantized'"
                )
            raise ValueError(
                f"backward_override must be 'high_precision' or 'dequantized', "
                f"got {backward_override!r}"
            )
        if weight_block not in ("1x16", "16x16"):
            raise ValueError(
                f"weight_block must be '1x16' or '16x16', got {weight_block!r}"
            )

        num_tokens, K = input_act.shape
        num_experts, N, weight_K = weight.shape
        if weight_K != K:
            raise ValueError(
                f"input and weight contraction dimensions differ: {K} and {weight_K}"
            )
        if K % _ALIGNMENT != 0 or N % _ALIGNMENT != 0:
            raise ValueError(
                f"K and N must be divisible by {_ALIGNMENT}; got K={K}, N={N}"
            )
        if _DEVICE_ASSERTS:
            group_sizes = torch.diff(
                group_end_offsets, prepend=group_end_offsets.new_zeros(1)
            )
            torch.ops.aten._assert_async.msg(
                torch.all(group_sizes > 0), "offs must describe non-empty groups"
            )
            torch.ops.aten._assert_async.msg(
                group_end_offsets[-1] == num_tokens,
                "the final group-end offset must equal A.shape[0]",
            )
            if not pad_token_groups_for_grouped_mm:
                torch.ops.aten._assert_async.msg(
                    torch.all(group_sizes % _ALIGNMENT == 0),
                    "every token group must be 128-row aligned when padding is disabled",
                )

        input_act = input_act.to(torch.bfloat16).contiguous()
        weight = weight.to(torch.bfloat16).contiguous()
        original_input = input_act

        padded_group_start_offsets = None
        if pad_token_groups_for_grouped_mm:
            # The fused pad/unpad CUDA kernels only accept alignment_size 32
            # and at most 32 groups; this op needs 128-row alignment with any
            # expert count, so it pins the pure-torch path.
            input_act, padded_group_start_offsets, padded_group_end_offsets = (
                pad_token_groups(
                    input_act,
                    group_end_offsets,
                    alignment_size=_ALIGNMENT,
                    kernel_preference=KernelPreference.EMULATED,
                )
            )
        else:
            padded_group_end_offsets = group_end_offsets

        row_amax = input_act.abs().amax(dim=1)
        group_amax = None
        if row_scaled_activation:
            x_amax = row_amax.to(torch.float32)
        else:
            x_amax, group_amax = _expand_group_amax(
                row_amax, padded_group_end_offsets, num_experts
            )
        weight_amax = weight.abs().amax(dim=(1, 2)).to(torch.float32)

        x_codes, x_scales = four_over_six_quantize(
            input_act,
            x_amax,
            block="1x16",
            err_mode=err_mode,
            e4m3_scale_bound=e4m3_scale_bound,
        )
        w_codes, w_scales = _quantize_expert_weights(
            weight, weight_amax, weight_block, err_mode, e4m3_scale_bound
        )
        w_global = _global_decode_scale(weight_amax, e4m3_scale_bound)

        if row_scaled_activation:
            # TransformerEngine has no fused row-scaled NVFP4 grouped GEMM;
            # its general_grouped_gemm loops dense GEMMs per group, and so
            # does this: FP32 output with the constant 1/(6*bound) factor in
            # the per-tensor slot, scaled by the raw per-row amaxes.
            x_global = torch.full(
                (),
                1.0 / (FP4_E2M1_MAX * float(e4m3_scale_bound)),
                dtype=torch.float32,
                device=input_act.device,
            )
            group_bounds = torch.stack(
                (
                    padded_group_end_offsets
                    - torch.diff(
                        padded_group_end_offsets,
                        prepend=padded_group_end_offsets.new_zeros(1),
                    ),
                    padded_group_end_offsets,
                ),
                dim=1,
            ).tolist()
            output = input_act.new_zeros(input_act.shape[0], N)
            for e, (start, end) in enumerate(group_bounds):
                if start == end:
                    continue
                group_out = _scaled_mm_nvfp4(
                    x_codes[start:end],
                    x_scales[start:end],
                    x_global,
                    w_codes[e].t(),
                    w_scales[e],
                    w_global[e],
                    torch.float32,
                )
                output[start:end] = (
                    group_out * x_amax[start:end].view(-1, 1)
                ).to(torch.bfloat16)
        else:
            output = F.scaled_grouped_mm(
                x_codes.view(torch.float4_e2m1fn_x2),
                w_codes.view(torch.float4_e2m1fn_x2).transpose(-2, -1),
                # scaled_grouped_mm consumes swizzled scale bytes viewed at the
                # logical 2D shape (the layout the group quantize kernels
                # return); the view needs the 128-row alignment enforced above.
                scale_a=[
                    to_blocked(x_scales).view(x_scales.shape),
                    _global_decode_scale(group_amax, e4m3_scale_bound),
                ],
                scale_recipe_a=_SCALE_RECIPE,
                scale_b=[
                    torch.stack([to_blocked(s) for s in w_scales]).reshape(
                        num_experts, -1
                    ),
                    w_global,
                ],
                scale_recipe_b=_SCALE_RECIPE,
                swizzle_a=_SWIZZLE,
                swizzle_b=_SWIZZLE,
                offs=padded_group_end_offsets,
                output_dtype=torch.bfloat16,
            )

        if pad_token_groups_for_grouped_mm:
            output = unpad_token_groups(
                output,
                group_end_offsets,
                padded_group_start_offsets,
                num_tokens,
                alignment_size=_ALIGNMENT,
                kernel_preference=KernelPreference.EMULATED,
            )

        if backward_override == "high_precision":
            ctx.save_for_backward(original_input, weight, group_end_offsets)
        else:
            if padded_group_start_offsets is None:
                padded_group_start_offsets = group_end_offsets.new_zeros(0)
            ctx.save_for_backward(
                x_codes,
                x_scales,
                x_amax,
                w_codes,
                w_scales,
                weight_amax,
                group_end_offsets,
                padded_group_start_offsets,
            )
        ctx.backward_override = backward_override
        ctx.e4m3_scale_bound = e4m3_scale_bound
        ctx.pad_token_groups_for_grouped_mm = pad_token_groups_for_grouped_mm
        ctx.num_tokens = num_tokens
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.to(torch.bfloat16).contiguous()

        if ctx.backward_override == "high_precision":
            input_act, weight, group_end_offsets = ctx.saved_tensors
        else:
            (
                x_codes,
                x_scales,
                x_amax,
                w_codes,
                w_scales,
                weight_amax,
                group_end_offsets,
                padded_group_start_offsets,
            ) = ctx.saved_tensors
            input_act = four_over_six_dequantize(
                x_codes, x_scales, x_amax, e4m3_scale_bound=ctx.e4m3_scale_bound
            )
            if ctx.pad_token_groups_for_grouped_mm:
                input_act = unpad_token_groups(
                    input_act,
                    group_end_offsets,
                    padded_group_start_offsets,
                    ctx.num_tokens,
                    alignment_size=_ALIGNMENT,
                    kernel_preference=KernelPreference.EMULATED,
                )
            weight = _dequantize_expert_weights(
                w_codes, w_scales, weight_amax, ctx.e4m3_scale_bound
            )

        grad_input = torch._grouped_mm(
            grad_output,
            weight,
            offs=group_end_offsets,
            out_dtype=torch.bfloat16,
        )
        grad_weight = torch._grouped_mm(
            grad_output.transpose(-2, -1),
            input_act,
            offs=group_end_offsets,
            out_dtype=torch.bfloat16,
        )
        return grad_input, grad_weight, None, None, None, None, None, None, None
