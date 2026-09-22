# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Differentiable NVFP4 grouped GEMMs for the V2 and V1_REQUANT recipes.

The MoE counterparts of ``nvfp4_linear_v2``. Same recipes, same kernels, same
numerators -- the only differences are that ``offsets`` describes more than one group
and that the three GEMMs are ``F.scaled_grouped_mm`` instead of ``F.scaled_mm``:
forward and dgrad are 2d-3d, wgrad is 2d-2d, all with ``offs``.

Design doc §17 routes an MoE FFN's FC1 (``w1``/``w3``) to V1_REQUANT and FC2
(``w2``) to V2. That routing is the caller's to make -- both entrypoints accept any
of the three GEMMs, and the doc's own test plan requires that swapping them still
runs, so the split is a configuration choice rather than a hard-coded assumption.

Recipe V1 is untouched by this module; it stays in ``nvfp4_grouped_mm.py``.
"""

from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F

from torchao.prototype.moe_training.nvfp4_training.group_col_cast_requantize_cutedsl import (
    cutedsl_group_col_cast_requant_amax,
    cutedsl_group_col_cast_requantize,
)
from torchao.prototype.moe_training.nvfp4_training.group_col_cast_requantize_triton import (
    triton_group_col_cast_requant_amax,
    triton_group_col_cast_requantize,
)
from torchao.prototype.moe_training.nvfp4_training.group_col_rht_requantize_cutedsl import (
    cutedsl_group_col_rht_requant_amax,
    cutedsl_group_col_rht_requantize,
)
from torchao.prototype.moe_training.nvfp4_training.group_col_rht_requantize_triton import (
    triton_group_col_rht_requant_amax,
    triton_group_col_rht_requantize,
)
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_cutedsl import (
    cutedsl_group_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_triton import (
    triton_group_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
    _DEVICE_ASSERTS,
    VARYING_FIRST_DIM,
)
from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_cutedsl import (
    cutedsl_group_rht_quantize_row_col,
)
from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_triton import (
    triton_group_rht_quantize_row_col,
)
from torchao.prototype.moe_training.nvfp4_training.group_row_cast_quantize_cutedsl import (
    cutedsl_group_row_cast_quantize,
)
from torchao.prototype.moe_training.nvfp4_training.group_row_cast_quantize_triton import (
    triton_group_row_cast_quantize,
)
from torchao.prototype.moe_training.nvfp4_training.group_row_rht_col_rht_amax_cutedsl import (
    cutedsl_group_row_rht_col_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.group_row_rht_col_rht_amax_triton import (
    triton_group_row_rht_col_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.group_row_rht_col_rht_quantize_ms_eden_cutedsl import (
    cutedsl_group_row_rht_col_rht_quantize_ms_eden,
)
from torchao.prototype.moe_training.nvfp4_training.group_row_rht_col_rht_quantize_ms_eden_triton import (
    triton_group_row_rht_col_rht_quantize_ms_eden,
)
from torchao.prototype.moe_training.nvfp4_training.group_weight_amax_triton import (
    triton_group_weight_amax,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm import (
    _resolve_backends,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear_v2 import (
    _backward_rng_state,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_recipe import (
    EDEN_NUMERATOR,
    NVFP4_CAST_NUMERATOR,
    _amax_to_scale,
)
from torchao.prototype.moe_training.utils import (
    conditional_nostrict_trace,
    pad_token_groups,
    unpad_token_groups,
)
from torchao.quantization.quantize_.common import KernelPreference
from torchao.utils import is_sm_at_least_100

_ALIGNMENT = 128
_SCALE_RECIPE = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
_SWIZZLE = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]


def _validate_grouped_inputs(
    input_act: torch.Tensor,
    weight: torch.Tensor,
    group_end_offsets: Optional[torch.Tensor],
    sr_seed: torch.Tensor,
    pad_token_groups_for_grouped_mm: bool,
) -> tuple[int, int, int, int]:
    """Shared host validation. Returns ``(num_tokens, K, num_experts, N)``."""
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
    if sr_seed.ndim != 1 or sr_seed.numel() != 1:
        raise ValueError("sr_seed must be a one-element tensor")
    if sr_seed.dtype != torch.int64 or not sr_seed.is_cuda:
        raise ValueError("sr_seed must be a CUDA int64 tensor")
    if not (input_act.is_cuda and weight.is_cuda and group_end_offsets.is_cuda):
        raise ValueError("input_act, weight, and offs must be CUDA tensors")
    if not (
        input_act.device == weight.device == group_end_offsets.device == sr_seed.device
    ):
        raise ValueError("all tensor arguments must be on the same device")
    if not is_sm_at_least_100():
        raise NotImplementedError("NVFP4 grouped training GEMM requires SM100+")

    num_tokens, K = input_act.shape
    num_experts, N, weight_K = weight.shape
    if weight_K != K:
        raise ValueError(
            f"input and weight contraction dimensions differ: {K} and {weight_K}"
        )
    if K % _ALIGNMENT != 0 or N % _ALIGNMENT != 0:
        raise ValueError(f"K and N must be divisible by {_ALIGNMENT}; got K={K}, N={N}")

    if _DEVICE_ASSERTS:
        group_sizes = torch.diff(
            group_end_offsets, prepend=group_end_offsets.new_zeros(1)
        )
        torch.ops.aten._assert_async.msg(
            torch.all(group_sizes > 0), "offs must describe non-empty groups"
        )
        torch.ops.aten._assert_async.msg(
            group_end_offsets[-1] <= num_tokens,
            "the final group-end offset must not exceed A.shape[0]",
        )
        if pad_token_groups_for_grouped_mm:
            torch.ops.aten._assert_async.msg(
                group_end_offsets[-1] == num_tokens,
                "internally padded input offsets must cover A.shape[0]",
            )
        else:
            torch.ops.aten._assert_async.msg(
                torch.all(group_sizes % _ALIGNMENT == 0),
                "every token group must be 128-row aligned when padding is disabled",
            )
    return num_tokens, K, num_experts, N


def _validate_sign_tensor(sign: torch.Tensor, name: str, device) -> None:
    if sign.ndim != 1 or sign.numel() != 128:
        raise ValueError(f"{name} must be a (128,) tensor, got {tuple(sign.shape)}")
    if not sign.is_cuda or sign.device != device:
        raise ValueError(f"{name} must be on the same device as the activation")


def _quantize_weight_rowwise(
    weight: torch.Tensor, num_experts: int, use_cutedsl_weight: bool
):
    """§11.1 over the expert stack. Returns ``(row_fp4_w, row_sf_w, weight_amax)``."""
    weight_amax = triton_group_weight_amax(weight, num_experts)
    group_row_cast_quantize = (
        cutedsl_group_row_cast_quantize
        if use_cutedsl_weight
        else triton_group_row_cast_quantize
    )
    row_fp4_w, row_sf_w = group_row_cast_quantize(weight, weight_amax, num_experts)
    return row_fp4_w, row_sf_w, weight_amax


def _forward_gemm(row_fp4_x, row_sf_x, amax_x, row_fp4_w, row_sf_w, amax_w, offs):
    """§12. Both operands are casts, so both carry numerator 2688."""
    return F.scaled_grouped_mm(
        row_fp4_x.view(torch.float4_e2m1fn_x2),
        # Rowwise W codes are (E, N, K//2); the grouped RHS wants (E, K, N).
        row_fp4_w.view(torch.float4_e2m1fn_x2).transpose(-2, -1),
        scale_a=[row_sf_x, _amax_to_scale(amax_x, NVFP4_CAST_NUMERATOR)],
        scale_recipe_a=_SCALE_RECIPE,
        scale_b=[row_sf_w.flatten(1), _amax_to_scale(amax_w, NVFP4_CAST_NUMERATOR)],
        scale_recipe_b=_SCALE_RECIPE,
        swizzle_a=_SWIZZLE,
        swizzle_b=_SWIZZLE,
        offs=offs,
        output_dtype=torch.bfloat16,
    )


def _dgrad_gemm(
    row_fp4_dy, row_sf_dy, dy_scale, col_fp4_w_t, col_sf_w_t, w_t_scale, offs
):
    """§13. ``dy_scale`` is 1536 for V2 (MS-EDEN) and 2688 for V1_REQUANT (SR cast);
    ``w_t_scale`` is 2688 in both, because the weight is a cast operand either way."""
    return F.scaled_grouped_mm(
        row_fp4_dy.view(torch.float4_e2m1fn_x2),
        col_fp4_w_t.view(torch.float4_e2m1fn_x2).transpose(-2, -1),
        scale_a=[row_sf_dy, dy_scale],
        scale_recipe_a=_SCALE_RECIPE,
        scale_b=[col_sf_w_t.flatten(1), w_t_scale],
        scale_recipe_b=_SCALE_RECIPE,
        swizzle_a=_SWIZZLE,
        swizzle_b=_SWIZZLE,
        offs=offs,
        output_dtype=torch.bfloat16,
    )


def _wgrad_gemm(
    col_fp4_dy_t, col_sf_dy_t, dy_t_scale, col_fp4_x_t, col_sf_x_t, x_t_scale, offs
):
    """§14. 2d-2d with ``offs``. Both operands carry R_m, so the transform cancels."""
    return F.scaled_grouped_mm(
        col_fp4_dy_t.view(torch.float4_e2m1fn_x2),
        col_fp4_x_t.view(torch.float4_e2m1fn_x2).transpose(-2, -1),
        scale_a=[col_sf_dy_t, dy_t_scale],
        scale_recipe_a=_SCALE_RECIPE,
        scale_b=[col_sf_x_t, x_t_scale],
        scale_recipe_b=_SCALE_RECIPE,
        swizzle_a=_SWIZZLE,
        swizzle_b=_SWIZZLE,
        offs=offs,
        output_dtype=torch.bfloat16,
    )


class _NVFP4GroupedMMV2(torch.autograd.Function):
    """Grouped V2: RHT-128 forward, MS-EDEN backward, rotated weight requantization."""

    @staticmethod
    def forward(
        ctx,
        input_act: torch.Tensor,
        weight: torch.Tensor,
        wgrad_rht: torch.Tensor,
        dgrad_rht: torch.Tensor,
        sr_seed: torch.Tensor,
        group_end_offsets: Optional[torch.Tensor],
        pad_token_groups_for_grouped_mm: bool,
        kernel_preference: KernelPreference,
        use_fast_math: bool,
        ms_eden_fast_path: bool,
    ) -> torch.Tensor:
        num_tokens, K, num_experts, N = _validate_grouped_inputs(
            input_act,
            weight,
            group_end_offsets,
            sr_seed,
            pad_token_groups_for_grouped_mm,
        )
        _validate_sign_tensor(wgrad_rht, "wgrad_rht", input_act.device)
        _validate_sign_tensor(dgrad_rht, "dgrad_rht", input_act.device)
        use_cutedsl_rht, use_cutedsl_weight = _resolve_backends(
            kernel_preference, num_experts
        )
        if ms_eden_fast_path and not use_cutedsl_rht:
            raise ValueError(
                "ms_eden_fast_path=True requires the MS-EDEN op on CuteDSL; this call "
                "resolves it to Triton"
            )

        input_act = input_act.to(torch.bfloat16).contiguous()
        weight = weight.to(torch.bfloat16).contiguous()

        padded_group_start_offsets = None
        if pad_token_groups_for_grouped_mm:
            input_act, padded_group_start_offsets, padded_group_end_offsets = (
                pad_token_groups(
                    input_act, group_end_offsets, alignment_size=_ALIGNMENT
                )
            )
        else:
            padded_group_end_offsets = group_end_offsets

        packed_sequence_length = input_act.shape[0]
        logical_packed_length = padded_group_end_offsets[-1:]
        group_rht_amax = (
            cutedsl_group_rht_amax if use_cutedsl_rht else triton_group_rht_amax
        )
        group_rht_quantize_row_col = (
            cutedsl_group_rht_quantize_row_col
            if use_cutedsl_rht
            else triton_group_rht_quantize_row_col
        )

        # §11.8 then §11.9, on the same ops V1_REQUANT uses: dynamic_rht switches
        # them from the memoized RHT-16 tuple to the resampled RHT-128 buffer.
        # rng_state/enable_stochastic_rounding are (None, False) -- V2's forward is
        # RTNE and its stochastic rounding lives in MS-EDEN.
        amax_rht_x_t, amax_x = group_rht_amax(
            input_act,
            [],
            padded_group_end_offsets,
            num_experts,
            packed_sequence_length,
            K,
            VARYING_FIRST_DIM,
            logical_packed_length=logical_packed_length,
            sign_tensor=wgrad_rht,
            dynamic_rht=True,
        )
        (
            row_fp4_x,
            row_sf_x,
            col_fp4_rht_x_t,
            col_sf_rht_x_t,
        ) = group_rht_quantize_row_col(
            input_act,
            [],
            padded_group_end_offsets,
            num_experts,
            packed_sequence_length,
            K,
            VARYING_FIRST_DIM,
            amax_x,
            amax_rht_x_t,
            None,
            False,
            logical_packed_length,
            use_fast_math,
            sign_tensor=wgrad_rht,
            dynamic_rht=True,
        )

        row_fp4_w, row_sf_w, weight_amax = _quantize_weight_rowwise(
            weight, num_experts, use_cutedsl_weight
        )
        output = _forward_gemm(
            row_fp4_x,
            row_sf_x,
            amax_x,
            row_fp4_w,
            row_sf_w,
            weight_amax,
            padded_group_end_offsets,
        )
        if pad_token_groups_for_grouped_mm:
            output = unpad_token_groups(
                output,
                group_end_offsets,
                padded_group_start_offsets,
                num_tokens,
                alignment_size=_ALIGNMENT,
            )

        ctx.save_for_backward(
            col_fp4_rht_x_t,
            col_sf_rht_x_t,
            amax_rht_x_t,
            row_fp4_w,
            row_sf_w,
            weight_amax,
            group_end_offsets,
            padded_group_start_offsets,
            padded_group_end_offsets,
            wgrad_rht,
            dgrad_rht,
            sr_seed,
        )
        ctx.pad_token_groups_for_grouped_mm = pad_token_groups_for_grouped_mm
        ctx.num_tokens = num_tokens
        ctx.num_experts = num_experts
        ctx.use_cutedsl_rht = use_cutedsl_rht
        ctx.use_cutedsl_weight = use_cutedsl_weight
        ctx.ms_eden_fast_path = ms_eden_fast_path
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            col_fp4_rht_x_t,
            col_sf_rht_x_t,
            amax_rht_x_t,
            row_fp4_w,
            row_sf_w,
            weight_amax,
            original_group_end_offsets,
            padded_group_start_offsets,
            padded_group_end_offsets,
            wgrad_rht,
            dgrad_rht,
            sr_seed,
        ) = ctx.saved_tensors

        grad_output = grad_output.to(torch.bfloat16).contiguous()
        if ctx.pad_token_groups_for_grouped_mm:
            grad_output, _, _ = pad_token_groups(
                grad_output, original_group_end_offsets, alignment_size=_ALIGNMENT
            )
        num_experts = ctx.num_experts
        packed_sequence_length, N = grad_output.shape
        logical_packed_length = padded_group_end_offsets[-1:]
        group_row_rht_col_rht_amax = (
            cutedsl_group_row_rht_col_rht_amax
            if ctx.use_cutedsl_rht
            else triton_group_row_rht_col_rht_amax
        )
        group_row_rht_col_rht_quantize_ms_eden = (
            partial(
                cutedsl_group_row_rht_col_rht_quantize_ms_eden,
                fast_path=ctx.ms_eden_fast_path,
            )
            if ctx.use_cutedsl_rht
            else triton_group_row_rht_col_rht_quantize_ms_eden
        )
        group_col_rht_requant_amax = (
            cutedsl_group_col_rht_requant_amax
            if ctx.use_cutedsl_weight
            else triton_group_col_rht_requant_amax
        )
        group_col_rht_requantize = (
            cutedsl_group_col_rht_requantize
            if ctx.use_cutedsl_weight
            else triton_group_col_rht_requantize
        )

        amax_rht_dy, amax_rht_dy_t = group_row_rht_col_rht_amax(
            grad_output,
            dgrad_rht,
            wgrad_rht,
            padded_group_end_offsets,
            num_experts,
            packed_sequence_length,
            N,
            VARYING_FIRST_DIM,
            logical_packed_length,
        )
        (
            row_fp4_rht_dy,
            row_sf_rht_dy,
            col_fp4_rht_dy_t,
            col_sf_rht_dy_t,
        ) = group_row_rht_col_rht_quantize_ms_eden(
            grad_output,
            amax_rht_dy,
            amax_rht_dy_t,
            dgrad_rht,
            wgrad_rht,
            padded_group_end_offsets,
            num_experts,
            packed_sequence_length,
            N,
            VARYING_FIRST_DIM,
            _backward_rng_state(sr_seed),
            logical_packed_length,
        )

        amax_rht_w_qdq_t = group_col_rht_requant_amax(
            row_fp4_w, row_sf_w, weight_amax, dgrad_rht, num_experts
        )
        col_fp4_rht_w_t, col_sf_rht_w_t = group_col_rht_requantize(
            row_fp4_w, row_sf_w, weight_amax, amax_rht_w_qdq_t, dgrad_rht, num_experts
        )

        grad_input = _dgrad_gemm(
            row_fp4_rht_dy,
            row_sf_rht_dy,
            _amax_to_scale(amax_rht_dy, EDEN_NUMERATOR),
            col_fp4_rht_w_t,
            col_sf_rht_w_t,
            _amax_to_scale(amax_rht_w_qdq_t, NVFP4_CAST_NUMERATOR),
            padded_group_end_offsets,
        )
        grad_weight = _wgrad_gemm(
            col_fp4_rht_dy_t,
            col_sf_rht_dy_t,
            _amax_to_scale(amax_rht_dy_t, EDEN_NUMERATOR),
            col_fp4_rht_x_t,
            col_sf_rht_x_t,
            _amax_to_scale(amax_rht_x_t, NVFP4_CAST_NUMERATOR),
            padded_group_end_offsets,
        )
        if ctx.pad_token_groups_for_grouped_mm:
            grad_input = unpad_token_groups(
                grad_input,
                original_group_end_offsets,
                padded_group_start_offsets,
                ctx.num_tokens,
                alignment_size=_ALIGNMENT,
            )
        # input_act, weight, wgrad_rht, dgrad_rht, sr_seed, offs, pad,
        # kernel_preference, use_fast_math, ms_eden_fast_path
        return grad_input, grad_weight, None, None, None, None, None, None, None, None


class _NVFP4GroupedMMV1Requant(torch.autograd.Function):
    """Grouped V1_REQUANT: RHT-16 forward, SR backward, unrotated requantization."""

    @staticmethod
    def forward(
        ctx,
        input_act: torch.Tensor,
        weight: torch.Tensor,
        sign_vector: tuple,
        sr_seed: torch.Tensor,
        group_end_offsets: Optional[torch.Tensor],
        pad_token_groups_for_grouped_mm: bool,
        kernel_preference: KernelPreference,
        use_fast_math: bool,
    ) -> torch.Tensor:
        if not isinstance(sign_vector, (tuple, list)) or len(sign_vector) != 16:
            raise ValueError("sign_vector must be a tuple or list with 16 elements")
        if any(sign not in (-1, 1) for sign in sign_vector):
            raise ValueError("sign_vector elements must be -1 or 1")
        num_tokens, K, num_experts, N = _validate_grouped_inputs(
            input_act,
            weight,
            group_end_offsets,
            sr_seed,
            pad_token_groups_for_grouped_mm,
        )
        use_cutedsl_rht, use_cutedsl_weight = _resolve_backends(
            kernel_preference, num_experts
        )
        sign_vector = tuple(sign_vector)
        sv = list(sign_vector)

        input_act = input_act.to(torch.bfloat16).contiguous()
        weight = weight.to(torch.bfloat16).contiguous()

        padded_group_start_offsets = None
        if pad_token_groups_for_grouped_mm:
            input_act, padded_group_start_offsets, padded_group_end_offsets = (
                pad_token_groups(
                    input_act, group_end_offsets, alignment_size=_ALIGNMENT
                )
            )
        else:
            padded_group_end_offsets = group_end_offsets

        packed_sequence_length = input_act.shape[0]
        logical_packed_length = padded_group_end_offsets[-1:]
        group_rht_amax = (
            cutedsl_group_rht_amax if use_cutedsl_rht else triton_group_rht_amax
        )
        group_rht_quantize_row_col = (
            cutedsl_group_rht_quantize_row_col
            if use_cutedsl_rht
            else triton_group_rht_quantize_row_col
        )

        amax_rht_x_t, amax_x = group_rht_amax(
            input_act,
            sv,
            padded_group_end_offsets,
            num_experts,
            packed_sequence_length,
            K,
            VARYING_FIRST_DIM,
            logical_packed_length=logical_packed_length,
        )
        (
            row_fp4_x,
            row_sf_x,
            col_fp4_rht_x_t,
            col_sf_rht_x_t,
        ) = group_rht_quantize_row_col(
            input_act,
            sv,
            padded_group_end_offsets,
            num_experts,
            packed_sequence_length,
            K,
            VARYING_FIRST_DIM,
            amax_x,
            amax_rht_x_t,
            None,
            False,
            logical_packed_length,
            use_fast_math,
        )

        row_fp4_w, row_sf_w, weight_amax = _quantize_weight_rowwise(
            weight, num_experts, use_cutedsl_weight
        )
        output = _forward_gemm(
            row_fp4_x,
            row_sf_x,
            amax_x,
            row_fp4_w,
            row_sf_w,
            weight_amax,
            padded_group_end_offsets,
        )
        if pad_token_groups_for_grouped_mm:
            output = unpad_token_groups(
                output,
                group_end_offsets,
                padded_group_start_offsets,
                num_tokens,
                alignment_size=_ALIGNMENT,
            )

        ctx.save_for_backward(
            col_fp4_rht_x_t,
            col_sf_rht_x_t,
            amax_rht_x_t,
            row_fp4_w,
            row_sf_w,
            weight_amax,
            group_end_offsets,
            padded_group_start_offsets,
            padded_group_end_offsets,
            sr_seed,
        )
        ctx.pad_token_groups_for_grouped_mm = pad_token_groups_for_grouped_mm
        ctx.num_tokens = num_tokens
        ctx.num_experts = num_experts
        ctx.sign_vector = sign_vector
        ctx.use_cutedsl_rht = use_cutedsl_rht
        ctx.use_cutedsl_weight = use_cutedsl_weight
        ctx.use_fast_math = use_fast_math
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            col_fp4_rht_x_t,
            col_sf_rht_x_t,
            amax_rht_x_t,
            row_fp4_w,
            row_sf_w,
            weight_amax,
            original_group_end_offsets,
            padded_group_start_offsets,
            padded_group_end_offsets,
            sr_seed,
        ) = ctx.saved_tensors

        grad_output = grad_output.to(torch.bfloat16).contiguous()
        if ctx.pad_token_groups_for_grouped_mm:
            grad_output, _, _ = pad_token_groups(
                grad_output, original_group_end_offsets, alignment_size=_ALIGNMENT
            )
        num_experts = ctx.num_experts
        packed_sequence_length, N = grad_output.shape
        logical_packed_length = padded_group_end_offsets[-1:]
        sv = list(ctx.sign_vector)
        group_rht_amax = (
            cutedsl_group_rht_amax if ctx.use_cutedsl_rht else triton_group_rht_amax
        )
        group_rht_quantize_row_col = (
            cutedsl_group_rht_quantize_row_col
            if ctx.use_cutedsl_rht
            else triton_group_rht_quantize_row_col
        )
        group_col_cast_requant_amax = (
            cutedsl_group_col_cast_requant_amax
            if ctx.use_cutedsl_weight
            else triton_group_col_cast_requant_amax
        )
        group_col_cast_requantize = (
            cutedsl_group_col_cast_requantize
            if ctx.use_cutedsl_weight
            else triton_group_col_cast_requantize
        )

        amax_rht_dy_t, amax_dy = group_rht_amax(
            grad_output,
            sv,
            padded_group_end_offsets,
            num_experts,
            packed_sequence_length,
            N,
            VARYING_FIRST_DIM,
            logical_packed_length=logical_packed_length,
        )
        (
            row_fp4_dy,
            row_sf_dy,
            col_fp4_rht_dy_t,
            col_sf_rht_dy_t,
        ) = group_rht_quantize_row_col(
            grad_output,
            sv,
            padded_group_end_offsets,
            num_experts,
            packed_sequence_length,
            N,
            VARYING_FIRST_DIM,
            amax_dy,
            amax_rht_dy_t,
            _backward_rng_state(sr_seed),
            True,
            logical_packed_length,
            ctx.use_fast_math,
        )

        amax_w_qdq_t = group_col_cast_requant_amax(
            row_fp4_w, row_sf_w, weight_amax, num_experts
        )
        col_fp4_w_t, col_sf_w_t = group_col_cast_requantize(
            row_fp4_w, row_sf_w, weight_amax, amax_w_qdq_t, num_experts
        )

        grad_input = _dgrad_gemm(
            row_fp4_dy,
            row_sf_dy,
            _amax_to_scale(amax_dy, NVFP4_CAST_NUMERATOR),
            col_fp4_w_t,
            col_sf_w_t,
            _amax_to_scale(amax_w_qdq_t, NVFP4_CAST_NUMERATOR),
            padded_group_end_offsets,
        )
        grad_weight = _wgrad_gemm(
            col_fp4_rht_dy_t,
            col_sf_rht_dy_t,
            _amax_to_scale(amax_rht_dy_t, NVFP4_CAST_NUMERATOR),
            col_fp4_rht_x_t,
            col_sf_rht_x_t,
            _amax_to_scale(amax_rht_x_t, NVFP4_CAST_NUMERATOR),
            padded_group_end_offsets,
        )
        if ctx.pad_token_groups_for_grouped_mm:
            grad_input = unpad_token_groups(
                grad_input,
                original_group_end_offsets,
                padded_group_start_offsets,
                ctx.num_tokens,
                alignment_size=_ALIGNMENT,
            )
        # input_act, weight, sign_vector, sr_seed, offs, pad, kernel_preference,
        # use_fast_math
        return grad_input, grad_weight, None, None, None, None, None, None


@conditional_nostrict_trace
def nvfp4_v2_grouped_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    wgrad_rht: torch.Tensor,
    dgrad_rht: torch.Tensor,
    sr_seed: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    pad_token_groups_for_grouped_mm: bool = False,
    kernel_preference: KernelPreference = KernelPreference.AUTO,
    use_fast_math: bool = True,
    ms_eden_fast_path: bool = False,
) -> torch.Tensor:
    """Grouped ``A @ B[g].t()`` under the V2 recipe.

    Args:
        A: ``(M, K)`` activations, rows contiguous by expert.
        B: ``(E, N, K)`` expert weights, **un-transposed**. Note this is torchao's
            grouped-mm convention and the opposite of torchtitan's ``B_t`` seam.
        wgrad_rht: ``(128,)`` int8 sign buffer, resampled per microbatch.
        dgrad_rht: ``(128,)`` int8 sign buffer, resampled per optimizer step.
        sr_seed: one-element int64 CUDA tensor, the Philox key.
        offs: ``(E,)`` int32 cumulative row-end offsets, no leading zero -- exactly
            what ``torch.cumsum(num_tokens_per_expert, 0, dtype=torch.int32)`` gives.
            Required.
        bias: optional ``(N,)``, added after the GEMM.
        pad_token_groups_for_grouped_mm: pad each group up to 128 rows internally and
            strip the padding before returning. With padding off, every group size
            must already be 128-aligned.
        kernel_preference: selects the quantization backend. AUTO (default) runs each
            op on CuteDSL where it can and falls back to Triton per op; TRITON forces
            Triton; CUTEDSL demands CuteDSL and raises if it cannot run. The
            per-expert weight amax is Triton on every path -- it has no CuteDSL twin.
        use_fast_math: match TransformerEngine under ``NVTE_USE_FAST_MATH=1``.
        ms_eden_fast_path: round the MS-EDEN scales of the backward in hardware
            (``cvt.rs``) rather than in software. The FP4 codes are bitwise with the
            default path and with Triton; each scale byte lands on one of the two E4M3
            neighbours of the same corrected scale, so the step is not bitwise with
            Triton. CuteDSL only: raises if the MS-EDEN op resolves to Triton.

    When padding is off, ``offs[-1]`` may be less than ``M``: rows from ``offs[-1]``
    on are the dispatcher's spare capacity, are never read, and carry no contract in
    the output or in ``grad_input``. Slice to ``offs[-1]`` before consuming either.
    """
    output = _NVFP4GroupedMMV2.apply(
        A,
        B,
        wgrad_rht,
        dgrad_rht,
        sr_seed,
        offs,
        pad_token_groups_for_grouped_mm,
        kernel_preference,
        use_fast_math,
        ms_eden_fast_path,
    )
    if bias is not None:
        output = output + bias.to(output.dtype)
    return output


@conditional_nostrict_trace
def nvfp4_v1_requant_grouped_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    sign_vector,
    sr_seed: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    pad_token_groups_for_grouped_mm: bool = False,
    kernel_preference: KernelPreference = KernelPreference.AUTO,
    use_fast_math: bool = True,
) -> torch.Tensor:
    """Grouped ``A @ B[g].t()`` under the V1_REQUANT recipe.

    Same contract as ``nvfp4_v2_grouped_mm`` except for the sign vector: this recipe
    takes one static 16-element ``{-1, +1}`` tuple, fixed for the whole run, and has
    no ``dgrad_rht`` because it applies no transform on the dgrad path.
    """
    output = _NVFP4GroupedMMV1Requant.apply(
        A,
        B,
        tuple(sign_vector),
        sr_seed,
        offs,
        pad_token_groups_for_grouped_mm,
        kernel_preference,
        use_fast_math,
    )
    if bias is not None:
        output = output + bias.to(output.dtype)
    return output
