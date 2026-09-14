# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Dense NVFP4 linear layers for the V2 and V1_REQUANT recipes.

    y  = x @ w.t()
    dx = dy @ w
    dw = dy.t() @ x

**Every quantize and amax call here is a grouped op driven at ``num_tensors = 1``.**
There is no linear kernel set for these recipes: a dense linear is the degenerate
one-group case, and maintaining a second code path for it would buy a few percent
at the cost of a second set of numerics to keep in agreement. The only linear-specific
pieces are ``_degenerate_group_args`` and the choice of ``F.scaled_mm`` over
``F.scaled_grouped_mm`` for the three GEMMs -- a one-group grouped GEMM is strictly
slower and buys nothing.

Recipe V1 is untouched by this module; it stays in ``nvfp4_linear.py``.

What the two recipes share and where they diverge:

* Forward is structurally identical -- quantize ``x`` and ``x.t()``, quantize ``w``
  rowwise only, and GEMM. They differ in Hadamard size (16 vs 128) and in whether
  the sign vector is a static tuple or a live device buffer.
* Backward diverges completely. V1_REQUANT rounds ``dy`` stochastically and rebuilds
  ``w.t()`` with no rotation; V2 applies MS-EDEN to ``dy`` on both axes and rebuilds
  ``w.t()`` rotated by ``R_n``.
"""

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
from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import (
    _resolve_use_cutedsl,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_recipe import (
    EDEN_NUMERATOR,
    NVFP4_CAST_NUMERATOR,
    _amax_to_scale,
)
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

_ALIGNMENT = 128
_SCALE_RECIPE = [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise]
_SWIZZLE = [F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE]


def _degenerate_group_args(num_rows: int, device_key: str) -> torch.Tensor:
    """The one-element ``offsets`` tensor describing a dense tensor as a single group.

    ``logical_packed_length`` is the same tensor -- for one group, ``offsets[-1:]``
    *is* the logical length -- so one tensor serves both arguments.

    This was a per-device buffer, overwritten in place so its address stayed stable
    across CUDA-graph replays. That could not work: Dynamo traces under a
    FakeTensorMode in every compile mode, so filling a real buffer during tracing
    raises, and the buffer poisoned ``torch.compile`` for any process that had run a
    single eager forward or called ``prepare_for_cuda_graph`` -- including one set up
    for V1, which prewarmed this without using it. Allocating per call restores
    ``torch.compile`` unconditionally.

    The cost is that ``mode="reduce-overhead"`` allocates this into the graph pool
    and fails capture. It already did: the buffer was bypassed while tracing when
    cold, which is the same fresh allocation, so that path fails identically with or
    without the buffer. Making capture work needs the write hoisted out of the traced
    region, not a stable address.
    """
    return torch.tensor([num_rows], dtype=torch.int32, device=device_key)


def _nvfp4_fp4_matmul(
    a_fp4: torch.Tensor,
    a_sf: torch.Tensor,
    b_fp4: torch.Tensor,
    b_sf: torch.Tensor,
    global_scale_a: torch.Tensor,
    global_scale_b: torch.Tensor,
) -> torch.Tensor:
    """NVFP4 ``a @ b.t()`` from packed FP4 operands and swizzled block scales.

    ``global_scale_a`` / ``global_scale_b`` must each come from ``_amax_to_scale``
    called with the numerator of the *quantizer that produced that operand*. V2's
    dgrad and wgrad both pair a 1536 operand with a 2688 one in a single call.
    """
    return F.scaled_mm(
        a_fp4.view(torch.float4_e2m1fn_x2),
        b_fp4.view(torch.float4_e2m1fn_x2).t(),
        scale_a=[a_sf.flatten(), global_scale_a],
        scale_recipe_a=_SCALE_RECIPE,
        scale_b=[b_sf.flatten(), global_scale_b],
        scale_recipe_b=_SCALE_RECIPE,
        swizzle_a=_SWIZZLE,
        swizzle_b=_SWIZZLE,
        output_dtype=torch.bfloat16,
    )


def _check_shapes(M: int, K: int, N: int) -> None:
    if M % _ALIGNMENT or K % _ALIGNMENT or N % _ALIGNMENT:
        raise ValueError(
            f"NVFP4 linear requires M, K, N all divisible by {_ALIGNMENT}; "
            f"got M={M}, K={K}, N={N}"
        )


def _quantize_weight_rowwise(weight: torch.Tensor, use_cutedsl: bool):
    """§11.1 at ``num_experts = 1``. Returns ``(row_fp4_w, row_sf_w, amax_w)``.

    All three are kept in their grouped ``(1, ...)`` form: the requantization ops in
    backward are grouped too, so squeezing here only to unsqueeze there would be
    churn. Only the GEMM needs a 2D view, and it takes one at the call site.
    """
    weight_3d = weight.unsqueeze(0)
    amax_w = triton_group_weight_amax(weight_3d, 1)
    group_row_cast_quantize = (
        cutedsl_group_row_cast_quantize
        if use_cutedsl
        else triton_group_row_cast_quantize
    )
    row_fp4_w, row_sf_w = group_row_cast_quantize(weight_3d, amax_w, 1)
    return row_fp4_w, row_sf_w, amax_w


def _backward_rng_state(sr_seed: torch.Tensor) -> torch.Tensor:
    """Philox state ``[col_seed, col_offset, row_seed, row_offset]`` for one backward.

    Offsets are drawn fresh from the default CUDA generator, with no ``generator=``
    argument on purpose: under ``torch.compile(mode="reduce-overhead")`` the default
    CUDA generator is a first-class CUDA-graph side input that the framework advances
    between replays, so each replay gets new noise without any counter plumbing of
    our own. Row and column streams are decorrelated by ``sr_seed ^ 1`` on the key as
    well as by independent offsets.
    """
    col_offset = torch.randint(0, 2**32, (1,), dtype=torch.int64, device=sr_seed.device)
    row_offset = torch.randint(0, 2**32, (1,), dtype=torch.int64, device=sr_seed.device)
    return torch.cat((sr_seed, col_offset, sr_seed ^ 1, row_offset))


@torch._dynamo.allow_in_graph
class _NVFP4LinearV2(torch.autograd.Function):
    """RHT-128 forward, MS-EDEN backward, lazy rotated weight requantization.

    Design doc §15. Kernel sequence::

        fwd:  §11.8 -> §11.9 -> amax(w) -> §11.1 -> §12
        bwd:  §11.2 -> §11.3 -> §11.4 -> §11.5 -> §13 -> §14

    Saved for backward: the columnwise activation operand, the packed rowwise weight,
    and three amaxes. Neither the bf16 activation nor a bf16 ``w.t()`` is saved --
    roughly 0.5625 bytes per weight element plus the activation transpose.
    """

    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp: torch.Tensor,
        bias: Optional[torch.Tensor],
        wgrad_rht: torch.Tensor,
        dgrad_rht: torch.Tensor,
        sr_seed: torch.Tensor,
        use_cutedsl: bool = False,
        use_fast_math: bool = True,
    ):
        M = input_hp.shape[-2]
        K = input_hp.shape[-1]
        N = weight_hp.shape[0]
        _check_shapes(M, K, N)
        input_hp = input_hp.to(torch.bfloat16)
        weight_hp = weight_hp.to(torch.bfloat16)
        input_2d = input_hp.reshape(-1, K).contiguous()
        M = input_2d.shape[0]
        offsets = _degenerate_group_args(M, str(input_2d.device))
        group_rht_amax = (
            cutedsl_group_rht_amax if use_cutedsl else triton_group_rht_amax
        )
        group_rht_quantize_row_col = (
            cutedsl_group_rht_quantize_row_col
            if use_cutedsl
            else triton_group_rht_quantize_row_col
        )

        # §11.8 then §11.9: amax first so the columnwise bound is taken post-RHT.
        # Same ops as V1_REQUANT below, driven at RHT-128 with a resampled sign
        # buffer instead of RHT-16 with a static tuple -- hence dynamic_rht, which
        # keeps the per-launch product out of get_rht_matrix's by-value cache.
        # Both operands are RTNE here: V2's stochastic rounding lives in MS-EDEN.
        amax_rht_x_t, amax_x = group_rht_amax(
            input_2d,
            [],
            offsets,
            1,
            M,
            K,
            VARYING_FIRST_DIM,
            logical_packed_length=offsets,
            sign_tensor=wgrad_rht,
            dynamic_rht=True,
        )
        (
            row_fp4_x,
            row_sf_x,
            col_fp4_rht_x_t,
            col_sf_rht_x_t,
        ) = group_rht_quantize_row_col(
            input_2d,
            [],
            offsets,
            1,
            M,
            K,
            VARYING_FIRST_DIM,
            amax_x,
            amax_rht_x_t,
            None,
            False,
            offsets,
            use_fast_math,
            sign_tensor=wgrad_rht,
            dynamic_rht=True,
        )

        row_fp4_w, row_sf_w, amax_w = _quantize_weight_rowwise(weight_hp, use_cutedsl)

        # §12. Both operands are plain casts, so both carry numerator 2688. This is
        # the one GEMM where the numerators match, which is why a numerator bug shows
        # up as "backward wrong, forward fine".
        output = _nvfp4_fp4_matmul(
            row_fp4_x,
            row_sf_x,
            row_fp4_w[0],
            row_sf_w,
            _amax_to_scale(amax_x, NVFP4_CAST_NUMERATOR),
            _amax_to_scale(amax_w, NVFP4_CAST_NUMERATOR),
        )
        output = output.reshape(*input_hp.shape[:-1], N)
        if bias is not None:
            output = output + bias

        ctx.save_for_backward(
            col_fp4_rht_x_t,
            col_sf_rht_x_t,
            amax_rht_x_t,
            row_fp4_w,
            row_sf_w,
            amax_w,
            wgrad_rht,
            dgrad_rht,
            sr_seed,
        )
        ctx.input_orig_shape = input_hp.shape
        ctx.has_bias = bias is not None
        ctx.use_cutedsl = use_cutedsl
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
            amax_w,
            wgrad_rht,
            dgrad_rht,
            sr_seed,
        ) = ctx.saved_tensors
        grad_output = grad_output.to(torch.bfloat16).contiguous()
        N = grad_output.shape[-1]
        dy_2d = grad_output.reshape(-1, N)
        M = dy_2d.shape[0]
        offsets = _degenerate_group_args(M, str(dy_2d.device))
        group_row_rht_col_rht_amax = (
            cutedsl_group_row_rht_col_rht_amax
            if ctx.use_cutedsl
            else triton_group_row_rht_col_rht_amax
        )
        group_row_rht_col_rht_quantize_ms_eden = (
            cutedsl_group_row_rht_col_rht_quantize_ms_eden
            if ctx.use_cutedsl
            else triton_group_row_rht_col_rht_quantize_ms_eden
        )
        group_col_rht_requant_amax = (
            cutedsl_group_col_rht_requant_amax
            if ctx.use_cutedsl
            else triton_group_col_rht_requant_amax
        )
        group_col_rht_requantize = (
            cutedsl_group_col_rht_requantize
            if ctx.use_cutedsl
            else triton_group_col_rht_requantize
        )

        # §11.2 -> §11.3. dgrad_rht rotates the row axis, wgrad_rht the transposed
        # one; swapping them is silent and yields a wrong gradient.
        amax_rht_dy, amax_rht_dy_t = group_row_rht_col_rht_amax(
            dy_2d, dgrad_rht, wgrad_rht, offsets, 1, M, N, VARYING_FIRST_DIM, offsets
        )
        (
            row_fp4_rht_dy,
            row_sf_rht_dy,
            col_fp4_rht_dy_t,
            col_sf_rht_dy_t,
        ) = group_row_rht_col_rht_quantize_ms_eden(
            dy_2d,
            amax_rht_dy,
            amax_rht_dy_t,
            dgrad_rht,
            wgrad_rht,
            offsets,
            1,
            M,
            N,
            VARYING_FIRST_DIM,
            _backward_rng_state(sr_seed),
            offsets,
        )

        # §11.4 -> §11.5: rebuild w.t() from the packed forward weight, rotated by
        # R_n so it cancels against dy's rotation in the dgrad GEMM.
        amax_rht_w_qdq_t = group_col_rht_requant_amax(
            row_fp4_w, row_sf_w, amax_w, dgrad_rht, 1
        )
        col_fp4_rht_w_t, col_sf_rht_w_t = group_col_rht_requantize(
            row_fp4_w, row_sf_w, amax_w, amax_rht_w_qdq_t, dgrad_rht, 1
        )

        # §13: (dy @ R_n) @ (w.t() @ R_n).t() = dy @ w.
        # MS-EDEN dy pairs with a cast weight -- 1536 with 2688, not one per GEMM.
        grad_input = _nvfp4_fp4_matmul(
            row_fp4_rht_dy,
            row_sf_rht_dy,
            col_fp4_rht_w_t[0],
            col_sf_rht_w_t,
            _amax_to_scale(amax_rht_dy, EDEN_NUMERATOR),
            _amax_to_scale(amax_rht_w_qdq_t, NVFP4_CAST_NUMERATOR),
        )
        grad_input = grad_input.reshape(ctx.input_orig_shape)

        # §14: (dy.t() @ R_m) @ (x.t() @ R_m).t() = dy.t() @ x.
        grad_weight = _nvfp4_fp4_matmul(
            col_fp4_rht_dy_t,
            col_sf_rht_dy_t,
            col_fp4_rht_x_t,
            col_sf_rht_x_t,
            _amax_to_scale(amax_rht_dy_t, EDEN_NUMERATOR),
            _amax_to_scale(amax_rht_x_t, NVFP4_CAST_NUMERATOR),
        )

        grad_bias = (
            grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
            if ctx.has_bias
            else None
        )
        # input_hp, weight_hp, bias, wgrad_rht, dgrad_rht, sr_seed, use_cutedsl,
        # use_fast_math
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


@torch._dynamo.allow_in_graph
class _NVFP4LinearV1Requant(torch.autograd.Function):
    """RHT-16 forward, stochastic-rounding backward, lazy unrotated requantization.

    Design doc §16. Kernel sequence::

        fwd:  grouped RHT-16 amax -> grouped RHT-16 quantize -> amax(w) -> §11.1 -> §12
        bwd:  grouped RHT-16 amax -> grouped RHT-16 quantize (SR) -> §11.6 -> §11.7
              -> §13 -> §14

    The forward is V2's with RHT-16 and a static sign vector. The recipes diverge
    only in backward. There is deliberately no ``dgrad_rht``: V1 applies no transform
    on the dgrad path, so there is nothing for one to cancel against.
    """

    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp: torch.Tensor,
        bias: Optional[torch.Tensor],
        sign_vector: tuple,
        sr_seed: torch.Tensor,
        use_cutedsl: bool = False,
        use_fast_math: bool = True,
    ):
        sign_vector = tuple(sign_vector)
        M = input_hp.shape[-2]
        K = input_hp.shape[-1]
        N = weight_hp.shape[0]
        _check_shapes(M, K, N)
        input_hp = input_hp.to(torch.bfloat16)
        weight_hp = weight_hp.to(torch.bfloat16)
        input_2d = input_hp.reshape(-1, K).contiguous()
        M = input_2d.shape[0]
        offsets = _degenerate_group_args(M, str(input_2d.device))
        sv = list(sign_vector)
        group_rht_amax = (
            cutedsl_group_rht_amax if use_cutedsl else triton_group_rht_amax
        )
        group_rht_quantize_row_col = (
            cutedsl_group_rht_quantize_row_col
            if use_cutedsl
            else triton_group_rht_quantize_row_col
        )

        amax_rht_x_t, amax_x = group_rht_amax(
            input_2d,
            sv,
            offsets,
            1,
            M,
            K,
            VARYING_FIRST_DIM,
            logical_packed_length=offsets,
        )
        (
            row_fp4_x,
            row_sf_x,
            col_fp4_rht_x_t,
            col_sf_rht_x_t,
        ) = group_rht_quantize_row_col(
            input_2d,
            sv,
            offsets,
            1,
            M,
            K,
            VARYING_FIRST_DIM,
            amax_x,
            amax_rht_x_t,
            None,
            False,
            offsets,
            use_fast_math,
        )

        row_fp4_w, row_sf_w, amax_w = _quantize_weight_rowwise(weight_hp, use_cutedsl)

        output = _nvfp4_fp4_matmul(
            row_fp4_x,
            row_sf_x,
            row_fp4_w[0],
            row_sf_w,
            _amax_to_scale(amax_x, NVFP4_CAST_NUMERATOR),
            _amax_to_scale(amax_w, NVFP4_CAST_NUMERATOR),
        )
        output = output.reshape(*input_hp.shape[:-1], N)
        if bias is not None:
            output = output + bias

        ctx.save_for_backward(
            col_fp4_rht_x_t,
            col_sf_rht_x_t,
            amax_rht_x_t,
            row_fp4_w,
            row_sf_w,
            amax_w,
            sr_seed,
        )
        ctx.input_orig_shape = input_hp.shape
        ctx.has_bias = bias is not None
        ctx.sign_vector = sign_vector
        ctx.use_cutedsl = use_cutedsl
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
            amax_w,
            sr_seed,
        ) = ctx.saved_tensors
        grad_output = grad_output.to(torch.bfloat16).contiguous()
        N = grad_output.shape[-1]
        dy_2d = grad_output.reshape(-1, N)
        M = dy_2d.shape[0]
        offsets = _degenerate_group_args(M, str(dy_2d.device))
        sv = list(ctx.sign_vector)
        group_rht_amax = (
            cutedsl_group_rht_amax if ctx.use_cutedsl else triton_group_rht_amax
        )
        group_rht_quantize_row_col = (
            cutedsl_group_rht_quantize_row_col
            if ctx.use_cutedsl
            else triton_group_rht_quantize_row_col
        )
        group_col_cast_requant_amax = (
            cutedsl_group_col_cast_requant_amax
            if ctx.use_cutedsl
            else triton_group_col_cast_requant_amax
        )
        group_col_cast_requantize = (
            cutedsl_group_col_cast_requantize
            if ctx.use_cutedsl
            else triton_group_col_cast_requantize
        )

        amax_rht_dy_t, amax_dy = group_rht_amax(
            dy_2d,
            sv,
            offsets,
            1,
            M,
            N,
            VARYING_FIRST_DIM,
            logical_packed_length=offsets,
        )
        (
            row_fp4_dy,
            row_sf_dy,
            col_fp4_rht_dy_t,
            col_sf_rht_dy_t,
        ) = group_rht_quantize_row_col(
            dy_2d,
            sv,
            offsets,
            1,
            M,
            N,
            VARYING_FIRST_DIM,
            amax_dy,
            amax_rht_dy_t,
            _backward_rng_state(sr_seed),
            True,  # stochastic rounding on dy
            offsets,
            ctx.use_fast_math,
        )

        # §11.6 -> §11.7: rebuild w.t() from the packed forward weight, unrotated.
        amax_w_qdq_t = group_col_cast_requant_amax(row_fp4_w, row_sf_w, amax_w, 1)
        col_fp4_w_t, col_sf_w_t = group_col_cast_requantize(
            row_fp4_w, row_sf_w, amax_w, amax_w_qdq_t, 1
        )

        # §13, no transform on this path: both operands are casts, both 2688.
        grad_input = _nvfp4_fp4_matmul(
            row_fp4_dy,
            row_sf_dy,
            col_fp4_w_t[0],
            col_sf_w_t,
            _amax_to_scale(amax_dy, NVFP4_CAST_NUMERATOR),
            _amax_to_scale(amax_w_qdq_t, NVFP4_CAST_NUMERATOR),
        )
        grad_input = grad_input.reshape(ctx.input_orig_shape)

        # §14: R_m cancels between the two operands, as in V2.
        grad_weight = _nvfp4_fp4_matmul(
            col_fp4_rht_dy_t,
            col_sf_rht_dy_t,
            col_fp4_rht_x_t,
            col_sf_rht_x_t,
            _amax_to_scale(amax_rht_dy_t, NVFP4_CAST_NUMERATOR),
            _amax_to_scale(amax_rht_x_t, NVFP4_CAST_NUMERATOR),
        )

        grad_bias = (
            grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
            if ctx.has_bias
            else None
        )
        # input_hp, weight_hp, bias, sign_vector, sr_seed, use_cutedsl, use_fast_math
        return grad_input, grad_weight, grad_bias, None, None, None, None


def nvfp4_linear_v2(
    input_hp: torch.Tensor,
    weight_hp: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    wgrad_rht: torch.Tensor,
    dgrad_rht: torch.Tensor,
    sr_seed: torch.Tensor,
    kernel_preference: KernelPreference = KernelPreference.AUTO,
    use_fast_math: bool = True,
) -> torch.Tensor:
    """``input @ weight.t() + bias`` under the V2 recipe.

    Args:
        input_hp: ``[..., in_features]``, cast to bfloat16 internally.
        weight_hp: ``[out_features, in_features]``.
        bias: optional ``[out_features]``.
        wgrad_rht: ``(128,)`` int8 sign buffer, resampled per accumulation microbatch.
        dgrad_rht: ``(128,)`` int8 sign buffer, resampled per optimizer step.
        sr_seed: one-element int64 CUDA tensor, the Philox key for MS-EDEN.
        kernel_preference: Backend for the quantize and amax ops, AUTO (default),
            TRITON, or CUTEDSL. AUTO takes CuteDSL when the runtime allows and falls
            back to Triton otherwise; CUTEDSL is the same choice made loudly, raising
            rather than falling back. The weight amax is Triton on every path -- it
            has no CuteDSL twin.
        use_fast_math: match TransformerEngine under ``NVTE_USE_FAST_MATH=1``.

    Both sign buffers must be the module-owned tensors that
    ``resample_nvfp4_rht_signs`` updates in place, not fresh allocations.
    """
    use_cutedsl = _resolve_use_cutedsl(kernel_preference)
    return _NVFP4LinearV2.apply(
        input_hp,
        weight_hp,
        bias,
        wgrad_rht,
        dgrad_rht,
        sr_seed,
        use_cutedsl,
        use_fast_math,
    )


def nvfp4_linear_v1_requant(
    input_hp: torch.Tensor,
    weight_hp: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    sign_vector,
    sr_seed: torch.Tensor,
    kernel_preference: KernelPreference = KernelPreference.AUTO,
    use_fast_math: bool = True,
) -> torch.Tensor:
    """``input @ weight.t() + bias`` under the V1_REQUANT recipe.

    Args:
        input_hp: ``[..., in_features]``, cast to bfloat16 internally.
        weight_hp: ``[out_features, in_features]``.
        bias: optional ``[out_features]``.
        sign_vector: static 16-element {-1, +1} tuple. Fixed for the whole run, so it
            resolves through the sign-keyed ``get_rht_matrix`` cache -- one entry.
        sr_seed: one-element int64 CUDA tensor, the Philox key for stochastic rounding.
        kernel_preference: Backend for the quantize and amax ops, AUTO (default),
            TRITON, or CUTEDSL. AUTO takes CuteDSL when the runtime allows and falls
            back to Triton otherwise; CUTEDSL is the same choice made loudly, raising
            rather than falling back. The weight amax is Triton on every path -- it
            has no CuteDSL twin.
        use_fast_math: match TransformerEngine under ``NVTE_USE_FAST_MATH=1``.
    """
    use_cutedsl = _resolve_use_cutedsl(kernel_preference)
    return _NVFP4LinearV1Requant.apply(
        input_hp,
        weight_hp,
        bias,
        tuple(sign_vector),
        sr_seed,
        use_cutedsl,
        use_fast_math,
    )
