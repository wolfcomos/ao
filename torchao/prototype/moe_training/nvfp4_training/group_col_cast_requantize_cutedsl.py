# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuteDSL grouped lazy columnwise NVFP4 weight requantization (SM100+).

Drop-in backends for ``triton_group_col_cast_requant_amax`` (§11.6) and
``triton_group_col_cast_requantize`` (§11.7): same signatures, same returns. Both kernels
read the packed forward weight -- the rowwise codes and swizzled scales of
``group_row_cast_quantize`` -- and never the BF16 weight, so the amax bounds the very
``W_qdq`` the dgrad operand is requantized from (the invariant the Triton module
docstring calls load-bearing); see ``_cutedsl_kernels_impl``.

No sign vector: these apply no transform.
"""

from typing import Tuple

import torch

from .group_hadamard_utils import (
    _validate_requant_amax,
    _validate_requant_weight_inputs,
)
from .hadamard_cutedsl_utils import raise_if_cutedsl_nvfp4_unavailable


@torch.library.custom_op(
    "torchao::cutedsl_group_col_cast_requant_amax", mutates_args=()
)
def cutedsl_group_col_cast_requant_amax(
    row_fp4_w: torch.Tensor,
    row_sf_w: torch.Tensor,
    global_amax: torch.Tensor,
    num_tensors: int,
) -> torch.Tensor:
    """Per-expert amax of the dequantized forward weight, transposed (CuteDSL, SM100+).

    Signature and returns match ``triton_group_col_cast_requant_amax``: ``(E,)`` float32
    ``out[g] = dequantize(row_fp4_w[g]).bf16().abs().amax()``, computed from the
    *quantized* weight and bitwise the Triton op's for finite scale bytes (a NaN scale
    byte, which §11.1 never emits, is NaN in both backends but with a different
    payload). An expert whose ``global_amax`` is NaN or inf reconstructs to zero and
    reports 0.0, as the Triton kernel does.

    Raises:
        NotImplementedError: pre-SM100 or a missing CuteDSL runtime.
        ValueError: bad shapes/dtypes, or M, N not divisible by 128.
    """
    raise_if_cutedsl_nvfp4_unavailable("cutedsl_group_col_cast_requant_amax")
    _validate_requant_weight_inputs(
        row_fp4_w,
        row_sf_w,
        global_amax,
        num_tensors,
        "cutedsl_group_col_cast_requant_amax",
    )

    from ._cutedsl_kernels_impl import _cutedsl_group_col_cast_requant_amax_impl

    return _cutedsl_group_col_cast_requant_amax_impl(row_fp4_w, row_sf_w, global_amax)


@cutedsl_group_col_cast_requant_amax.register_fake
def _(row_fp4_w, row_sf_w, global_amax, num_tensors):
    return row_fp4_w.new_empty((row_fp4_w.shape[0],), dtype=torch.float32)


@torch.library.custom_op("torchao::cutedsl_group_col_cast_requantize", mutates_args=())
def cutedsl_group_col_cast_requantize(
    row_fp4_w: torch.Tensor,
    row_sf_w: torch.Tensor,
    global_amax: torch.Tensor,
    amax_w_qdq_t: torch.Tensor,
    num_tensors: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-expert columnwise NVFP4 requantization of the forward weight (CuteDSL, SM100+).

    Signature and returns match ``triton_group_col_cast_requantize``: ``(E, N, M//2)``
    uint8 codes (rowwise ``W_qdq.T``) and ``(E, N//128, M//64, 32, 16)`` float8_e4m3fn
    swizzled scales, bitwise the Triton op's. ``amax_w_qdq_t`` must come from §11.6. The
    transpose is a plain SMEM gather, not an MMA against an identity, so the sign of an
    exact zero survives as in Triton.

    Raises:
        NotImplementedError: pre-SM100 or a missing CuteDSL runtime.
        ValueError: bad shapes/dtypes, or M, N not divisible by 128.
    """
    raise_if_cutedsl_nvfp4_unavailable("cutedsl_group_col_cast_requantize")
    E, M, N = _validate_requant_weight_inputs(
        row_fp4_w,
        row_sf_w,
        global_amax,
        num_tensors,
        "cutedsl_group_col_cast_requantize",
    )
    _validate_requant_amax(amax_w_qdq_t, "amax_w_qdq_t", E, row_fp4_w.device)

    from ._cutedsl_kernels_impl import _cutedsl_group_col_cast_requantize_impl

    return _cutedsl_group_col_cast_requantize_impl(
        row_fp4_w, row_sf_w, global_amax, amax_w_qdq_t
    )


@cutedsl_group_col_cast_requantize.register_fake
def _(row_fp4_w, row_sf_w, global_amax, amax_w_qdq_t, num_tensors):
    E, M, packed_N = row_fp4_w.shape
    N = packed_N * 2
    qa_t = row_fp4_w.new_empty((E, N, M // 2), dtype=torch.uint8)
    sfa_t = row_fp4_w.new_empty(
        (E, N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn
    )
    return qa_t, sfa_t
