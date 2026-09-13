# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuteDSL grouped rotated columnwise weight requantization (SM100+).

Drop-in backends for ``triton_group_col_rht_requant_amax`` (§11.4) and
``triton_group_col_rht_requantize`` (§11.5): same signatures, same returns. Both
kernels rebuild the dequantized weight tile on chip from the rowwise codes and
scales, rotate its transpose by ``R_n`` through one tcgen05 chain and share that
producer, as the Triton twins share ``_load_rht_requant_weight_tile``; see
``_cutedsl_group_kernels_impl``.
"""

from typing import Tuple

import torch

from .group_hadamard_utils import (
    _validate_requant_amax,
    _validate_requant_weight_inputs,
)
from .hadamard_cutedsl_utils import raise_if_cutedsl_nvfp4_unavailable
from .hadamard_utils import _device_key, get_hadamard_matrix

RHT_SIZE = 128


def _sign_operand(dgrad_rht: torch.Tensor, device: torch.device) -> torch.Tensor:
    if dgrad_rht.ndim != 1 or dgrad_rht.numel() != RHT_SIZE:
        raise ValueError(
            f"dgrad_rht must be a ({RHT_SIZE},) tensor, "
            f"got shape {tuple(dgrad_rht.shape)}"
        )
    if not dgrad_rht.is_cuda or dgrad_rht.device != device:
        raise ValueError("dgrad_rht must be on the same device as row_fp4_w")
    # ``.to`` is a no-op for the recipe's int8 buffers and converts bf16/f32 signs
    # as the Triton op would. ``.contiguous`` makes a strided sign view a plain buffer,
    # as the Triton op's ``sign.to(bf16)[:, None] * H`` does implicitly; both calls are
    # no-ops for the recipe's contiguous int8 buffer (the same object, so CUDA-graph
    # addresses stay stable).
    return dgrad_rht.to(torch.int8).contiguous()


@torch.library.custom_op("torchao::cutedsl_group_col_rht_requant_amax", mutates_args=())
def cutedsl_group_col_rht_requant_amax(
    row_fp4_w: torch.Tensor,
    row_sf_w: torch.Tensor,
    global_amax: torch.Tensor,
    dgrad_rht: torch.Tensor,
    num_tensors: int,
) -> torch.Tensor:
    """Per-expert amax of the rotated dequantized forward weight transpose (CuteDSL, SM100+).

    Signature and returns match ``triton_group_col_rht_requant_amax``: ``(E,)`` float32
    ``out[g] = amax(abs(dequantize(row_fp4_w[g]).bf16().t() @ R_n))``, computed from the
    *quantized* weight. An expert whose ``global_amax`` is NaN or inf reconstructs to
    zero and reports 0.0, as the Triton kernel does.

    Raises:
        NotImplementedError: pre-SM100 or a missing CuteDSL runtime.
        ValueError: bad shapes/dtypes, or a ``dgrad_rht`` that is not a ``(128,)`` tensor
            on ``row_fp4_w``'s device.
    """
    raise_if_cutedsl_nvfp4_unavailable("cutedsl_group_col_rht_requant_amax")
    _validate_requant_weight_inputs(
        row_fp4_w,
        row_sf_w,
        global_amax,
        num_tensors,
        "cutedsl_group_col_rht_requant_amax",
    )
    signs = _sign_operand(dgrad_rht, row_fp4_w.device)
    h128 = get_hadamard_matrix(RHT_SIZE, _device_key(row_fp4_w.device), torch.bfloat16)

    from ._cutedsl_group_kernels_impl import _cutedsl_group_col_rht_requant_amax_impl

    return _cutedsl_group_col_rht_requant_amax_impl(
        row_fp4_w, row_sf_w, global_amax, signs, h128, num_tensors
    )


@cutedsl_group_col_rht_requant_amax.register_fake
def _(row_fp4_w, row_sf_w, global_amax, dgrad_rht, num_tensors):
    return row_fp4_w.new_empty((row_fp4_w.shape[0],), dtype=torch.float32)


@torch.library.custom_op("torchao::cutedsl_group_col_rht_requantize", mutates_args=())
def cutedsl_group_col_rht_requantize(
    row_fp4_w: torch.Tensor,
    row_sf_w: torch.Tensor,
    global_amax: torch.Tensor,
    amax_rht_w_qdq_t: torch.Tensor,
    dgrad_rht: torch.Tensor,
    num_tensors: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-expert rotated columnwise NVFP4 requantization of the weight (CuteDSL, SM100+).

    Signature and returns match ``triton_group_col_rht_requantize``: ``(E, N, M//2)``
    uint8 codes and ``(E, N//128, M//64, 32, 16)`` float8_e4m3fn swizzled scales, to be
    decoded with ``NVFP4_CAST_NUMERATOR`` (2688). ``amax_rht_w_qdq_t`` must come from
    §11.4 with the same ``dgrad_rht``.

    Raises:
        NotImplementedError: pre-SM100 or a missing CuteDSL runtime.
        ValueError: bad shapes/dtypes, or a ``dgrad_rht`` that is not a ``(128,)`` tensor
            on ``row_fp4_w``'s device.
    """
    raise_if_cutedsl_nvfp4_unavailable("cutedsl_group_col_rht_requantize")
    E, M, N = _validate_requant_weight_inputs(
        row_fp4_w,
        row_sf_w,
        global_amax,
        num_tensors,
        "cutedsl_group_col_rht_requantize",
    )
    _validate_requant_amax(amax_rht_w_qdq_t, "amax_rht_w_qdq_t", E, row_fp4_w.device)
    signs = _sign_operand(dgrad_rht, row_fp4_w.device)
    h128 = get_hadamard_matrix(RHT_SIZE, _device_key(row_fp4_w.device), torch.bfloat16)

    from ._cutedsl_group_kernels_impl import _cutedsl_group_col_rht_requantize_impl

    return _cutedsl_group_col_rht_requantize_impl(
        row_fp4_w, row_sf_w, global_amax, amax_rht_w_qdq_t, signs, h128, num_tensors
    )


@cutedsl_group_col_rht_requantize.register_fake
def _(row_fp4_w, row_sf_w, global_amax, amax_rht_w_qdq_t, dgrad_rht, num_tensors):
    E, M, packed_N = row_fp4_w.shape
    N = packed_N * 2
    qa_t = row_fp4_w.new_empty((E, N, M // 2), dtype=torch.uint8)
    sfa_t = row_fp4_w.new_empty(
        (E, N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn
    )
    return qa_t, sfa_t
