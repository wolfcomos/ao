# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuteDSL grouped rowwise-RHT + columnwise-RHT amax (SM100+).

Drop-in backend for ``triton_group_row_rht_col_rht_amax``: same signature, same
returns. One tcgen05 kernel applies both RHT-128 transforms to every 128x128 tile
from a single shared-memory copy of it and reduces both amaxes per group; see
``_cutedsl_group_kernels_impl``.
"""

from typing import Optional, Tuple

import torch

from .group_hadamard_utils import _validate_grouped_hadamard_inputs
from .hadamard_cutedsl_utils import raise_if_cutedsl_nvfp4_unavailable
from .hadamard_utils import _device_key, get_hadamard_matrix

RHT_SIZE = 128


@torch.library.custom_op("torchao::cutedsl_group_row_rht_col_rht_amax", mutates_args=())
def cutedsl_group_row_rht_col_rht_amax(
    dy: torch.Tensor,
    dgrad_rht: torch.Tensor,
    wgrad_rht: torch.Tensor,
    offsets: torch.Tensor,
    num_tensors: int,
    packed_sequence_length: int,
    hidden_size: int,
    shape_rep: int,
    logical_packed_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-group global amaxes for the V2 backward MS-EDEN operands (CuteDSL, SM100+).

    Signature and returns match ``triton_group_row_rht_col_rht_amax``. ``dy`` is the
    packed ``(packed_sequence_length, hidden_size)`` bfloat16 capacity buffer; rows at or
    after ``logical_packed_length == offsets[-1]`` are untouched allocation capacity and
    are never read. ``shape_rep`` is validated but does not reach the kernel: group
    membership is read from ``offsets`` alone, which is correct for both representations.

    Returns ``(amax_rht_dy, amax_rht_dy_t)``, each ``(num_tensors,)`` float32 -- rowwise
    first, as the Triton op (and unlike ``cutedsl_group_rht_amax``, whose rowwise value
    is untransformed). ``dgrad_rht`` rotates the row axis and ``wgrad_rht`` the transposed
    axis; a crossed pair yields a wrong gradient, not an error.

    NaN propagates to both amaxes of its group, as in the Triton kernel: the reduction uses
    ``max.NaN.f32`` and the cross-CTA atomic is a ``max.u32`` on the bit pattern, where a
    NaN outranks every finite float.

    Raises:
        NotImplementedError: pre-SM100 or a missing CuteDSL runtime.
        ValueError: bad shapes/dtypes, a sign tensor that is not a ``(128,)`` tensor on
            ``dy``'s device, or ``num_tensors`` above the kernel's group cap.
    """
    raise_if_cutedsl_nvfp4_unavailable("cutedsl_group_row_rht_col_rht_amax")

    from ._cutedsl_group_kernels_impl import (
        MAX_GROUPS,
        _cutedsl_group_row_rht_col_rht_amax_impl,
    )

    for name, sv in (("dgrad_rht", dgrad_rht), ("wgrad_rht", wgrad_rht)):
        if sv.ndim != 1 or sv.numel() != RHT_SIZE:
            raise ValueError(
                f"{name} must be a ({RHT_SIZE},) tensor, got shape {tuple(sv.shape)}"
            )
        if not sv.is_cuda or sv.device != dy.device:
            raise ValueError(f"{name} must be on the same device as dy")
    # H128 is symmetric, so ``H128 * signs[None, :]`` is ``get_dynamic_rht_matrix(signs).t()``
    # -- the (N, K) UMMA operand -- without the transpose copy; ``.to`` is a no-op for the
    # int8 / bfloat16 buffers the recipe passes and only converts other dtypes. Formed per
    # launch: the sign buffers are live tensors resampled in place, never a cache key.
    h128 = get_hadamard_matrix(RHT_SIZE, _device_key(dy.device), torch.bfloat16)
    row_rht_nk = torch.mul(h128, dgrad_rht[None, :]).to(torch.bfloat16)
    col_rht_nk = torch.mul(h128, wgrad_rht[None, :]).to(torch.bfloat16)
    _validate_grouped_hadamard_inputs(
        dy,
        row_rht_nk,
        offsets,
        num_tensors,
        packed_sequence_length,
        hidden_size,
        shape_rep,
        logical_packed_length,
        rht_size=RHT_SIZE,
    )
    if num_tensors > MAX_GROUPS:
        raise ValueError(
            f"num_tensors must be <= {MAX_GROUPS} for the CuteDSL grouped kernel, "
            f"got {num_tensors}"
        )

    return _cutedsl_group_row_rht_col_rht_amax_impl(
        dy,
        row_rht_nk,
        col_rht_nk,
        offsets,
        num_tensors,
        logical_packed_length=logical_packed_length,
    )


@cutedsl_group_row_rht_col_rht_amax.register_fake
def _(
    dy,
    dgrad_rht,
    wgrad_rht,
    offsets,
    num_tensors,
    packed_sequence_length,
    hidden_size,
    shape_rep,
    logical_packed_length=None,
):
    amax_rht_dy = dy.new_empty((num_tensors,), dtype=torch.float32)
    amax_rht_dy_t = dy.new_empty((num_tensors,), dtype=torch.float32)
    return amax_rht_dy, amax_rht_dy_t
