# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuteDSL grouped RHT global amax reduction (SM100+).

Drop-in backend for ``triton_group_rht_amax``: same signature, same returns.
Shares the TransformerEngine-derived mainloop with the grouped fused quantize
kernel, with max-reduce epilogues in place of the quantize epilogues; see
``_cutedsl_group_kernels_impl``.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .group_hadamard_utils import _validate_grouped_hadamard_inputs
from .hadamard_cutedsl_utils import raise_if_cutedsl_nvfp4_unavailable
from .hadamard_utils import _device_key, get_hadamard_matrix, get_rht_matrix

_DEFAULT_SCALING_TYPE = (
    int(F.ScalingType.TensorWise) if hasattr(F, "ScalingType") else 0
)
RHT_SIZE = 128


@torch.library.custom_op("torchao::cutedsl_group_rht_amax", mutates_args=())
def cutedsl_group_rht_amax(
    A: torch.Tensor,
    sign_vector: List[int],
    offsets: torch.Tensor,
    num_tensors: int,
    packed_sequence_length: int,
    hidden_size: int,
    shape_rep: int,
    scaling_type: int = _DEFAULT_SCALING_TYPE,
    logical_packed_length: Optional[torch.Tensor] = None,
    sign_tensor: Optional[torch.Tensor] = None,
    dynamic_rht: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-group post-RHT columnwise amax and raw rowwise amax (graph-safe).

    Signature and returns match ``triton_group_rht_amax``. ``A`` is the packed
    ``(packed_sequence_length, hidden_size)`` bfloat16 capacity buffer; rows at
    or after ``logical_packed_length == offsets[-1]`` are untouched allocation
    capacity and must not be consumed. Rows before it, including zero-valued
    per-group padding, are processed normally.

    ``shape_rep`` is validated but does not reach the kernel: group membership
    is read from ``offsets`` alone, which is correct for both SAME_BOTH_DIMS and
    VARYING_FIRST_DIM.

    ``dynamic_rht=True`` selects the V2 RHT-128 path: ``sign_tensor`` is the live
    ``(128,)`` sign buffer, ``sign_vector`` is ignored (pass ``[]``), and
    ``R^T = H128 * signs[None, :]`` is formed per launch, never memoized. Only the
    ``(128,)`` buffer is served; the Triton op also accepts a ``(16,)`` one.

    Returns ``(col_amax, row_amax)``, each ``(num_tensors,)`` float32, where
    ``col_amax[g] = max|RHT(A_g.t())|`` and ``row_amax[g] = max|A_g|``.

    NaN propagates to both amaxes, as in the Triton and non-grouped CuteDSL
    kernels: the reduction uses ``max.NaN.f32`` and the cross-CTA atomic is a
    ``max.u32`` on the bit pattern, where a NaN outranks every finite float.

    Raises:
        NotImplementedError: pre-SM100 or a missing CuteDSL runtime.
        ValueError: bad shapes/dtypes, ``num_tensors`` above the kernel's group
            cap, or an unsupported ``scaling_type``.
    """
    raise_if_cutedsl_nvfp4_unavailable("cutedsl_group_rht_amax")

    from ._cutedsl_group_kernels_impl import (
        MAX_GROUPS,
        _cutedsl_group_rht_amax_impl,
        _cutedsl_group_row_cast_col_rht_amax_impl,
    )

    if dynamic_rht:
        if sign_tensor is None:
            raise ValueError("dynamic_rht=True requires a sign_tensor")
        if sign_tensor.ndim != 1 or sign_tensor.numel() != RHT_SIZE:
            raise ValueError(
                f"sign_tensor must be a ({RHT_SIZE},) tensor, got shape {tuple(sign_tensor.shape)}"
            )
        if not sign_tensor.is_cuda or sign_tensor.device != A.device:
            raise ValueError("sign_tensor must be on the same device as A")
        # H128 is symmetric, so ``H128 * signs[None, :]`` is ``get_dynamic_rht_matrix(signs).t()``
        # -- the (N, K) UMMA operand -- and the kernel forms it in SMEM from the cached
        # ``H128`` and the signs at every launch: the sign buffer is a live tensor resampled
        # in place, never a cache key. ``.to`` / ``.contiguous`` are no-ops for the recipe's
        # int8 buffer (the same object, so CUDA-graph addresses stay stable) and convert a
        # bf16 / f32 or strided sign view as the Triton op's ``sign.to(bf16)[:, None] * H``
        # does implicitly.
        signs = sign_tensor.to(torch.int8).contiguous()
        B = get_hadamard_matrix(RHT_SIZE, _device_key(A.device), torch.bfloat16)
    elif sign_tensor is not None:
        raise ValueError("sign_tensor is only used when dynamic_rht=True")
    else:
        B = get_rht_matrix(
            tuple(sign_vector), _device_key(A.device), torch.bfloat16, 16
        )
    _validate_grouped_hadamard_inputs(
        A,
        B,
        offsets,
        num_tensors,
        packed_sequence_length,
        hidden_size,
        shape_rep,
        logical_packed_length,
        rht_size=B.shape[0],
    )
    if num_tensors > MAX_GROUPS:
        raise ValueError(
            f"num_tensors must be <= {MAX_GROUPS} for the CuteDSL grouped kernel, "
            f"got {num_tensors}"
        )
    if scaling_type != _DEFAULT_SCALING_TYPE:
        raise ValueError(
            f"scaling_type={scaling_type!r} is not supported; "
            "only ScalingType.TensorWise is implemented."
        )

    if dynamic_rht:
        return _cutedsl_group_row_cast_col_rht_amax_impl(
            A,
            signs,
            B,
            offsets,
            num_tensors,
            logical_packed_length=logical_packed_length,
        )
    return _cutedsl_group_rht_amax_impl(
        A,
        offsets,
        num_tensors,
        tuple(sign_vector),
        logical_packed_length=logical_packed_length,
    )


@cutedsl_group_rht_amax.register_fake
def _(
    A,
    sign_vector,
    offsets,
    num_tensors,
    packed_sequence_length,
    hidden_size,
    shape_rep,
    scaling_type=_DEFAULT_SCALING_TYPE,
    logical_packed_length=None,
    sign_tensor=None,
    dynamic_rht=False,
):
    col_amax = A.new_empty((num_tensors,), dtype=torch.float32)
    row_amax = A.new_empty((num_tensors,), dtype=torch.float32)
    return col_amax, row_amax
