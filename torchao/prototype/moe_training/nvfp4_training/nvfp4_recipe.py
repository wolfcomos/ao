# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 training recipe selection and the per-operand decode numerators.

Three recipes coexist. They differ in how the backward weight operand is built,
in the Hadamard size, and in how ``dy`` is rounded:

===============  ==================  ==============  ==================  ==========
Recipe           ``w.t()`` (dgrad)   Hadamard        ``dy`` rounding     Sign signs
===============  ==================  ==============  ==================  ==========
``V1``           2D 16x16 quantize   16              stochastic          static
``V1_REQUANT``   ``col_cast_requantize`` from the    stochastic          static
                 packed forward weight, 16
``V2``           ``col_rht_requantize`` from the     MS-EDEN             dynamic
                 packed forward weight, 128
===============  ==================  ==============  ==================  ==========

``V1`` is the shipped recipe and stays the default: selecting nothing must keep
producing what torchao produced before this module existed.
"""

from enum import Enum

import torch

# Per-tensor decode numerator for operands produced by a plain NVFP4 cast, whose
# block scales are capped at the E4M3 max of 448.
NVFP4_CAST_NUMERATOR = 448.0 * 6.0  # 2688.0

# Per-tensor decode numerator for MS-EDEN operands. MS-EDEN caps the block scale at
# 256 rather than 448 so the stochastically-rounded scale correction has headroom,
# which makes the decode numerator smaller by the same factor.
EDEN_BLOCK_SCALE_MAX = 256.0
EDEN_NUMERATOR = EDEN_BLOCK_SCALE_MAX * 6.0  # 1536.0


class NVFP4Recipe(str, Enum):
    """Which NVFP4 training recipe a linear or grouped GEMM runs."""

    V1 = "v1"
    """Shipped recipe: RHT-16, stochastic rounding, 2D 16x16 weight quantize.

    The 2D weight quantize emits one scale per 16x16 tile and broadcasts it to all
    16 rows, so ``W`` and ``W.T`` share a scale byte. That shared byte is what makes
    the forward and dgrad weight operands mutually consistent.
    """

    V1_REQUANT = "v1_requant"
    """RHT-16, stochastic rounding, 1D 1x16 weight quantize + lazy requantization.

    Reaches the same forward/dgrad consistency by a different route: the backward
    transpose is derived from the *dequantized forward weight*, so both GEMMs see
    one and the same ``W_qdq``. Forward stores only packed codes, swizzled E4M3
    scales and the scalar original-weight amax -- about 0.5625 bytes per element.
    """

    V2 = "v2"
    """RHT-128, MS-EDEN, 1D 1x16 weight quantize + lazy RHT requantization.

    Both gradient operands are rotated, with independent sign vectors resampled on
    a cadence, so the transform cancels inside each GEMM that uses it.
    """


def recipe_rht_size(recipe: NVFP4Recipe) -> int:
    """Hadamard dimension the recipe rotates by."""
    return 128 if recipe is NVFP4Recipe.V2 else 16


def recipe_has_dgrad_rht(recipe: NVFP4Recipe) -> bool:
    """Whether the recipe rotates the dgrad axis, and so needs a second sign vector.

    Only V2 does. V1 and V1_REQUANT apply no transform on the dgrad path, so there
    is nothing for a ``dgrad_rht`` to cancel against and none is drawn.
    """
    return recipe is NVFP4Recipe.V2


def recipe_uses_dynamic_signs(recipe: NVFP4Recipe) -> bool:
    """Whether sign vectors are resampled during the run.

    Static signs may be cached by value through ``get_rht_matrix``, whose
    ``lru_cache(maxsize=None)`` is sound precisely because the key set is one
    element. Dynamic signs must not: a cache keyed on a resampled vector grows one
    entry per resample for the run's lifetime.
    """
    return recipe is NVFP4Recipe.V2


def _amax_to_scale(amax: torch.Tensor, numerator: float) -> torch.Tensor:
    """Per-tensor decode scale ``amax / numerator`` for one GEMM operand.

    The numerator belongs to the *quantizer that produced this operand*, not to the
    GEMM: V2's dgrad pairs an MS-EDEN ``dy`` (1536) with a cast weight (2688) in a
    single ``scaled_mm``. Passing one numerator per GEMM leaves the forward correct
    and fails backward by roughly 40%, which is why every call site names its own.

    ``_amax_to_scale(a, NVFP4_CAST_NUMERATOR)`` is numerically identical to
    ``torchao.prototype.mx_formats.nvfp4_tensor.per_tensor_amax_to_scale``; it is
    spelled out here so the numerator is visible at the call site.
    """
    return amax.to(torch.float32) / numerator


def _require_cutedsl(kernel_preference) -> bool:
    """V2 and lazy weight requantization have a single, explicit kernel backend."""
    from torchao.quantization.quantize_.common import KernelPreference

    from .hadamard_cutedsl_utils import raise_if_cutedsl_nvfp4_unavailable

    if kernel_preference not in (KernelPreference.AUTO, KernelPreference.CUTEDSL):
        raise ValueError(
            "NVFP4 V2 and V1_REQUANT require kernel_preference AUTO or CUTEDSL; TRITON is not supported"
        )
    raise_if_cutedsl_nvfp4_unavailable("NVFP4 V2 / V1_REQUANT")
    return True


def _require_cutedsl_backends(kernel_preference, num_experts: int) -> tuple[bool, bool]:
    _require_cutedsl(kernel_preference)
    from ._cutedsl_group_kernels_impl import MAX_GROUPS

    if num_experts > MAX_GROUPS:
        raise ValueError(
            f"CuTeDSL supports at most {MAX_GROUPS} experts, got {num_experts}"
        )
    return True, True
