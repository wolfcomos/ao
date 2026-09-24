# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuteDSL grouped rowwise 1x16 NVFP4 E2M1 weight quantization (no RHT), SM100+.

Drop-in for ``triton_group_row_cast_quantize``: same signature, same output contract,
and byte-for-byte identical output. One launch covers the whole ``(E, M, N)`` stack --
experts are equal-sized and contiguous, so the expert is a grid coordinate and only the
per-expert global amax is expert-indexed. Weights never use stochastic rounding: RTNE
only, so there is no ``rng_state``.

A dense linear is the degenerate ``num_experts = 1`` case -- pass ``w.unsqueeze(0)``.
"""

from typing import Tuple

import torch

from .hadamard_cutedsl_utils import raise_if_cutedsl_nvfp4_unavailable


@torch.library.custom_op("torchao::cutedsl_group_row_cast_quantize", mutates_args=())
def cutedsl_group_row_cast_quantize(
    A: torch.Tensor,
    global_amax: torch.Tensor,
    num_tensors: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-expert rowwise 1x16 NVFP4 E2M1 weight quantization. RTNE, no RHT (CuteDSL, SM100+).

    Args:
        A: Dense ``(E, M, N)`` BF16 weights, contiguous. M and N must be
            divisible by 128.
        global_amax: ``(E,)`` float32 per-expert absolute maxima, as produced by
            ``triton_group_weight_amax``. Expert ``g`` is quantized with
            ``global_amax[g]``, never a reduction across experts.
        num_tensors: Number of experts; must equal ``E``.

    Returns:
        A 2-tuple matching ``triton_group_row_cast_quantize``:
          - ``(E, M, N//2)`` uint8 rowwise FP4 codes.
          - ``(E, M//128, N//64, 32, 16)`` float8_e4m3fn swizzled scales.

        The arity is two, not four: there is deliberately no columnwise output.

    Raises:
        NotImplementedError: pre-SM100 / missing CuteDSL runtime.
        ValueError: bad dtype/shape/storage, or M or N not divisible by 128.
    """
    raise_if_cutedsl_nvfp4_unavailable("cutedsl_group_row_cast_quantize")
    if A.dtype != torch.bfloat16:
        raise ValueError(f"Expected bfloat16, got {A.dtype}")
    if A.ndim != 3:
        raise ValueError("Tensor A must be 3-D")
    if not A.is_contiguous():
        raise ValueError("A must be contiguous")

    E, M, N = A.shape
    if E != num_tensors:
        raise ValueError(f"Expected {num_tensors} experts, got {E}")
    if global_amax.shape != (E,):
        raise ValueError(f"global_amax must have shape ({E},)")
    if global_amax.dtype != torch.float32:
        raise ValueError(f"Expected float32 global_amax, got {global_amax.dtype}")
    if not global_amax.is_cuda or global_amax.device != A.device:
        raise ValueError("global_amax must be on the same device as A")
    if not global_amax.is_contiguous():
        raise ValueError("global_amax must be contiguous")
    if M % 128 != 0 or N % 128 != 0:
        raise ValueError(
            f"Expected M divisible by 128 and N divisible by 128, got M={M}, N={N}"
        )

    from ._cutedsl_kernels_impl import _cutedsl_group_row_cast_quantize_impl

    return _cutedsl_group_row_cast_quantize_impl(A, global_amax)


@cutedsl_group_row_cast_quantize.register_fake
def _(A, global_amax, num_tensors):
    E, M, N = A.shape
    qa = A.new_empty((E, M, N // 2), dtype=torch.uint8)
    sfa = A.new_empty((E, M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn)
    return qa, sfa
