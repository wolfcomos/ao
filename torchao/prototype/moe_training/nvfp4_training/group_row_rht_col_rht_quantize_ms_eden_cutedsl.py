# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuteDSL grouped MS-EDEN quantize, RHT-128 on both axes (SM100+).

Drop-in backend for ``triton_group_row_rht_col_rht_quantize_ms_eden``: same signature, same
four returns in the same rowwise-first order, decoded with numerator 1536. One tcgen05 kernel
applies both RHT-128 transforms to every 128x128 tile from a single shared-memory copy of it
(the ``cutedsl_group_row_rht_col_rht_amax`` mainloop) and quantizes both accumulators in
MS-EDEN's two steps -- RTNE codes, then a corrected, stochastically rounded E4M3 block
scale; see ``_cutedsl_group_kernels_impl``.
"""

from typing import Optional, Tuple

import torch

from .group_hadamard_utils import (
    _validate_graph_amax,
    _validate_grouped_hadamard_inputs,
    _validate_rng_state,
)
from .hadamard_cutedsl_utils import raise_if_cutedsl_nvfp4_unavailable
from .hadamard_utils import _device_key, get_hadamard_matrix

RHT_SIZE = 128


@torch.library.custom_op(
    "torchao::cutedsl_group_row_rht_col_rht_quantize_ms_eden", mutates_args=()
)
def cutedsl_group_row_rht_col_rht_quantize_ms_eden(
    dy: torch.Tensor,
    amax_rht_dy: torch.Tensor,
    amax_rht_dy_t: torch.Tensor,
    dgrad_rht: torch.Tensor,
    wgrad_rht: torch.Tensor,
    offsets: torch.Tensor,
    num_tensors: int,
    packed_sequence_length: int,
    hidden_size: int,
    shape_rep: int,
    rng_state: torch.Tensor,
    logical_packed_length: Optional[torch.Tensor] = None,
    fast_path: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-group MS-EDEN quantization of the rotated gradient and its transpose (CuteDSL, SM100+).

    Signature and returns match ``triton_group_row_rht_col_rht_quantize_ms_eden``. ``dy`` is the
    packed ``(packed_sequence_length, hidden_size)`` bfloat16 capacity buffer; rows at or after
    ``logical_packed_length == offsets[-1]`` are never read and their outputs are left as
    allocated. ``amax_rht_dy`` / ``amax_rht_dy_t`` are the two returns of
    ``*_group_row_rht_col_rht_amax``; ``dgrad_rht`` rotates the row axis and ``wgrad_rht`` the
    transposed axis (a crossed pair yields a wrong gradient, not an error). ``shape_rep`` is
    validated but does not reach the kernel: group membership is read from ``offsets`` alone.

    ``rng_state`` is the int64 ``[col_seed, col_offset, row_seed, row_offset]`` the Triton op
    takes; MS-EDEN always draws, so it is required.

    ``fast_path=True`` selects the hardware stochastic rounding variant: one Philox counter per
    16 scales of a row (``randint4x``) and one ``cvt.rs.satfinite.e4m3x4.f32`` for the four
    scales a warp holds, a different random-bit construction from the software rounding. Its
    scale bytes are therefore not bitwise with the Triton op's -- each lands on one of the two
    E4M3 neighbours of the same corrected scale -- while the codes are identical. That holds for
    blocks whose rotated values keep the fast cross dot finite (guaranteed for |value| below
    ~3.5e36, the onset for a block of sixteen saturated codes; blocks with fewer large values
    overflow later); where it overflows, the fast path stores the uncorrected E4M3 scale, exact
    under the stochastic rounding. ``False`` (the default) is bitwise with the Triton op.

    Returns ``(row_fp4_rht_dy, row_sf_rht_dy, col_fp4_rht_dy_t, col_sf_rht_dy_t)`` -- rowwise
    first: ``(psl, hidden//2)`` uint8, ``(psl, hidden//16)`` float8_e4m3fn (a view of the
    swizzled storage), ``(hidden, psl//2)`` uint8, ``(hidden, psl//16)`` float8_e4m3fn (per-group
    swizzled). Decode both with ``EDEN_NUMERATOR`` (1536), never 2688. Inputs are finite.

    Raises:
        NotImplementedError: pre-SM100 or a missing CuteDSL runtime.
        TypeError: ``rng_state`` is not a tensor.
        ValueError: bad shapes/dtypes, a sign tensor that is not a ``(128,)`` tensor on
            ``dy``'s device, or ``num_tensors`` above the kernel's group cap.
    """
    raise_if_cutedsl_nvfp4_unavailable("cutedsl_group_row_rht_col_rht_quantize_ms_eden")

    from ._cutedsl_group_kernels_impl import (
        MAX_GROUPS,
        _cutedsl_group_row_rht_col_rht_quantize_ms_eden_impl,
    )

    for name, sv in (("dgrad_rht", dgrad_rht), ("wgrad_rht", wgrad_rht)):
        if sv.ndim != 1 or sv.numel() != RHT_SIZE:
            raise ValueError(
                f"{name} must be a ({RHT_SIZE},) tensor, got shape {tuple(sv.shape)}"
            )
        if not sv.is_cuda or sv.device != dy.device:
            raise ValueError(f"{name} must be on the same device as dy")
    # H128 is symmetric, so ``H128 * signs[None, :]`` is ``get_dynamic_rht_matrix(signs).t()``
    # -- the (N, K) UMMA operand -- without the transpose copy; the kernel forms both from
    # ``h128`` and the live sign buffers (an exact bf16 sign flip). ``.to`` is a no-op for the
    # int8 buffers the recipe passes and only converts other dtypes; ``.contiguous`` makes a
    # strided sign view a plain buffer. Formed per launch: the sign buffers are live tensors
    # resampled in place, never a cache key.
    h128 = get_hadamard_matrix(RHT_SIZE, _device_key(dy.device), torch.bfloat16)
    _validate_grouped_hadamard_inputs(
        dy,
        h128,
        offsets,
        num_tensors,
        packed_sequence_length,
        hidden_size,
        shape_rep,
        logical_packed_length,
        rht_size=RHT_SIZE,
    )
    row_amax = _validate_graph_amax(amax_rht_dy, "amax_rht_dy", num_tensors, dy.device)
    col_amax = _validate_graph_amax(
        amax_rht_dy_t, "amax_rht_dy_t", num_tensors, dy.device
    )
    # MS-EDEN always draws, so SR is unconditionally on for validation purposes.
    rng_state = _validate_rng_state(rng_state, dy.device, True)
    if num_tensors > MAX_GROUPS:
        raise ValueError(
            f"num_tensors must be <= {MAX_GROUPS} for the CuteDSL grouped kernel, "
            f"got {num_tensors}"
        )

    row_fp4, row_sf, col_fp4, col_sf = (
        _cutedsl_group_row_rht_col_rht_quantize_ms_eden_impl(
            dy,
            h128,
            dgrad_rht.to(torch.int8).contiguous(),
            wgrad_rht.to(torch.int8).contiguous(),
            row_amax,
            col_amax,
            offsets,
            num_tensors,
            rng_state,
            logical_packed_length=logical_packed_length,
            fast_path=fast_path,
        )
    )
    # Rowwise pair first, matching every sibling quantize op.
    return (
        row_fp4,
        row_sf.view(packed_sequence_length, hidden_size // 16),
        col_fp4,
        col_sf.view(hidden_size, packed_sequence_length // 16),
    )


@cutedsl_group_row_rht_col_rht_quantize_ms_eden.register_fake
def _(
    dy,
    amax_rht_dy,
    amax_rht_dy_t,
    dgrad_rht,
    wgrad_rht,
    offsets,
    num_tensors,
    packed_sequence_length,
    hidden_size,
    shape_rep,
    rng_state,
    logical_packed_length=None,
    fast_path=False,
):
    qd = dy.new_empty((hidden_size, packed_sequence_length // 2), dtype=torch.uint8)
    sfd = dy.new_empty(
        (hidden_size, packed_sequence_length // 16), dtype=torch.float8_e4m3fn
    )
    qa_base = dy.new_empty(
        (packed_sequence_length, hidden_size // 2), dtype=torch.uint8
    )
    sfa = dy.new_empty(
        (packed_sequence_length, hidden_size // 16), dtype=torch.float8_e4m3fn
    )
    return qa_base, sfa, qd, sfd
