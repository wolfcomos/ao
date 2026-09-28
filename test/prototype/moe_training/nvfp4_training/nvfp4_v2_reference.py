# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Independent PyTorch references for NVFP4 V2 and lazy weight requantization.

V2's NaN handling, dequantization order and grouped layouts live here so changes
needed by CuTeDSL V2 tests cannot change the established V1 oracle. Shared data
structures and unchanged primitives come from ``nvfp4_reference``; arithmetic
that needs the V2 contract is defined locally. MS-EDEN software scales use a
PyTorch Philox implementation, while hardware-SR scales have property tests.
"""

from dataclasses import dataclass
from typing import Tuple

import torch

from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    get_dynamic_rht_matrix,
)
from torchao.prototype.mx_formats.kernels import f32_to_f4_unpacked, pack_uint4
from torchao.prototype.mx_formats.utils import from_blocked, to_blocked

from .nvfp4_reference import (
    _FP32_MAX,
    FP4_E2M1_MAX,
    FP8_E4M3_MAX,
    NVFP4ReferenceOutput,
    _group_sizes,
)


def to_blocked_grouped(plain: torch.Tensor, group_sizes) -> torch.Tensor:
    """``to_blocked`` for a scale tensor whose *inner* axis is grouped.

    A grouped GEMM reads each group's block scales as an independently blocked buffer,
    the buffers concatenated flat. Blocking the whole extent instead scatters each
    group's tiles through the buffer and the GEMM then reads them from the wrong offset,
    so the tiling has to restart at every group boundary.

    The outer axis needs no equivalent: it is the slowest-varying term, so a group
    occupying whole 128-row tiles is already contiguous and plain ``to_blocked`` works.

    Args:
        plain: (rows, sum(group_sizes) // 16) scale tensor in row-major layout.
        group_sizes: per-group element counts along the grouped axis; each must be a
            multiple of 16, which is what makes the group-local tile index exact.

    Returns:
        Flat 1-D tensor, the groups' blocked buffers concatenated.
    """
    parts, col = [], 0
    for size in group_sizes:
        width = size // 16
        if width == 0:
            continue
        parts.append(to_blocked(plain[:, col : col + width]).flatten())
        col += width
    return torch.cat(parts)


def from_blocked_grouped(blocked: torch.Tensor, rows: int, group_sizes) -> torch.Tensor:
    """Inverse of ``to_blocked_grouped``: grouped blocked buffer -> plain (rows, cols).

    Each group was blocked over its own extent, so each has to be un-blocked over its
    own extent too. Group row counts are 128-aligned, which makes every group's width
    a multiple of 4 and therefore its blocked buffer exactly ``rows * width`` elements
    with no tile padding to skip.
    """
    flat = blocked.flatten()
    parts, pos = [], 0
    for size in group_sizes:
        width = size // 16
        count = rows * width
        if width == 0:
            continue
        parts.append(
            from_blocked(flat[pos : pos + count].reshape(rows, width), rows, width)
        )
        pos += count
    return torch.cat(parts, dim=1)


def global_encode_scale(
    global_amax: torch.Tensor, fp8_max: float = FP8_E4M3_MAX
) -> torch.Tensor:
    """``compute_global_encode_scaling_factor_FP4``: ``fp8_max * 6 / amax``, guarded.

    ``fp8_max`` is 448 for every plain NVFP4 cast, giving the familiar 2688 numerator.
    MS-EDEN operands pass 256, giving 1536: their block-scale ceiling is lower so the
    stochastically-rounded scale correction has headroom.
    """
    amax = global_amax.to(torch.float32)
    candidate = torch.full_like(amax, fp8_max * FP4_E2M1_MAX) / amax
    candidate = torch.fmin(candidate, torch.full_like(candidate, _FP32_MAX))
    # amax == 0 gives inf; an enormous amax underflows the scale to zero. Both -> identity.
    return torch.where(
        (amax == 0.0) | (candidate == 0.0), torch.ones_like(candidate), candidate
    )


def _block_scale(
    block_amax: torch.Tensor, s_enc: torch.Tensor, fp8_max: float = FP8_E4M3_MAX
) -> torch.Tensor:
    """``compute_decoding_scaling_factor``: one rounding, upper clamp only."""
    scale = block_amax * (s_enc * (1.0 / FP4_E2M1_MAX))
    return torch.fmin(scale, torch.full_like(scale, fp8_max)).to(torch.float8_e4m3fn)


def _encode_scale(block_scale_fp8: torch.Tensor, s_enc: torch.Tensor) -> torch.Tensor:
    """Correctly rounded reciprocal of the effective decode scale."""
    denom = block_scale_fp8.to(torch.float32) * (1.0 / s_enc)
    value = 1.0 / denom
    return torch.fmin(value, torch.full_like(value, _FP32_MAX))


def pack_fp4(scaled: torch.Tensor) -> torch.Tensor:
    """(R, C) f32 -> (R, C//2) uint8, low nibble = even element."""
    clamped = torch.nan_to_num(scaled, nan=FP4_E2M1_MAX).clamp(
        -FP4_E2M1_MAX, FP4_E2M1_MAX
    )
    return pack_uint4(f32_to_f4_unpacked(clamped.contiguous()))


def _block_amax(x: torch.Tensor, block_rows: int) -> torch.Tensor:
    """Per-tile amax over ``(block_rows, 16)`` tiles -> (R//block_rows, C//16) f32."""
    rows, cols = x.shape
    tiles = x.abs().reshape(rows // block_rows, block_rows, cols // 16, 16)
    finite = torch.where(torch.isnan(tiles), -torch.inf, tiles)
    result = finite.amax(dim=(1, 3))
    return torch.where(torch.isnan(tiles).all(dim=(1, 3)), torch.nan, result)


def nvfp4_reference_quantize(
    x: torch.Tensor,
    global_amax: torch.Tensor,
    *,
    block: str = "1x16",
    layout: str = "plain",
    fp8_max: float = FP8_E4M3_MAX,
) -> NVFP4ReferenceOutput:
    """NVFP4 quantize a 2-D tensor with TE's arithmetic.

    Args:
        x: (R, C) bfloat16 or float32. Upcast to float32, which is lossless from bf16 and
            is what the kernels do before the block reduction.
        global_amax: scalar float32 tensor-wide amax.
        block: ``"1x16"`` (activations) or ``"16x16"`` (2D weight scaling).
        layout: ``"plain"`` returns (R, C//16) scales; ``"swizzled"`` returns the
            ``to_blocked`` byte sequence the kernels emit.
        fp8_max: block-scale ceiling. 448 for every plain NVFP4 cast; MS-EDEN passes
            256 so the corrected scale has headroom before it overflows E4M3.
    """
    if block not in ("1x16", "16x16"):
        raise ValueError(f"block must be '1x16' or '16x16', got {block!r}")
    if layout not in ("plain", "swizzled"):
        raise ValueError(f"layout must be 'plain' or 'swizzled', got {layout!r}")
    block_rows = 1 if block == "1x16" else 16

    xf = x.float()
    s_enc = global_encode_scale(global_amax, fp8_max)
    block_scale = _block_scale(_block_amax(xf, block_rows), s_enc, fp8_max)
    enc = _encode_scale(block_scale, s_enc)

    # Broadcast the per-tile scale back over its elements.
    expanded = enc.repeat_interleave(block_rows, dim=0).repeat_interleave(16, dim=1)
    scaled = xf * expanded

    scales_plain = block_scale.repeat_interleave(block_rows, dim=0)
    scales = to_blocked(scales_plain) if layout == "swizzled" else scales_plain
    return NVFP4ReferenceOutput(
        codes=pack_fp4(scaled),
        scales=scales,
        block_scale=block_scale,
        encode_scale=enc,
        values=xf,
        scaled=scaled,
        block_rows=block_rows,
    )


def reference_weight_quantize_2d(
    W: torch.Tensor, global_amax: torch.Tensor, *, layout: str = "swizzled"
) -> Tuple[NVFP4ReferenceOutput, NVFP4ReferenceOutput]:
    """``(rowwise, colwise)`` references for ``*_weight_quantize_2d`` (16x16, no RHT).

    Colwise is the same recipe applied to ``W.t()``; ``max|W.t()| == max|W|``, so both
    take the same global amax.
    """
    rowwise = nvfp4_reference_quantize(W, global_amax, block="16x16", layout=layout)
    colwise = nvfp4_reference_quantize(
        W.t().contiguous(), global_amax, block="16x16", layout=layout
    )
    return rowwise, colwise


# MS-EDEN's block-scale ceiling, giving decode numerator 256 * 6 = 1536.
EDEN_BLOCK_SCALE_MAX = 256.0

# E2M1 magnitudes at magnitude codes 0..7. Bit 3 of a nibble is the sign.
_FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def decode_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    """(R, C//2) packed uint8 -> (R, C) float32 grid values, even element in the low
    nibble.

    Exact by construction: every E2M1 value is representable in float32, so this is
    the inverse of ``pack_fp4`` with no rounding of its own.
    """
    lut = torch.tensor(_FP4_MAGNITUDES, dtype=torch.float32, device=codes.device)
    lo = codes & 0xF
    hi = codes >> 4
    nibbles = (
        torch.stack((lo, hi), dim=-1).reshape(codes.shape[0], codes.shape[1] * 2).long()
    )
    magnitude = lut[nibbles & 0x7]
    return torch.where(nibbles & 0x8 != 0, -magnitude, magnitude)


def pack_fp4_stochastic(
    scaled: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """``pack_fp4`` with per-element stochastic rounding: each element lands on the E2M1
    neighbour above its (clamped) magnitude with probability equal to its fractional
    position between the two neighbours, so ``E[decode] == clamp(scaled)`` element by
    element. The textbook unbiased NVFP4 quantizer, for comparison against MS-EDEN.
    """
    grid = torch.tensor(_FP4_MAGNITUDES, dtype=torch.float32, device=scaled.device)
    magnitude = scaled.abs().clamp(max=FP4_E2M1_MAX)
    upper = torch.bucketize(magnitude, grid, right=True).clamp(max=7)
    lo, hi = grid[upper - 1], grid[upper]
    u = torch.rand(magnitude.shape, generator=generator, device=scaled.device)
    up = u * (hi - lo) < magnitude - lo
    return pack_fp4(torch.where(up, hi, lo).copysign(scaled))


def reference_dequantize_rowwise(
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_amax: torch.Tensor,
    *,
    is_swizzled: bool = True,
    fp8_max: float = FP8_E4M3_MAX,
) -> torch.Tensor:
    """Reconstruct ``W_qdq`` in float32 from packed codes and E4M3 block scales.

    The exact inverse of ``nvfp4_reference_quantize``'s scale chain:
    ``value * f32(block_scale) * (1 / S_enc)``. Deliberately *not* routed through
    ``NVFP4Tensor.dequantize``, which applies an ``E4M3_EPS`` floor and is therefore
    only good to one fp8 ULP -- the requantization ops must be pinned bitwise.
    """
    rows = codes.shape[0]
    cols = codes.shape[1] * 2
    values = decode_fp4_codes(codes)
    plain = (from_blocked(scales, rows, cols // 16) if is_swizzled else scales).float()
    s_enc = global_encode_scale(global_amax, fp8_max)
    return (values * plain.repeat_interleave(16, dim=1)) * (1.0 / s_enc)


@dataclass(frozen=True)
class MSEdenReferenceOutput:
    """The deterministic part of MS-EDEN -- everything up to the stochastic draw.

    MS-EDEN's only random step is the E4M3 rounding of the corrected block scale, and
    stochastic rounding is unbiased for whatever it rounds. So the FP4 codes are
    seed-independent, and ``corrected_scale`` / ``ideal_dequant`` are the targets of
    ``E[.]`` over the RNG *exactly* rather than approximately: the codes carry no
    randomness, so they factor straight out of the expectation.

    Note what ``ideal_dequant`` is not. It is not ``x``. The Eden correction
    ``<v, v> / <v, q>`` rescales a block so its inner product with the input is
    preserved, which is a different objective from reproducing the input, so
    ``E[dequant]`` converges on the corrected reconstruction and stays a full FP4
    quantization error away from ``x``.
    """

    codes: torch.Tensor  # (R, C//2) uint8, RTNE, identical for every seed
    block_scale: torch.Tensor  # (R, C//16) float32, pre-correction E4M3 value
    corrected_scale: torch.Tensor  # (R, C//16) float32, E[sampled scale]
    ideal_dequant: torch.Tensor  # (R, C) float32, E[dequant]


def reference_ms_eden(
    x: torch.Tensor, global_amax: torch.Tensor
) -> MSEdenReferenceOutput:
    """MS-EDEN quantize a 2-D tensor, stopping short of the stochastic scale rounding.

    The E4M3 ceiling is ``EDEN_BLOCK_SCALE_MAX`` rather than 448, which is what makes
    the per-tensor numerator 1536. The per-block correction falls back to 1.0 where
    the ratio is undefined -- a block that packs to all-zero codes has
    ``<v, q> == 0`` -- matching the kernel's guard.

    The PyTorch ratio is correctly rounded; the software kernel retains PTX
    ``div.full`` (up to two ULP error). Byte equality is tested on the covered
    inputs, not asserted as a universal property of approximate division. The
    hardware-SR fast path is validated separately by its code and scale properties.
    """
    base = nvfp4_reference_quantize(
        x, global_amax, block="1x16", layout="plain", fp8_max=EDEN_BLOCK_SCALE_MAX
    )
    rows, cols = base.values.shape
    scaled = base.scaled.reshape(rows, cols // 16, 16)
    values = decode_fp4_codes(base.codes).reshape(rows, cols // 16, 16)

    ratio = _sum16_rn(scaled * scaled) / _sum16_rn(scaled * values)
    correction = torch.where(torch.isfinite(ratio), ratio, torch.ones_like(ratio))

    block_scale = base.block_scale.float()
    corrected_scale = block_scale * correction
    s_enc = global_encode_scale(global_amax, EDEN_BLOCK_SCALE_MAX)
    decode = corrected_scale * (1.0 / s_enc)
    return MSEdenReferenceOutput(
        codes=base.codes,
        block_scale=block_scale,
        corrected_scale=corrected_scale,
        ideal_dequant=(values * decode.unsqueeze(-1)).reshape(rows, cols),
    )


def reference_dynamic_rht(
    A: torch.Tensor, sign_vector: torch.Tensor, *, transpose: bool
) -> torch.Tensor:
    """``A @ R`` or ``A.t() @ R`` where ``R = diag(sign_vector) @ H / sqrt(n)``.

    ``n`` is read from ``sign_vector``'s length, so this serves both the RHT-16 and
    RHT-128 paths. The bf16 downcast on the way out matches the kernels, which round
    their fp32 Hadamard accumulator to bf16 before consuming it.
    """
    n = sign_vector.numel()
    R = get_dynamic_rht_matrix(sign_vector, torch.bfloat16)
    source = A.t().contiguous() if transpose else A
    rows, cols = source.shape
    return (
        (source.reshape(-1, n).to(torch.bfloat16) @ R)
        .reshape(rows, cols)
        .to(torch.bfloat16)
    )


def reference_row_cast_col_rht_amax(
    A: torch.Tensor, sign_vector: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """§11.8: ``(amax_rht_a_t, amax_a)`` -- transformed first, both scalar f32."""
    return (
        reference_dynamic_rht(A, sign_vector, transpose=True).float().abs().max(),
        A.float().abs().max(),
    )


def reference_row_cast_col_rht_quantize(
    A: torch.Tensor,
    row_global_amax: torch.Tensor,
    col_global_amax: torch.Tensor,
    sign_vector: torch.Tensor,
    *,
    layout: str = "swizzled",
) -> Tuple[NVFP4ReferenceOutput, NVFP4ReferenceOutput]:
    """§11.9: ``(row, col)`` references, matching the op's rowwise-first return."""
    row = nvfp4_reference_quantize(A, row_global_amax, block="1x16", layout=layout)
    col = nvfp4_reference_quantize(
        reference_dynamic_rht(A, sign_vector, transpose=True),
        col_global_amax,
        block="1x16",
        layout=layout,
    )
    return row, col


def reference_row_rht_col_rht_amax(
    dy: torch.Tensor, dgrad_rht: torch.Tensor, wgrad_rht: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """§11.2: ``(amax_rht_dy, amax_rht_dy_t)`` -- rowwise first, both scalar f32.

    ``dgrad_rht`` rotates the un-transposed tensor and ``wgrad_rht`` the transposed
    one. Swapping them changes both outputs, which is the discriminating test.
    """
    return (
        reference_dynamic_rht(dy, dgrad_rht, transpose=False).float().abs().max(),
        reference_dynamic_rht(dy, wgrad_rht, transpose=True).float().abs().max(),
    )


def reference_row_cast_quantize(
    W: torch.Tensor, global_amax: torch.Tensor, *, layout: str = "swizzled"
) -> NVFP4ReferenceOutput:
    """§11.1: rowwise 1x16 NVFP4, RTNE, no RHT, and **no columnwise output**.

    Contrast ``reference_weight_quantize_2d``, which is 16x16 and returns a pair.
    """
    return nvfp4_reference_quantize(W, global_amax, block="1x16", layout=layout)


def reference_col_cast_requant_amax(
    row_fp4_w: torch.Tensor, row_sf_w: torch.Tensor, global_amax: torch.Tensor
) -> torch.Tensor:
    """§11.6: ``amax(abs(W_qdq.bf16().t()))``, scalar f32.

    Equal to ``amax(abs(W_qdq))`` because a transpose does not change the element
    set, and generally *not* equal to ``global_amax``, which bounds the original
    weight rather than the quantized-dequantized one.
    """
    w_qdq = reference_dequantize_rowwise(row_fp4_w, row_sf_w, global_amax)
    w_qdq = torch.where(torch.isfinite(global_amax), w_qdq, torch.zeros_like(w_qdq))
    return w_qdq.to(torch.bfloat16).float().abs().max()


def reference_col_cast_requantize(
    row_fp4_w: torch.Tensor,
    row_sf_w: torch.Tensor,
    global_amax: torch.Tensor,
    amax_w_qdq_t: torch.Tensor,
    *,
    layout: str = "swizzled",
) -> NVFP4ReferenceOutput:
    """§11.7: rowwise 1x16 NVFP4 of ``W_qdq.bf16().t()``."""
    w_qdq = reference_dequantize_rowwise(row_fp4_w, row_sf_w, global_amax)
    w_qdq = torch.where(torch.isfinite(global_amax), w_qdq, torch.zeros_like(w_qdq))
    return nvfp4_reference_quantize(
        w_qdq.to(torch.bfloat16).t().contiguous(),
        amax_w_qdq_t,
        block="1x16",
        layout=layout,
    )


def reference_col_rht_requant_amax(
    row_fp4_w: torch.Tensor,
    row_sf_w: torch.Tensor,
    global_amax: torch.Tensor,
    dgrad_rht: torch.Tensor,
) -> torch.Tensor:
    """§11.4: ``amax(abs(W_qdq.bf16().t() @ R_n))``, scalar f32."""
    w_qdq = reference_dequantize_rowwise(row_fp4_w, row_sf_w, global_amax)
    w_qdq = torch.where(torch.isfinite(global_amax), w_qdq, torch.zeros_like(w_qdq))
    rotated = reference_dynamic_rht(w_qdq.to(torch.bfloat16), dgrad_rht, transpose=True)
    return rotated.float().abs().max()


def reference_col_rht_requantize(
    row_fp4_w: torch.Tensor,
    row_sf_w: torch.Tensor,
    global_amax: torch.Tensor,
    amax_rht_w_qdq_t: torch.Tensor,
    dgrad_rht: torch.Tensor,
    *,
    layout: str = "swizzled",
) -> NVFP4ReferenceOutput:
    """§11.5: rowwise 1x16 NVFP4 of ``W_qdq.bf16().t() @ R_n``."""
    w_qdq = reference_dequantize_rowwise(row_fp4_w, row_sf_w, global_amax)
    w_qdq = torch.where(torch.isfinite(global_amax), w_qdq, torch.zeros_like(w_qdq))
    rotated = reference_dynamic_rht(w_qdq.to(torch.bfloat16), dgrad_rht, transpose=True)
    return nvfp4_reference_quantize(
        rotated, amax_rht_w_qdq_t, block="1x16", layout=layout
    )


# ---------------------------------------------------------------------------
# Grouped V2 / V1_REQUANT oracles
#
# Every one of these is the corresponding linear oracle applied per group. That is
# not a shortcut -- it *is* the specification: a grouped kernel is correct exactly
# when each group matches the linear reference, and a kernel that is wrong only at
# num_tensors > 1 has a group-index bug rather than a numerics bug.
# ---------------------------------------------------------------------------


def reference_group_row_cast_quantize(
    W: torch.Tensor, global_amax: torch.Tensor, *, layout: str = "swizzled"
):
    """§11.1 per expert. ``W`` is ``(E, M, N)``; ``global_amax`` is ``(E,)``."""
    return [
        reference_row_cast_quantize(W[e], global_amax[e], layout=layout)
        for e in range(W.shape[0])
    ]


def reference_group_col_cast_requant_amax(
    row_fp4_w: torch.Tensor, row_sf_w: torch.Tensor, global_amax: torch.Tensor
) -> torch.Tensor:
    """§11.6 per expert -> ``(E,)`` float32."""
    return torch.stack(
        [
            reference_col_cast_requant_amax(row_fp4_w[e], row_sf_w[e], global_amax[e])
            for e in range(row_fp4_w.shape[0])
        ]
    )


def reference_group_col_cast_requantize(
    row_fp4_w: torch.Tensor,
    row_sf_w: torch.Tensor,
    global_amax: torch.Tensor,
    amax_w_qdq_t: torch.Tensor,
    *,
    layout: str = "swizzled",
):
    """§11.7 per expert."""
    return [
        reference_col_cast_requantize(
            row_fp4_w[e], row_sf_w[e], global_amax[e], amax_w_qdq_t[e], layout=layout
        )
        for e in range(row_fp4_w.shape[0])
    ]


def reference_group_col_rht_requant_amax(
    row_fp4_w: torch.Tensor,
    row_sf_w: torch.Tensor,
    global_amax: torch.Tensor,
    dgrad_rht: torch.Tensor,
) -> torch.Tensor:
    """§11.4 per expert -> ``(E,)`` float32. One ``dgrad_rht`` serves every expert."""
    return torch.stack(
        [
            reference_col_rht_requant_amax(
                row_fp4_w[e], row_sf_w[e], global_amax[e], dgrad_rht
            )
            for e in range(row_fp4_w.shape[0])
        ]
    )


def reference_group_col_rht_requantize(
    row_fp4_w: torch.Tensor,
    row_sf_w: torch.Tensor,
    global_amax: torch.Tensor,
    amax_rht_w_qdq_t: torch.Tensor,
    dgrad_rht: torch.Tensor,
    *,
    layout: str = "swizzled",
):
    """§11.5 per expert."""
    return [
        reference_col_rht_requantize(
            row_fp4_w[e],
            row_sf_w[e],
            global_amax[e],
            amax_rht_w_qdq_t[e],
            dgrad_rht,
            layout=layout,
        )
        for e in range(row_fp4_w.shape[0])
    ]


def reference_group_row_cast_col_rht_amax(
    A: torch.Tensor,
    sign_vector: torch.Tensor,
    offsets: torch.Tensor,
    num_tensors: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """§11.8 per group -> two ``(num_tensors,)`` float32 tensors, transformed first."""
    col, row, start = [], [], 0
    for size in _group_sizes(offsets, num_tensors):
        group = A[start : start + size]
        if size == 0:
            col.append(A.new_zeros((), dtype=torch.float32))
            row.append(A.new_zeros((), dtype=torch.float32))
            continue
        c, r = reference_row_cast_col_rht_amax(group, sign_vector)
        col.append(c)
        row.append(r)
        start += size
    return torch.stack(col), torch.stack(row)


def reference_group_row_rht_col_rht_amax(
    dy: torch.Tensor,
    dgrad_rht: torch.Tensor,
    wgrad_rht: torch.Tensor,
    offsets: torch.Tensor,
    num_tensors: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """§11.2 per group -> two ``(num_tensors,)`` float32 tensors, rowwise first."""
    rows, cols, start = [], [], 0
    for size in _group_sizes(offsets, num_tensors):
        group = dy[start : start + size]
        if size == 0:
            rows.append(dy.new_zeros((), dtype=torch.float32))
            cols.append(dy.new_zeros((), dtype=torch.float32))
            continue
        r, c = reference_row_rht_col_rht_amax(group, dgrad_rht, wgrad_rht)
        rows.append(r)
        cols.append(c)
        start += size
    return torch.stack(rows), torch.stack(cols)


def reference_group_row_cast_col_rht_quantize(
    A: torch.Tensor,
    row_global_amax: torch.Tensor,
    col_global_amax: torch.Tensor,
    sign_vector: torch.Tensor,
    offsets: torch.Tensor,
    num_tensors: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """§11.9 per group -> ``(row_codes, row_sf, col_codes, col_sf)``.

    Assembled into the kernel's whole-buffer shapes rather than returned per group,
    because the columnwise scale buffer cannot be checked group by group: it puts the
    grouped token axis on the inner, 64-blocked side, so its swizzle tiling restarts
    at every group boundary and only the assembled byte sequence is meaningful. That
    is the same reason ``reference_group_rht_quantize_row_col`` is shaped this way,
    and the layout this is guarding is the one a mispasted columnwise store gets wrong.
    """
    psl, hidden = A.shape
    sizes = _group_sizes(offsets, num_tensors)

    row_codes = A.new_zeros((psl, hidden // 2), dtype=torch.uint8)
    row_sf_plain = A.new_zeros((psl, hidden // 16), dtype=torch.float8_e4m3fn)
    col_codes = A.new_zeros((hidden, psl // 2), dtype=torch.uint8)
    col_sf_blocks = []

    start = 0
    for g, size in enumerate(sizes):
        end = start + size
        if size == 0:
            continue
        row, col = reference_row_cast_col_rht_quantize(
            A[start:end],
            row_global_amax[g],
            col_global_amax[g],
            sign_vector,
            layout="plain",
        )
        row_codes[start:end] = row.codes
        row_sf_plain[start:end] = row.scales
        col_codes[:, start // 2 : end // 2] = col.codes
        col_sf_blocks.append(col.scales)
        start += size

    row_sf = to_blocked(row_sf_plain).view(psl, hidden // 16)
    col_sf = A.new_zeros((hidden, psl // 16), dtype=torch.float8_e4m3fn)
    valid_scales = to_blocked_grouped(torch.cat(col_sf_blocks, dim=1), sizes)
    col_sf.flatten()[: valid_scales.numel()] = valid_scales.flatten()
    return row_codes, row_sf, col_codes, col_sf


def reference_group_row_rht_col_rht_quantize_ms_eden(
    dy: torch.Tensor,
    amax_rht_dy: torch.Tensor,
    amax_rht_dy_t: torch.Tensor,
    dgrad_rht: torch.Tensor,
    wgrad_rht: torch.Tensor,
    offsets: torch.Tensor,
    num_tensors: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """§11.3 per group -> ``(row_codes, row_scale, col_codes, col_scale)``.

    Rowwise first, matching the op. The codes are assembled into the kernel's
    whole-buffer shapes and are exact; the scales are the *unrounded corrected*
    fp32 scales in plain layout, since the kernel's bytes are one stochastic draw
    away and can only be bounded, not matched. Compare them after unswizzling --
    ``from_blocked`` for the rowwise buffer, ``from_blocked_grouped`` for the
    columnwise one, whose tiling restarts at each group boundary.
    """
    psl, hidden = dy.shape
    sizes = _group_sizes(offsets, num_tensors)

    row_codes = dy.new_zeros((psl, hidden // 2), dtype=torch.uint8)
    row_scale = dy.new_zeros((psl, hidden // 16), dtype=torch.float32)
    col_codes = dy.new_zeros((hidden, psl // 2), dtype=torch.uint8)
    col_scale = dy.new_zeros((hidden, psl // 16), dtype=torch.float32)

    start = 0
    for g, size in enumerate(sizes):
        end = start + size
        if size == 0:
            continue
        group = dy[start:end]
        row = reference_ms_eden(
            reference_dynamic_rht(group, dgrad_rht, transpose=False), amax_rht_dy[g]
        )
        col = reference_ms_eden(
            reference_dynamic_rht(group, wgrad_rht, transpose=True), amax_rht_dy_t[g]
        )
        row_codes[start:end] = row.codes
        row_scale[start:end] = row.corrected_scale
        col_codes[:, start // 2 : end // 2] = col.codes
        col_scale[:, start // 16 : end // 16] = col.corrected_scale
        start += size

    return row_codes, row_scale, col_codes, col_scale


def _sum16_rn(values: torch.Tensor) -> torch.Tensor:
    """Separate FP32 operations in the software MS-EDEN reduction order."""
    p = values.unbind(-1)
    even = ((p[0] + p[2]) + (p[4] + p[6])) + ((p[8] + p[10]) + (p[12] + p[14]))
    odd = ((p[1] + p[3]) + (p[5] + p[7])) + ((p[9] + p[11]) + (p[13] + p[15]))
    return even + odd


def philox4x32(
    seed: torch.Tensor, offset: torch.Tensor, index: torch.Tensor
) -> torch.Tensor:
    """Independent PyTorch Philox4x32-10, counter=(offset, index, 0, 0).

    The key/counter contract follows AO PR #4798's CuTeDSL Philox implementation.
    This oracle uses only integer tensor arithmetic, no kernel implementation or
    torch RNG state. int64 products retain the low 64 bits; masking the arithmetic
    right shift recovers the unsigned high word even when the product wraps.
    """
    mask = 0xFFFFFFFF
    c0 = torch.full_like(index, int(offset) & mask)
    c1 = index.to(torch.int64) & mask
    c2, c3 = torch.zeros_like(c1), torch.zeros_like(c1)
    key = int(seed)
    k0, k1 = key & mask, (key >> 32) & mask
    for _ in range(10):
        a, b = c0 * 0xD2511F53, c2 * 0xCD9E8D57
        c0, c1, c2, c3 = (
            ((b >> 32) & mask) ^ c1 ^ k0,
            b & mask,
            ((a >> 32) & mask) ^ c3 ^ k1,
            a & mask,
        )
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    return torch.stack((c0, c1, c2, c3), dim=-1)


def reference_software_e4m3_sr(corrected, seed, offset):
    """Round corrected scales using one Philox word per scale in plain layout."""
    index = torch.arange(
        corrected.numel(), device=corrected.device, dtype=torch.int64
    ).reshape(corrected.shape)
    words = philox4x32(seed, offset, index)[..., 0]
    saturated = torch.fmin(corrected, torch.full_like(corrected, 448.0))
    bits = (saturated * 2.0**-120).contiguous().view(torch.int32).to(
        torch.int64
    ) & 0xFFFFFFFF
    rounded = (bits + (words & 0xFFFFF)) >> 20
    byte = ((rounded & 0x7F) | ((rounded >> 4) & 0x80)).to(torch.uint8)
    return byte.view(torch.float8_e4m3fn)
