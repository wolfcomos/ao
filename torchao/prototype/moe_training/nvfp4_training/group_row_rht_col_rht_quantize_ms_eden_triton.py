# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped MS-EDEN quantize, RHT-128 on both axes.

The V2 backward gradient quantize, producing both operands from one read of ``dy``:

    row_fp4_rht_dy   = ms_eden(dy_g @ R_n)       # dgrad operand
    col_fp4_rht_dy_t = ms_eden(dy_g.t() @ R_m)   # wgrad operand

MS-EDEN replaces FP4 stochastic rounding with RTNE codes plus a corrected,
**stochastically-rounded E4M3 block scale** -- the scale is the only random step.
The block-scale ceiling is 256, not 448, leaving headroom for the correction. That
is why these operands decode with numerator ``256 * 6 = 1536`` while every cast
operand uses ``448 * 6 = 2688``. Pairing an MS-EDEN operand with the cast numerator
leaves the forward correct and fails backward by roughly 40%.

.. warning::

   **Return order is rowwise first**, matching every sibling grouped quantize op in
   this directory: ``(row_fp4_rht_dy, row_sf_rht_dy, col_fp4_rht_dy_t,
   col_sf_rht_dy_t)``, the same shape as ``triton_group_rht_quantize_row_col``.

A dense linear is the degenerate ``num_tensors = 1`` case: ``offsets = [M]``.
"""

from typing import Optional

import torch
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

from .group_hadamard_utils import _validate_graph_amax, _validate_rng_state

RHT_SIZE = 128

# MS-EDEN's block-scale ceiling. Lower than the 448.0 a plain NVFP4 cast uses, to
# leave headroom for the stochastically-rounded scale correction.
EDEN_BLOCK_SCALE_MAX = 256.0

if torch_version_at_least("2.10.0") and has_triton():
    from typing import Tuple

    import triton
    import triton.language as tl

    from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
        BLOCK_M,
        BLOCK_N,
        _get_group_idx_binary,
        _validate_grouped_hadamard_inputs,
    )
    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
        _nvfp4_quantize,
        _store_grouped_scales_swizzle,
        _store_scales_swizzle,
        _swizzle_scales,
        convert_4xfp4_packed_to_8xfp32,
        convert_8xfp32_to_4xfp4_packed,
        get_dynamic_rht_matrix,
    )

    # A jit'ed body cannot read a plain module-level float, so the ceiling crosses
    # into the kernel as a constexpr alias of the same constant.
    _EDEN_BLOCK_SCALE_MAX = tl.constexpr(EDEN_BLOCK_SCALE_MAX)

    # num_warps pinned at 4 for the same reason as the other grouped quantize
    # kernels: the body is register-heavy and the 8-warp win is M-dependent, while
    # M is dropped from the key so one config is cached across all M.
    _GROUP_MS_EDEN_CONFIGS: list[triton.Config] = [
        triton.Config({}, num_warps=4, num_stages=ns) for ns in (2, 3, 4)
    ]

    @triton.jit
    def _ms_eden_correction_with_sr(
        block_scale,
        scaled,
        packed_fp4,
        seed_ptr,
        offset_base_ptr,
        pid_inner,
        offs_outer,
        INNER,
        BLOCK_OUTER: tl.constexpr,
        BLOCK_INNER: tl.constexpr,
    ):
        """MS-EDEN's stochastic epilogue: correct the block scale, then round it.

        ``scaled`` is the pre-quantization fp32 tile (v) and ``packed_fp4`` the RTNE
        codes just written (q). The correction ``<v, v> / <v, q>`` rescales each
        16-element block so its inner product with the input is preserved -- a
        different objective from reproducing the input, which is why the expected
        dequant stays a full FP4 error away from it. It falls back to 1.0 where the
        ratio is undefined: a block that packs to all-zero codes has ``<v, q> == 0``.

        ``scaled`` must be the unclamped tile. When the block's E4M3 scale rounded down
        (half the blocks), its amax element sits in (6, 6.375] and saturates to code 6;
        the correction repays exactly that excess. Fed the clamped tile it sees ``v ==
        q == 6`` there, repays nothing, and the block comes out ~0.7 % small -- a
        one-sided bias the stochastic step cannot average away. ``reference_ms_eden``
        and the kernels this op was ported from take the unclamped values.

        The E4M3 rounding is MS-EDEN's only random step, and it happens here in fp32.
        The returned value already sits exactly on the E4M3 grid, so the implicit
        fp32 -> float8_e4m3fn conversion in the scale store is exact and cannot
        re-round the draw away.

        Indices are outer/inner rather than row/col because both operands call this:
        the rowwise tile is ``(BLOCK_M, BLOCK_N)`` over hidden extent ``N``, and the
        columnwise tile is ``(BLOCK_N, BLOCK_M)`` over packed-token extent ``M``.
        """
        dequant = convert_4xfp4_packed_to_8xfp32(packed_fp4)
        v = tl.reshape(scaled, [BLOCK_OUTER, BLOCK_INNER // 16, 16])
        q = tl.reshape(dequant, [BLOCK_OUTER, BLOCK_INNER // 16, 16])
        dot_sq = tl.sum(v * v, axis=-1)
        dot_cross = tl.sum(v * q, axis=-1)
        ratio = dot_sq / dot_cross
        is_finite = tl.abs(ratio) < float("inf")
        correction = tl.where((dot_cross != 0.0) & is_finite, ratio, 1.0)
        # Corrects the already-narrowed E4M3 block scale, not the pre-narrowing
        # pvscale, matching the reference. Triton widens the fp8 operand here.
        corrected = block_scale * correction

        # Apply stochastic rounding to the corrected block scale.
        seed = tl.load(seed_ptr)
        offset_base = tl.load(offset_base_ptr).to(tl.uint64) & 0xFFFFFFFF
        # Same int64 reasoning as the tile indices: this counter is
        # outer_extent * inner_extent // 16, which passes 2**31 on large token counts.
        BLOCK_INNER_SF: tl.constexpr = BLOCK_INNER // 16
        INNER_SF = INNER // 16
        outer_idx = offs_outer[:, None].to(tl.int64)
        inner_sf = pid_inner * BLOCK_INNER_SF + tl.arange(0, BLOCK_INNER_SF)
        linear_idx = outer_idx * INNER_SF + inner_sf[None, :]
        offset = (linear_idx.to(tl.uint64) << 32) | offset_base
        rbits = tl.randint(seed, offset).to(tl.uint32)

        # Truncate the fp32 mantissa to E4M3's 3 bits after adding a random addend
        # over the 20 discarded bits. The 2**-120 / 2**120 round trip keeps E4M3
        # subnormals clear of fp32's own subnormal range, where the shift would stop
        # being a truncation.
        r = corrected * (2.0**-120)
        r_int = r.to(tl.uint32, bitcast=True)
        u = r_int + (rbits & ((1 << 20) - 1))
        u = (u >> 20) << 20
        return u.to(tl.float32, bitcast=True) * (2.0**120)

    @triton.autotune(configs=_GROUP_MS_EDEN_CONFIGS, key=["N"])
    @triton.jit
    def _group_row_rht_col_rht_quantize_ms_eden_kernel(
        a_ptr,
        row_rht_ptr,
        col_rht_ptr,
        qa_ptr,
        sfa_ptr,
        qa_t_ptr,
        sfa_t_ptr,
        offsets_ptr,
        amax_row_rht_ptr,
        amax_col_rht_ptr,
        col_seed_base_ptr,
        col_offset_base_ptr,
        row_seed_base_ptr,
        row_offset_base_ptr,
        M,
        N,
        num_tensors: tl.constexpr,
        SHAPE_REP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        RHT_SIZE: tl.constexpr,
        logical_packed_length_ptr,
    ):
        """Grouped MS-EDEN quantization of ``dy @ R_n`` and ``dy.t() @ R_m``.

        Flat grid ``(cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N),)``. Both operands come out
        of one load of the ``(BLOCK_M, BLOCK_N)`` tile: the columnwise half rotates
        ``trans(a)`` by ``R_m`` and the rowwise half rotates ``a`` by ``R_n``, each
        against its own per-group amax and its own Philox stream.

        MS-EDEN specifics the body honors:

        * The global scales come from ``_nvfp4_quantize(..., EDEN_BLOCK_SCALE_MAX)``,
          not 448. That is what makes these operands' decode numerator 1536.
        * Codes are RTNE; randomness enters only through the E4M3 block scale, and
          only in ``_ms_eden_correction_with_sr``.
        * The correction reads back the codes it just packed, so each pack must
          precede its own scale epilogue -- it measures that exact rounding.

        The columnwise scale store is the one place grouping changes the layout: each
        group owns a separately swizzled ``(hidden, group_tokens//16)`` block and the
        allocation is their flat concatenation. **Compute that per-group word offset
        in 64-bit** -- at ``hidden = 7168`` a 32-bit product wraps past roughly 300k
        rows. ``_store_grouped_scales_swizzle`` already handles this.

        Rows at or beyond ``logical_packed_length`` are left as allocated.
        """
        # The RHT runs along the inner axis of each reshape below, so a
        # (.., RHT_SIZE) chunk must stay inside one row of the reshaped operand.
        # Columnwise that row is one hidden column's run of BLOCK_M tokens;
        # rowwise it is one token's run of BLOCK_N hidden elements. A chunk that
        # straddles two of them raises nothing and rotates the wrong axis.
        tl.static_assert(
            BLOCK_M % RHT_SIZE == 0, "columnwise RHT requires BLOCK_M % RHT_SIZE == 0"
        )
        tl.static_assert(
            BLOCK_N % RHT_SIZE == 0, "rowwise RHT requires BLOCK_N % RHT_SIZE == 0"
        )
        VARYING_FIRST_DIM: tl.constexpr = 1
        FP4_E2M1_MAX: tl.constexpr = 6.0

        tile_idx = tl.program_id(0)
        num_tiles_token = tl.cdiv(M, BLOCK_M)
        num_tiles_hidden = tl.cdiv(N, BLOCK_N)
        # int64 tile indices: the A load (offs_m * N) and the packed-store offsets
        # (offs_m * N//2, offs_n * M//2) overflow int32 once the packed token count
        # times N exceeds 2**31, silently wrapping to bad addresses.
        pid_m = (tile_idx // num_tiles_hidden).to(tl.int64)
        pid_n = tile_idx - pid_m * num_tiles_hidden
        token_offset = pid_m * BLOCK_M
        logical_packed_length = tl.load(logical_packed_length_ptr)

        if token_offset < logical_packed_length:
            if SHAPE_REP == VARYING_FIRST_DIM:
                group_idx = _get_group_idx_binary(
                    token_offset,
                    offsets_ptr,
                    num_tensors,
                )
            else:
                group_idx = pid_m // (num_tiles_token // num_tensors)

            offs_m = token_offset + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            a = tl.load(a_ptr + offs_m[:, None] * N + offs_n[None, :])

            rht_offsets = (
                tl.arange(0, RHT_SIZE)[:, None] * RHT_SIZE
                + tl.arange(0, RHT_SIZE)[None, :]
            )
            row_rht = tl.load(row_rht_ptr + rht_offsets)
            col_rht = tl.load(col_rht_ptr + rht_offsets)

            # --- columnwise / wgrad: ms_eden(dy.t() @ R_m) ---
            col_global_amax = tl.load(amax_col_rht_ptr + group_idx)
            a_t = tl.trans(a)
            a_t_reshape = tl.reshape(a_t, [BLOCK_N * BLOCK_M // RHT_SIZE, RHT_SIZE])
            # No fast-math variant: the MS-EDEN reference rounds the RHT accumulator
            # through bfloat16, and this op exposes no switch to skip it.
            a_t_rht = tl.dot(a_t_reshape, col_rht).to(tl.bfloat16)

            # Unclamped for the correction (``_ms_eden_correction_with_sr``); the packer
            # gets a clamped copy.
            col_scale, col_scaled = _nvfp4_quantize(
                a_t_rht,
                col_global_amax,
                BLOCK_N,
                BLOCK_M,
                False,  # FAST_MATH
                _EDEN_BLOCK_SCALE_MAX,
                CLAMP=False,
            )
            col_pairs = (
                tl.clamp(col_scaled, -FP4_E2M1_MAX, FP4_E2M1_MAX)
                .reshape(BLOCK_N, BLOCK_M // 2, 2)
                .split()
            )
            col_fp4 = convert_8xfp32_to_4xfp4_packed(col_pairs)
            packed_inner_t = pid_m * (BLOCK_M // 2) + tl.arange(0, BLOCK_M // 2)
            packed_offsets_t = (
                offs_n[:, None].to(tl.int64) * (M // 2) + packed_inner_t[None, :]
            )
            tl.store(qa_t_ptr + packed_offsets_t, col_fp4)

            col_sf = _ms_eden_correction_with_sr(
                col_scale,
                col_scaled,
                col_fp4,
                col_seed_base_ptr,
                col_offset_base_ptr,
                pid_m,
                offs_n,
                M,
                BLOCK_N,
                BLOCK_M,
            )
            col_swizzled = _swizzle_scales(col_sf, BLOCK_N, BLOCK_M)
            # Columnwise puts the grouped token axis on the inner (64-blocked) side,
            # so the tiling restarts per group. The rowwise store below has it on
            # the outer axis, where a group is already contiguous.
            _store_grouped_scales_swizzle(
                col_swizzled,
                sfa_t_ptr,
                pid_n,
                pid_m,
                offsets_ptr,
                group_idx,
                N,
                BLOCK_N,
                BLOCK_M,
            )

            # --- rowwise / dgrad: ms_eden(dy @ R_n) ---
            row_global_amax = tl.load(amax_row_rht_ptr + group_idx)
            a_reshape = tl.reshape(a, [BLOCK_M * BLOCK_N // RHT_SIZE, RHT_SIZE])
            a_rht = tl.dot(a_reshape, row_rht).to(tl.bfloat16)

            row_scale, row_scaled = _nvfp4_quantize(
                a_rht,
                row_global_amax,
                BLOCK_M,
                BLOCK_N,
                False,  # FAST_MATH
                _EDEN_BLOCK_SCALE_MAX,
                CLAMP=False,
            )
            row_pairs = (
                tl.clamp(row_scaled, -FP4_E2M1_MAX, FP4_E2M1_MAX)
                .reshape(BLOCK_M, BLOCK_N // 2, 2)
                .split()
            )
            row_fp4 = convert_8xfp32_to_4xfp4_packed(row_pairs)
            packed_inner = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
            packed_offsets = offs_m[:, None] * (N // 2) + packed_inner[None, :]
            tl.store(qa_ptr + packed_offsets, row_fp4)

            row_sf = _ms_eden_correction_with_sr(
                row_scale,
                row_scaled,
                row_fp4,
                row_seed_base_ptr,
                row_offset_base_ptr,
                pid_n,
                offs_m,
                N,
                BLOCK_M,
                BLOCK_N,
            )
            row_swizzled = _swizzle_scales(row_sf, BLOCK_M, BLOCK_N)
            _store_scales_swizzle(
                row_swizzled,
                sfa_ptr,
                pid_m,
                pid_n,
                M,
                N,
                BLOCK_M,
                BLOCK_N,
            )

    @torch.library.custom_op(
        "torchao::triton_group_row_rht_col_rht_quantize_ms_eden", mutates_args=()
    )
    def triton_group_row_rht_col_rht_quantize_ms_eden(
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-group MS-EDEN quantization of the rotated gradient and its transpose.

        Args:
            dy: packed ``(sum_M, N)`` bfloat16 gradient, row-major.
            amax_rht_dy: ``(num_tensors,)`` float32, first return of
                ``triton_group_row_rht_col_rht_amax``.
            amax_rht_dy_t: ``(num_tensors,)`` float32, second return of the same op.
            dgrad_rht: ``(128,)`` {-1, +1} device tensor rotating the row axis.
            wgrad_rht: ``(128,)`` {-1, +1} device tensor rotating the transposed axis.
            offsets: int32 cumulative row-end offsets, one per group.
            num_tensors: number of groups.
            packed_sequence_length: allocated row capacity of dy.
            hidden_size: number of columns in dy.
            shape_rep: SAME_BOTH_DIMS or VARYING_FIRST_DIM.
            rng_state: int64 CUDA tensor ``[col_seed, col_offset, row_seed,
                row_offset]``. Always required -- MS-EDEN is never deterministic, so
                unlike the cast quantizers there is no RTNE-only mode to fall back to.
            logical_packed_length: one-element int32 CUDA tensor, ``offsets[-1]``.

        Returns:
            ``(row_fp4_rht_dy, row_sf_rht_dy, col_fp4_rht_dy_t, col_sf_rht_dy_t)`` --
            **rowwise first**, like every sibling grouped op; see the module
            docstring warning for why this now diverges from design doc §6/§11.3.
              - ``(sum_M, N//2)`` uint8 rowwise codes.
              - ``(sum_M, N//16)`` float8_e4m3fn rowwise scales.
              - ``(N, sum_M//2)`` uint8 columnwise transposed codes.
              - ``(N, sum_M//16)`` float8_e4m3fn columnwise scales.

            Decode both with ``EDEN_NUMERATOR`` (1536), never 2688.
        """
        for name, sv in (("dgrad_rht", dgrad_rht), ("wgrad_rht", wgrad_rht)):
            if sv.ndim != 1 or sv.numel() != RHT_SIZE:
                raise ValueError(
                    f"{name} must be a ({RHT_SIZE},) tensor, got shape {tuple(sv.shape)}"
                )
            if not sv.is_cuda or sv.device != dy.device:
                raise ValueError(f"{name} must be on the same device as dy")
        row_rht = get_dynamic_rht_matrix(dgrad_rht, torch.bfloat16)
        col_rht = get_dynamic_rht_matrix(wgrad_rht, torch.bfloat16)
        _validate_grouped_hadamard_inputs(
            dy,
            row_rht,
            offsets,
            num_tensors,
            packed_sequence_length,
            hidden_size,
            shape_rep,
            logical_packed_length,
            rht_size=RHT_SIZE,
        )
        row_amax = _validate_graph_amax(
            amax_rht_dy, "amax_rht_dy", num_tensors, dy.device
        )
        col_amax = _validate_graph_amax(
            amax_rht_dy_t, "amax_rht_dy_t", num_tensors, dy.device
        )
        # MS-EDEN always draws, so SR is unconditionally on for validation purposes.
        rng_state = _validate_rng_state(rng_state, dy.device, True)

        qa_base = torch.empty(
            (packed_sequence_length, hidden_size // 2),
            dtype=torch.uint8,
            device=dy.device,
        )
        sfa_storage = torch.empty(
            (packed_sequence_length // 128, hidden_size // 64, 32, 16),
            dtype=torch.float8_e4m3fn,
            device=dy.device,
        )
        sfa_return = sfa_storage.view(packed_sequence_length, hidden_size // 16)

        qd = torch.empty(
            (hidden_size, packed_sequence_length // 2),
            dtype=torch.uint8,
            device=dy.device,
        )
        sfd_storage = torch.empty(
            (hidden_size // 128, packed_sequence_length // 64, 32, 16),
            dtype=torch.float8_e4m3fn,
            device=dy.device,
        )
        sfd_return = sfd_storage.view(hidden_size, packed_sequence_length // 16)

        m, n = dy.shape
        if logical_packed_length is None:
            logical_packed_length = offsets[-1:]
        grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
        _group_row_rht_col_rht_quantize_ms_eden_kernel[grid](
            dy,
            row_rht,
            col_rht,
            qa_base,
            sfa_storage,
            qd,
            sfd_storage,
            offsets,
            row_amax,
            col_amax,
            rng_state[0:1],
            rng_state[1:2],
            rng_state[2:3],
            rng_state[3:4],
            m,
            n,
            num_tensors=num_tensors,
            SHAPE_REP=shape_rep,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            RHT_SIZE=RHT_SIZE,
            logical_packed_length_ptr=logical_packed_length,
        )
        # Rowwise pair first, matching every sibling quantize op.
        return qa_base, sfa_return, qd, sfd_return

    @triton_group_row_rht_col_rht_quantize_ms_eden.register_fake
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

else:

    def triton_group_row_rht_col_rht_quantize_ms_eden(
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
    ):
        raise NotImplementedError(
            "triton_group_row_rht_col_rht_quantize_ms_eden requires torch 2.10.0+ "
            "and triton installed"
        )
