# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CuteDSL grouped RHT + NVFP4 kernels for SM100, ported from TransformerEngine.

Structural port of ``nvte_group_hadamard_transform_cast_fusion_graph_safe``
(TE ``graph_safe_group_row_cast_col_hadamard_transform_cast_fusion.cu``): CLC
dynamic persistent scheduler, 16-warp specialization, and a 128-token epilogue
tile that is group-aligned by construction (group offsets are 128-aligned, so a
tile never straddles two experts). Numeric primitives are reused from
``_cutedsl_kernels_impl`` rather than ported from TE, so the outputs match
torchao's existing NVFP4 oracle, not TE bit-for-bit.

Axis naming follows torch, not TE: ``A`` is ``(tokens, hidden)`` row-major, so
``A.t()`` is TE's column-major ``(M=hidden, N=packed_tokens)`` with no copy. The
UMMA contracts 16 consecutive *tokens* against the 16x16 Hadamard.

Two kernels share that mainloop and differ only in their epilogues:
  - ``_Tcgen05GroupRowColFused``: quantizes col=RHT(A.t()) and row=A to NVFP4.
  - ``_Tcgen05GroupRhtAmax``: reduces col=max|RHT(A.t())|, row=max|A|, per group.

Columnwise scale factors are per-group swizzled ``[hidden, tokens_g]`` blocks
concatenated in one flat allocation. The group-local 64-token tile axis must
therefore restart at every group boundary; treating the allocation as one
globally swizzled ``[hidden, packed_tokens]`` buffer gives later groups the
wrong layout.

A third kernel, ``_Tcgen05GroupRowRhtColRhtAmax``, serves the V2 backward gradient
amax: both axes are transformed by an RHT-128 with independent sign vectors. It is
standalone (no CLC, a static persistent grid) and feeds two UMMA chains from one
TMA'd 128x128 tile against two resident ``R^T`` operands its epilogue warps write
into shared memory from the live sign vectors; its ``RHT128_`` constants are its
own, the unprefixed ones above belong to the RHT-16 kernels.

A fourth kernel, ``_Tcgen05GroupRowRhtColRhtQuantizeMsEden``, is the MS-EDEN quantize
that consumes that amax kernel's two outputs: the same standalone two-chain mainloop,
each accumulator quantized from TMEM to RTNE FP4 codes plus a corrected,
stochastically rounded E4M3 block scale drawn from Triton's Philox stream. Its 256
ceiling is ``EDEN_BLOCK_SCALE_MAX``, imported from the MS-EDEN Triton module (a
module-level constant defined ahead of that module's Triton guard, so the import is
Triton-free). Its ``FAST_PATH`` variant (``fast_path=True`` on the op) draws one Philox
counter per 16 scales of a row and rounds the four corrected scales a warp holds with
one hardware ``cvt.rs.satfinite.e4m3x4.f32``: a different stochastic stream (16 random
bits per scale, one draw per 16 scales), not the Triton stream.

Two more kernels, ``_Tcgen05GroupColRhtRequantAmax`` and ``_Tcgen05GroupColRhtRequantize``,
serve the V2 backward weight path: the dequantized weight tile is rebuilt on chip from
the rowwise NVFP4 codes and swizzled scales by eight producer warps, rotated by the
resident constant ``H128`` through one UMMA chain with the sign vector folded into the
per-row decode scale, and either amax-reduced per expert or requantized to columnwise
NVFP4. They take an expert-uniform ``(E, M, N)`` stack (expert = tile-derived, no
offsets, no group cap) and share the producer as the Triton file shares
``_load_rht_requant_weight_tile``.

Two more, ``_Tcgen05GroupRowCastColRhtAmax`` and ``_Tcgen05GroupRowCastColRhtQuantize``,
serve the V2 forward activation path at ``dynamic_rht=True``: the col chain of the
gradient amax kernel against one resident ``R^T`` tile, with the RHT-16 kernels' SMEM row
epilogues reading the same TMA'd stage, so the raw rowwise and the RHT-128 columnwise
amax / NVFP4 outputs come from one pass over the packed activation.
"""

import functools
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream, make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import blackwell_helpers as sm100_utils
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

from ._cutedsl_kernels_impl import (
    DEFAULT_SIGN_VECTOR,
    FP4_E2M1_MAX,
    FP8_E4M3_MAX,
    FP32_MAX,
    HADAMARD_DIM,
    TILE_BLOCKS,
    _abs_amax16,
    _abs_f32,
    _atom_max_f32_nonneg,
    _bf16hi_to_f32,
    _bf16lo_to_f32,
    _bf16round_f32x8,
    _cvt_e2m1x8_to_f32,
    _cvt_rn_e2m1x8_f32,
    _cvt_rs_e4m3x4_f32,
    _div_full_f32,
    _div_rn_f32,
    _e4m3x2_to_f16x2,
    _f16hi_to_f32,
    _f16lo_to_f32,
    _get_num_sms,
    _get_rht_buffer,
    _get_sr_rng_buffer,
    _max_f32,
    _min_f32,
    _mul_clamp_f32x8,
    _pack16_rn_from_enc,
    _quant16,
    _rcp_rn_f32,
    _round_rht_amax,
    _sr_e4m3_byte,
    _st_global_v4_u32,
    philox4_all,
    philox_prep,
    philox_word0,
)
from .group_row_rht_col_rht_quantize_ms_eden_triton import EDEN_BLOCK_SCALE_MAX

# --- tile shapes (TE :262-271). M = hidden, N = tokens, K = 16 (the RHT block) ---
M_TILE = 128  # hidden rows per tile
N_TILE = 16  # UMMA N = one RHT block
K = HADAMARD_DIM  # UMMA K = 16 contracted tokens
EPI_UNROLL = 8  # 128 tokens / 16 -> UMMAs per accumulator stage
TOKEN_TILE = N_TILE * EPI_UNROLL  # 128 tokens per mainloop/epilogue tile
MMA_TILER = (M_TILE, N_TILE, K)
K_TILE_MAX = 8  # token tiles per scheduler work item (TE :271)

ACC_STAGES = 4  # 512 TMEM columns / (8 * 16)
CLC_STAGES = 1
CLC_RESPONSE_I32 = 4  # 16-byte cluster-launch-control response

# Mainloop stage count from the SM100 shared-memory budget (TE :1257-1264). The
# A tile dominates; the reserve covers sB, every pipeline's mbarriers, the CLC
# response, and the TMEM holding buffer. The epilogues reduce in registers and
# atomic straight to global, so there is no per-group amax staging in SMEM.
_SMEM_CAPACITY = 232448
_A_TILE_BYTES = M_TILE * TOKEN_TILE * 2
_SMEM_RESERVE = 2048
MAINLOOP_STAGES = (_SMEM_CAPACITY - _SMEM_RESERVE) // _A_TILE_BYTES

# --- warp specialization (TE :395-416). 16 warps / 512 threads ---
MMA_WARP = 0
TMA_WARP = 1
SCHED_WARP = 2
IDLE_WARP = 3
COL_WARP_BEGIN = 4
COL_WARP_END = 8
ROW_WARP_BEGIN = 8
ROW_WARP_END = 16
N_WARPS = 16
TPB = 32 * N_WARPS
COL_THREADS = 32 * (COL_WARP_END - COL_WARP_BEGIN)  # 128
ROW_THREADS = 32 * (ROW_WARP_END - ROW_WARP_BEGIN)  # 256

# warpgroup_reg_alloc is warpgroup-granular, so all four warps of WG0 (incl. the
# idle warp) must agree on the dealloc. 128*32 + 128*192 + 256*136 = 63488 <= 65536.
REG_DEALLOC = 32
REG_COL = 192
REG_ROW = 136

TMEM_ALLOC_BAR = 1
TMEM_DEALLOC_BAR = 2

# Row epilogue thread map: 256 threads cover a (128 hidden, 128 token) tile as
# 16 hidden x 1 token per thread per pass, 4 passes.
ROW_HB = M_TILE // 16  # 8 hidden blocks of 16
ROW_TOK_PER_PASS = ROW_THREADS // ROW_HB  # 32 tokens per pass
ROW_PASSES = TOKEN_TILE // ROW_TOK_PER_PASS  # 4


# Group lookup is a binary search unrolled to a constexpr depth -- chosen to keep the
# epilogues branch-free -- so it resolves exactly 2**GROUP_SEARCH_STEPS groups and the
# group count is capped there. Raising the cap means raising the depth with it: at
# E > 2**GROUP_SEARCH_STEPS the search exits with hi - lo > 1 and returns a group index
# off by one, which is a silently wrong amax rather than a failure. Keep the two in
# sync; ``test_cutedsl_group_rht_amax_rejects_too_many_groups`` pins the boundary.
#
# The 64 is ours, not inherited: TE's grouped NVFP4 kernels also cap at 64, but for
# unrelated reasons -- the pointer-list kernels (kMaxTensorsPerKernel) are bounded by
# the 4 KB kernel-argument limit, and the graph-safe ones by a shared-memory scratch
# array. TE's graph-safe kernels, which are the ones this design mirrors (packed input
# plus device-side offsets, no pointer arrays), use an unbounded `while` search and have
# no depth limit at all. Nothing here forces 64; it is comfortably above the local
# expert counts these models train at (671B at EP=64 gives 4).
MAX_GROUPS = 64
GROUP_SEARCH_STEPS = 6
assert MAX_GROUPS <= 2**GROUP_SEARCH_STEPS, (
    f"MAX_GROUPS={MAX_GROUPS} exceeds what a depth-{GROUP_SEARCH_STEPS} search resolves "
    f"({2**GROUP_SEARCH_STEPS})"
)


def _group_idx(token, offsets_t, num_groups):
    """Group containing ``token``, from cumulative row-end offsets.

    Branch-free port of triton ``_get_group_idx_binary``: the halvings are
    unrolled to a constexpr depth, so there is no dynamic control flow inside
    the epilogues. Offsets alone determine membership, which is correct for
    both SAME_BOTH_DIMS and VARYING_FIRST_DIM. The result is CTA-uniform
    because a 128-token tile never straddles a 128-aligned group boundary.
    """
    lo = cutlass.Int32(0)
    hi = num_groups
    for _ in range(GROUP_SEARCH_STEPS):
        mid = lo + (hi - lo) // cutlass.Int32(2)
        probe = cutlass.select_(mid > cutlass.Int32(0), mid - cutlass.Int32(1), 0)
        ge = token >= offsets_t[probe]
        active = (hi - lo) > cutlass.Int32(1)
        lo = cutlass.select_(active, cutlass.select_(ge, mid, lo), lo)
        hi = cutlass.select_(active, cutlass.select_(ge, hi, mid), hi)
    return lo


@cute.jit
def _flush_group_max(run_max, amax_t, g, lane):
    """Reduce a warp's running max and commit it to ``amax_t[g]``.

    The epilogues carry ``run_max`` across every tile that shares a group, so
    this runs once per group per work item rather than once per tile. A zero
    flush -- an empty work item, or one whose first tile already crossed -- is a
    no-op against the pre-zeroed buffer.

    Needs ``@cute.jit``: the lane predicate is a dynamic branch, which only
    lowers inside an AST-preprocessed function.
    """
    for offset in range(5):
        run_max = _max_f32(
            run_max, cute.arch.shuffle_sync_bfly(run_max, 1 << (4 - offset))
        )
    if lane == cutlass.Int32(0):
        _atom_max_f32_nonneg(amax_t.iterator + g, run_max)


def _group_at_work_item(tile_n_base, offsets_t, num_groups):
    """Group of a work item's first tile, with that group's end offset.

    A work item is K_TILE_MAX consecutive token tiles, so one search covers all
    of them unless a tile crosses out of the returned group; the epilogues
    re-search on that crossing rather than stepping ``g``, which keeps the
    result identical to a per-tile lookup even when a group is empty. Both
    searches go through ``_group_idx``, so both inherit its ``MAX_GROUPS`` /
    ``GROUP_SEARCH_STEPS`` depth cap -- raising one without the other returns a
    silently wrong group index rather than failing.
    """
    g = _group_idx(tile_n_base * cutlass.Int32(TOKEN_TILE), offsets_t, num_groups)
    return g, offsets_t[g]


def _global_scale(amax, fp8_max=FP8_E4M3_MAX):
    """NVFP4 two-level scale scalars from a global amax (TE :779-785).

    Returns ``(encode, decode, encode / fp4_max)``; a zero amax yields identity
    scales so the block scales stay finite.
    """
    is_zero = amax == cutlass.Float32(0.0)
    safe = cutlass.Float32(cutlass.select_(is_zero, cutlass.Float32(1.0), amax))
    c = _min_f32(
        _div_rn_f32(cutlass.Float32(fp8_max * FP4_E2M1_MAX), safe),
        cutlass.Float32(FP32_MAX),
    )
    c = cutlass.Float32(
        cutlass.select_(c == cutlass.Float32(0.0), cutlass.Float32(1.0), c)
    )
    enc = cutlass.Float32(cutlass.select_(is_zero, cutlass.Float32(1.0), c))
    dec = _div_rn_f32(cutlass.Float32(1.0), enc)
    return enc, dec, enc * cutlass.Float32(1.0 / FP4_E2M1_MAX)


class _GroupRhtMainloop:
    """Shared mainloop plumbing for the two grouped RHT kernels.

    Both kernels stream the same A tiles through the same UMMA and scheduler and
    differ only in their epilogues, so the MMA/TMA/scheduler setup is built once
    here and the kernel bodies stay independent.
    """

    def _setup(self, mA: cute.Tensor, mB: cute.Tensor, hidden, tokens):
        mma_op = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            MMA_TILER,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.MN,  # A: contiguous along hidden
            OperandMajorMode.K,  # B: H^T stored (N=16, K=16) row-major
        )
        tiled_mma = cute.make_tiled_mma(cute.make_mma_atom(mma_op))

        # A tile: 128 hidden x 128 tokens -> 8 k-blocks of 16 tokens.
        a_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.MN_SW128, cutlass.BFloat16
        )
        a_shape = tiled_mma.partition_shape_A(
            cute.dice((M_TILE, N_TILE, TOKEN_TILE), (1, None, 1))
        )
        a_smem_layout_staged = tcgen05.tile_to_mma_shape(
            a_atom, cute.append(a_shape, MAINLOOP_STAGES), order=(1, 2, 3)
        )
        # Same bytes, plain (hidden, token, stage) grouping for the row warps.
        # This is the DSL equivalent of TE's as_position_independent_swizzle_tensor.
        a_clean_layout = cute.tile_to_shape(
            a_atom, (M_TILE, TOKEN_TILE, MAINLOOP_STAGES), order=(0, 1, 2)
        )

        b_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW32, cutlass.BFloat16
        )
        b_shape = tiled_mma.partition_shape_B(cute.dice(MMA_TILER, (None, 1, 1)))
        b_smem_layout_staged = tcgen05.tile_to_mma_shape(
            b_atom, cute.append(b_shape, 1), order=(1, 2, 3)
        )

        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            g2s,
            mA,
            cute.slice_(a_smem_layout_staged, (None, None, None, 0)),
            (M_TILE, N_TILE, TOKEN_TILE),
            tiled_mma,
            (1, 1, 1, 1),
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            g2s,
            mB,
            cute.slice_(b_smem_layout_staged, (None, None, None, 0)),
            MMA_TILER,
            tiled_mma,
            (1, 1, 1, 1),
        )

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )

        # TMEM accumulator: 4 stages x 8 sub-tiles x 16 columns = 512 columns.
        acc_shape = tiled_mma.partition_shape_C(MMA_TILER[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(cute.append(acc_shape, EPI_UNROLL), ACC_STAGES)
        )
        num_tmem_alloc_cols = sm100_utils.get_num_tmem_alloc_cols(tCtAcc_fake)

        # The Hadamard rides its own one-shot barrier (TE :546-553), so the
        # per-tile mainloop transaction covers the A tile only.
        num_tma_load_bytes = M_TILE * TOKEN_TILE * 2
        num_b_load_bytes = N_TILE * K * 2

        tiles_in_m = hidden // cutlass.Int32(M_TILE)
        tiles_in_n = tokens // cutlass.Int32(TOKEN_TILE)
        # One work item = up to K_TILE_MAX consecutive token tiles at a fixed
        # hidden slab, reproducing TE's tile_n_base = q * K_TILE_MAX (TE :301-306).
        tiles_in_n_outer = (
            tiles_in_n + cutlass.Int32(K_TILE_MAX - 1)
        ) // cutlass.Int32(K_TILE_MAX)
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            (tiles_in_m, tiles_in_n_outer, cutlass.Int32(1)),
            (1, 1, 1),
        )
        grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)

        return (
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            a_clean_layout,
            b_smem_layout_staged,
            tCtAcc_fake.layout,
            num_tmem_alloc_cols,
            num_tma_load_bytes,
            num_b_load_bytes,
            tiles_in_n,
            tile_sched_params,
            grid,
        )


class _Tcgen05GroupRowColFused(_GroupRhtMainloop):
    """Fused grouped RHT columnwise + raw rowwise NVFP4 quantization.

    One TMA-loaded A tile feeds two consumers: the UMMA (which applies the 16x16
    RHT along the token axis, accumulating in TMEM for the columnwise path) and
    the row warp group (which reads the same SMEM bytes for the rowwise path).
    """

    def __init__(self, sr: bool = False, fast_math: bool = False):
        self.sr = sr
        self.fast_math = fast_math

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # (hidden, tokens, 1) bf16, hidden contiguous
        mB: cute.Tensor,  # (16, 16, 1) bf16 = H^T
        mColFP4: cute.Tensor,  # (hidden, tokens//8) u32
        mColSF: cute.Tensor,  # flat u32 concatenation of per-group swizzled scales
        mRowFP4: cute.Tensor,  # (tokens, hidden//16) u64: the row code pair per store
        mRowSF: cute.Tensor,  # (tokens//128, hidden//64, 32, 16) e4m3
        row_amax_t: cute.Tensor,  # (num_tensors,) f32
        col_amax_t: cute.Tensor,  # (num_tensors,) f32
        sr_rng_t: cute.Tensor,  # (8,) i32 Philox state
        offsets_t: cute.Tensor,  # (num_tensors,) i32 cumulative row-end offsets
        logical_len_t: cute.Tensor,  # (1,) i32 valid padded token count
        hidden: cutlass.Int32,
        tokens: cutlass.Int32,
        num_tensors: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.c_layout = utils.LayoutEnum.from_tensor(mColFP4)
        (
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            a_clean_layout,
            b_smem_layout_staged,
            acc_fake_layout,
            num_tmem_alloc_cols,
            num_tma_load_bytes,
            num_b_load_bytes,
            tiles_in_n,
            tile_sched_params,
            grid,
        ) = self._setup(mA, mB, hidden, tokens)

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            mColFP4,
            mColSF,
            mRowFP4,
            mRowSF,
            row_amax_t,
            col_amax_t,
            sr_rng_t,
            offsets_t,
            logical_len_t,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            a_clean_layout,
            b_smem_layout_staged,
            acc_fake_layout,
            num_tmem_alloc_cols,
            num_tma_load_bytes,
            num_b_load_bytes,
            tiles_in_n,
            hidden,
            num_tensors,
            tile_sched_params,
        ).launch(grid=grid, block=(TPB, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mColFP4: cute.Tensor,
        mColSF: cute.Tensor,
        mRowFP4: cute.Tensor,
        mRowSF: cute.Tensor,
        row_amax_t: cute.Tensor,
        col_amax_t: cute.Tensor,
        sr_rng_t: cute.Tensor,
        offsets_t: cute.Tensor,
        logical_len_t: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        a_clean_layout: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        acc_fake_layout: cute.Layout,
        num_tmem_alloc_cols: cutlass.Constexpr,
        num_tma_load_bytes: cutlass.Constexpr,
        num_b_load_bytes: cutlass.Constexpr,
        tiles_in_n: cutlass.Int32,
        hidden: cutlass.Int32,
        num_tensors: cutlass.Int32,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, MAINLOOP_STAGES * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
            clc_mbar: cute.struct.MemRange[cutlass.Int64, CLC_STAGES * 2]
            # The scheduler reads each response as one 128-bit load, so the range
            # must carry 16-byte alignment (the struct otherwise aligns to Int64).
            clc_response: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, CLC_STAGES * 4], 16
            ]
            b_mbar: cutlass.Int64
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mainloop: one TMA producer, two heterogeneous consumer groups on the
        # same stage -- the UMMA (released via tcgen05.commit) and the 256 row
        # threads (released via a plain mbarrier arrive). TE hand-rolled this as
        # CustomizedPipelineTmaUmmaAsync; the DSL exposes it directly.
        ab_pipeline = pipeline.PipelineTmaMultiConsumersAsync.create(
            barrier_storage=storage.ab_mbar.data_ptr(),
            num_stages=MAINLOOP_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_umma=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group_async=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, ROW_THREADS
            ),
            tx_count=num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=ACC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            # One arrival per col warp: consumer_release runs under elect_one.
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, COL_WARP_END - COL_WARP_BEGIN
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Every active role consumes exactly one CLC stage per work item. The
        # idle warp must stay out of this count or the pipeline deadlocks.
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbar.data_ptr(),
            num_stages=CLC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                TPB - 32,  # 480
            ),
            tx_count=16,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_ALLOC_BAR,
            num_threads=32 + COL_THREADS,  # mma + col
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_DEALLOC_BAR, num_threads=COL_THREADS
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=COL_WARP_BEGIN,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        if warp_idx == SCHED_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(storage.b_mbar.ptr, 1)

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # A lives in SMEM once; the MMA sees the swizzled UMMA grouping and the
        # row warps see a plain (hidden, token, stage) view of the same bytes.
        a_cosize = cute.cosize(a_smem_layout_staged.outer)
        raw_a = smem.allocate_array(cutlass.BFloat16, a_cosize, byte_alignment=128)
        swz_ptr = cute.recast_ptr(
            raw_a, a_smem_layout_staged.inner, dtype=cutlass.BFloat16
        )
        sA = cute.make_tensor(swz_ptr, a_smem_layout_staged.outer)
        sA_clean = cute.make_tensor(swz_ptr, a_clean_layout.outer)
        sB = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        thr_mma = tiled_mma.get_slice(0)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_((M_TILE, N_TILE, TOKEN_TILE), (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(MMA_TILER, (0, None, None)), (None, None, None)
        )
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)

        cta_layout = cute.make_layout((1,))
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tBgB = tBgB[(None, 0, None, 0)]

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            storage.clc_response.data_ptr(),
        )
        work_tile = tile_sched.initial_work_tile_info()
        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, CLC_STAGES
        )

        # Graph-safe work bound: the scheduler is sized from host-known capacity,
        # but how much of that capacity holds real tokens is only known on device
        # (TE derives the same bound from offsets[num_tensors] at :224). Tiles at
        # or past this point are never loaded or stored.
        tiles_in_n_valid = logical_len_t[0] // cutlass.Int32(TOKEN_TILE)
        # tile_n * tiles_in_h + tile_m gives each tile a stable identity from its
        # coordinates alone. This kernel's CLC scheduler is persistent, so which tile a
        # CTA visits next is not fixed; deriving the SR Philox counter from coordinates
        # rather than from a running per-thread counter is what keeps the stream a pure
        # function of position and makes the same rng_state reproduce the same codes.
        tiles_in_h = hidden // cutlass.Int32(M_TILE)

        # ==================== TMA warp (mainloop producer) ====================
        if warp_idx == TMA_WARP:
            cute.arch.warpgroup_reg_dealloc(REG_DEALLOC)

            # One-shot Hadamard load: 16x16 bf16, never re-armed (TE :546-553).
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    storage.b_mbar.ptr, num_b_load_bytes
                )
            cute.copy(
                tma_atom_b,
                tBgB[(None, 0)],
                tBsB[(None, 0)],
                tma_bar_ptr=storage.b_mbar.ptr,
            )

            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, MAINLOOP_STAGES
            )
            while work_tile.is_valid_tile:
                tile_m, tile_n_base = _work_tile_coord(work_tile)
                n_cnt = _valid_tile_count(
                    tile_n_base,
                    _k_tile_count(tile_n_base, tiles_in_n),
                    tiles_in_n_valid,
                )
                for k_tile in cutlass.range(n_cnt, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, tile_m, tile_n_base + k_tile, 0)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    )
                    ab_producer_state.advance()

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            ab_pipeline.producer_tail(ab_producer_state)

        # ==================== Scheduler warp ====================
        if warp_idx == SCHED_WARP:
            cute.arch.warpgroup_reg_dealloc(REG_DEALLOC)
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, CLC_STAGES
            )
            while work_tile.is_valid_tile:
                clc_pipeline.producer_acquire(clc_producer_state)
                tile_sched.advance_to_next_work(
                    clc_pipeline.producer_get_barrier(clc_producer_state)
                )
                clc_producer_state.advance()

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            clc_pipeline.producer_tail(clc_producer_state)

        # ==================== Idle warp ====================
        # Must dealloc with the rest of warpgroup 0 and must NOT touch the CLC
        # pipeline (it is excluded from the 480 consumer arrivals).
        if warp_idx == IDLE_WARP:
            cute.arch.warpgroup_reg_dealloc(REG_DEALLOC)

        # ==================== MMA warp ====================
        if warp_idx == MMA_WARP:
            cute.arch.warpgroup_reg_dealloc(REG_DEALLOC)
            tmem.wait_for_alloc()
            tCtAcc_base = cute.make_tensor(
                tmem.retrieve_ptr(cutlass.Float32), acc_fake_layout
            )
            cute.arch.mbarrier_wait(storage.b_mbar.ptr, 0)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, MAINLOOP_STAGES
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, ACC_STAGES
            )
            while work_tile.is_valid_tile:
                _, tile_n_base = _work_tile_coord(work_tile)
                n_cnt = _valid_tile_count(
                    tile_n_base,
                    _k_tile_count(tile_n_base, tiles_in_n),
                    tiles_in_n_valid,
                )
                for k_tile in cutlass.range(n_cnt, unroll=1):
                    ab_pipeline.consumer_wait(ab_consumer_state)
                    acc_pipeline.producer_acquire(acc_producer_state)
                    for i in cutlass.range_constexpr(EPI_UNROLL):
                        # ScaleOut.Zero throughout: every UMMA fully overwrites
                        # its 128x16 accumulator over the K=16 Hadamard block.
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        acc = tCtAcc_base[
                            (None, None, None, i, acc_producer_state.index)
                        ]
                        cute.gemm(
                            tiled_mma,
                            acc,
                            tCrA[(None, None, i, ab_consumer_state.index)],
                            tCrB[(None, None, 0, 0)],
                            acc,
                        )
                    acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()
                    ab_pipeline.consumer_release(
                        ab_consumer_state, pipeline.PipelineOp.TCGen05Mma
                    )
                    ab_consumer_state.advance()

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            acc_pipeline.producer_tail(acc_producer_state)

        # ==================== Columnwise epilogue (TMEM -> NVFP4) ====================
        if warp_idx >= COL_WARP_BEGIN and warp_idx < COL_WARP_END:
            cute.arch.warpgroup_reg_alloc(REG_COL)
            tmem.allocate(num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)

            # TE loads this thread's full 8x16 accumulator fragment before
            # releasing the producer. Preserve the MMA's physical TMEM strides
            # while presenting the eight N subtiles as one 128-column tile.
            bulk_tCtAcc = cute.make_tensor(
                tmem_ptr,
                cute.make_layout(
                    (M_TILE, (N_TILE, EPI_UNROLL), ACC_STAGES),
                    stride=(65536, (1, N_TILE), M_TILE),
                ),
            )
            bulk_copy_atom_t2r = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition.x64),
                cutlass.Float32,
            )
            bulk_tiled_copy_t2r = tcgen05.make_tmem_copy(
                bulk_copy_atom_t2r, bulk_tCtAcc[(None, None, 0)]
            )
            bulk_thr_copy_t2r = bulk_tiled_copy_t2r.get_slice(tidx)
            bulk_tTR_tAcc = bulk_thr_copy_t2r.partition_S(bulk_tCtAcc)
            bulk_tTR_rAcc = cute.make_rmem_tensor(((64, 1), 1, 2), cutlass.Float32)

            # This thread owns one hidden row and, per sub-tile, the 16 tokens
            # of one NVFP4 block -> 8 blocks (128 tokens) per epilogue tile.
            h_local = tidx - COL_WARP_BEGIN * cutlass.Int32(32)
            rCol = cute.make_rmem_tensor((2 * EPI_UNROLL,), cutlass.Uint32)
            rColSF = cute.make_rmem_tensor((EPI_UNROLL,), cutlass.Float8E4M3FN)

            col_state = None
            if cutlass.const_expr(self.sr):
                col_state = philox_prep(
                    cutlass.Uint32(sr_rng_t[0]),
                    cutlass.Uint32(sr_rng_t[1]),
                    cutlass.Uint32(sr_rng_t[2]),
                )

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, ACC_STAGES
            )
            while work_tile.is_valid_tile:
                tile_m, tile_n_base = _work_tile_coord(work_tile)
                n_cnt = _valid_tile_count(
                    tile_n_base,
                    _k_tile_count(tile_n_base, tiles_in_n),
                    tiles_in_n_valid,
                )
                if n_cnt > cutlass.Int32(0):
                    g, g_end = _group_at_work_item(tile_n_base, offsets_t, num_tensors)
                    _, dec, enc_over_fp4max = _global_scale(col_amax_t[g])
                    for k_tile in cutlass.range(n_cnt, unroll=1):
                        tile_n = tile_n_base + k_tile
                        token = tile_n * cutlass.Int32(TOKEN_TILE)
                        if token >= g_end:
                            g = _group_idx(token, offsets_t, num_tensors)
                            g_end = offsets_t[g]
                            _, dec, enc_over_fp4max = _global_scale(col_amax_t[g])
                        h_global = tile_m * cutlass.Int32(M_TILE) + h_local
                        tile_id = tile_n * tiles_in_h + tile_m

                        acc_pipeline.consumer_wait(acc_consumer_state)
                        cute.copy(
                            bulk_tiled_copy_t2r,
                            bulk_tTR_tAcc[(None, None, None, acc_consumer_state.index)],
                            bulk_tTR_rAcc,
                        )
                        bulk_vals = bulk_tTR_rAcc.load().reshape((16, 8))
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()

                        for u in cutlass.range_constexpr(EPI_UNROLL):
                            vals = bulk_vals[(None, u)]
                            col_rb = None
                            if cutlass.const_expr(self.sr):
                                # One draw per 16-element block, indexed by its position in
                                # the columnwise (hidden, tokens) tile.
                                col_rb = philox4_all(
                                    col_state,
                                    tile_id * cutlass.Int32(TILE_BLOCKS)
                                    + h_local * cutlass.Int32(TOKEN_TILE // 16)
                                    + cutlass.Int32(u),
                                )
                            w0, w1, sf = _quant16(
                                vals,
                                enc_over_fp4max,
                                dec,
                                self.sr,
                                col_rb,
                                rht_acc=True,
                                fast_math=self.fast_math,
                            )
                            rCol[u * 2] = w0
                            rCol[u * 2 + 1] = w1
                            rColSF[u] = sf

                        gCol = cute.local_tile(
                            mColFP4, (M_TILE, TOKEN_TILE // 8), (tile_m, tile_n)
                        )
                        cute.autovec_copy(rCol, gCol[(h_local, None)])
                        _store_grouped_col_sf_u32(
                            mColSF,
                            rColSF,
                            h_global,
                            tile_n * cutlass.Int32(TOKEN_TILE // 16),
                            g,
                            offsets_t,
                            hidden,
                        )

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ==================== Rowwise epilogue (SMEM -> NVFP4) ====================
        if warp_idx >= ROW_WARP_BEGIN and warp_idx < ROW_WARP_END:
            cute.arch.warpgroup_reg_alloc(REG_ROW)
            r_local = tidx - ROW_WARP_BEGIN * cutlass.Int32(32)
            hb = r_local % cutlass.Int32(ROW_HB)  # 16-hidden block
            t0 = r_local // cutlass.Int32(ROW_HB)  # token within a pass

            blk = cute.make_rmem_tensor((16,), cutlass.Float32)
            rBlk = cute.make_rmem_tensor((16,), cutlass.BFloat16)
            rPair = cute.make_rmem_tensor((2,), cutlass.Uint32)
            row_state = None
            if cutlass.const_expr(self.sr):
                row_state = philox_prep(
                    cutlass.Uint32(sr_rng_t[4]),
                    cutlass.Uint32(sr_rng_t[5]),
                    cutlass.Uint32(sr_rng_t[6]),
                )
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, MAINLOOP_STAGES
            )
            while work_tile.is_valid_tile:
                tile_m, tile_n_base = _work_tile_coord(work_tile)
                n_cnt = _valid_tile_count(
                    tile_n_base,
                    _k_tile_count(tile_n_base, tiles_in_n),
                    tiles_in_n_valid,
                )
                if n_cnt > cutlass.Int32(0):
                    g, g_end = _group_at_work_item(tile_n_base, offsets_t, num_tensors)
                    _, r_dec, r_enc_over_fp4max = _global_scale(row_amax_t[g])
                    for k_tile in cutlass.range(n_cnt, unroll=1):
                        token = (tile_n_base + k_tile) * cutlass.Int32(TOKEN_TILE)
                        if token >= g_end:
                            g = _group_idx(token, offsets_t, num_tensors)
                            g_end = offsets_t[g]
                            _, r_dec, r_enc_over_fp4max = _global_scale(row_amax_t[g])

                        ab_pipeline.consumer_wait(ab_consumer_state)
                        stage = ab_consumer_state.index
                        tile_n = tile_n_base + k_tile
                        gRow = cute.local_tile(
                            mRowFP4, (TOKEN_TILE, M_TILE // 16), (tile_n, tile_m)
                        )
                        if cutlass.const_expr(self.sr):
                            tile_id = tile_n * tiles_in_h + tile_m
                        for p in cutlass.range_constexpr(ROW_PASSES):
                            tok = p * cutlass.Int32(ROW_TOK_PER_PASS) + t0
                            cute.autovec_copy(
                                cute.local_tile(
                                    sA_clean[(None, tok, stage)], (16,), (hb,)
                                ),
                                rBlk,
                            )
                            rWords = cute.recast_tensor(rBlk, cutlass.Uint32)
                            for j in cutlass.range_constexpr(8):
                                blk[2 * j] = _bf16lo_to_f32(rWords[j])
                                blk[2 * j + 1] = _bf16hi_to_f32(rWords[j])
                            row_rb = None
                            if cutlass.const_expr(self.sr):
                                # One draw per 16-element block, indexed by its position in
                                # the rowwise (tokens, hidden) tile.
                                row_rb = philox4_all(
                                    row_state,
                                    tile_id * cutlass.Int32(TILE_BLOCKS)
                                    + tok * cutlass.Int32(M_TILE // 16)
                                    + hb,
                                )
                            w0, w1, sf = _quant16(
                                blk,
                                r_enc_over_fp4max,
                                r_dec,
                                self.sr,
                                row_rb,
                                fast_math=self.fast_math,
                            )
                            # One 64-bit store, not two 32-bit ones. A warp covers 4
                            # tokens x ROW_HB hidden blocks, so consecutive lanes differ in
                            # hb: as two u32 stores each wrote 4B at an 8B lane stride,
                            # spanning 64B to fill 32B, and every sector came back half
                            # wasted (8.00 sectors/instruction against an ideal of 4.00).
                            # Storing the pair makes the lanes contiguous and lands at the
                            # ideal, which is what TE's STG.E.64 already does here.
                            rPair[0] = w0
                            rPair[1] = w1
                            pair64 = cute.recast_tensor(rPair, cutlass.Uint64)
                            gRow[(tok, hb)] = pair64[0]
                            _store_sf_byte(
                                mRowSF,
                                sf,
                                tile_n * cutlass.Int32(TOKEN_TILE) + tok,
                                tile_m * cutlass.Int32(ROW_HB) + hb,
                            )
                        ab_pipeline.consumer_release(
                            ab_consumer_state, pipeline.PipelineOp.AsyncThread
                        )
                        ab_consumer_state.advance()

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()


# maxsize=None and no defaults, for the reasons spelled out over ``_compile_fused_kernel``:
# an entry is a compiled kernel a CUDA-graph capture may depend on, so the cache must never
# evict, and the key is the literal (args, kwargs) shape, so every caller passes all three
# positionally or the pre-capture warm-up warms keys nothing will look up.
@functools.lru_cache(maxsize=None)
def _compile_group_fused_kernel(device_idx: int, sr: bool, fast_math: bool):
    """Compile the grouped fused kernel with symbolic shapes (cached per device+flags).

    ``sym_int`` divisibilities let one compiled kernel serve any
    ``hidden % 128``, ``tokens % 128``. Exact and fast arithmetic are separate
    cache entries; sign vector, amaxes, offsets, and RNG remain runtime buffers.
    """
    free = cute.sym_int
    h_sym = cute.sym_int(divisibility=M_TILE)
    t_sym = cute.sym_int(divisibility=TOKEN_TILE)

    fake_a = make_fake_tensor(
        cutlass.BFloat16, (h_sym, t_sym, 1), stride=(1, free(), 1)
    )
    fake_b = make_fake_tensor(
        cutlass.BFloat16, (HADAMARD_DIM, HADAMARD_DIM, 1), stride=(HADAMARD_DIM, 1, 1)
    )
    # The columnwise epilogue stores a thread's 16 contiguous u32 with one
    # autovec_copy, which widens only as far as it can prove alignment: the u32
    # default is 4B, so it lowers to sixteen scalar STG. The allocation is
    # torch.empty (256B) and tokens % TOKEN_TILE == 0 makes the row stride a
    # multiple of TOKEN_TILE // 8 u32 = 64B, so every row start is 16B aligned.
    fake_col_fp4 = make_fake_tensor(
        cutlass.Uint32,
        (h_sym, cute.sym_int(divisibility=TOKEN_TILE // 8)),
        stride=(cute.sym_int(divisibility=TOKEN_TILE // 8), 1),
        assumed_align=16,
    )
    fake_col_sf = make_fake_tensor(cutlass.Uint32, (free(),), stride=(1,))
    # u64: the row epilogue stores the two code words of a 16-hidden block together.
    # The allocation is torch.empty (256B) and hidden % 128 == 0 makes the row stride
    # hidden/2 bytes, a multiple of 64B, so every row start is 16B aligned.
    fake_row_fp4 = make_fake_tensor(
        cutlass.Uint64,
        (t_sym, cute.sym_int(divisibility=M_TILE // 16)),
        stride=(free(), 1),
        assumed_align=16,
    )
    fake_row_sf = make_fake_tensor(
        cutlass.Float8E4M3FN, (free(), free(), 32, 16), stride=(free(), 512, 16, 1)
    )
    fake_amax = make_fake_tensor(cutlass.Float32, (free(),), stride=(1,))
    fake_i32_1 = make_fake_tensor(cutlass.Int32, (1,), stride=(1,))
    # (8,) Philox state: [col_seed_lo/hi, col_off_lo/hi, row_seed_lo/hi, row_off_lo/hi].
    fake_sr_rng = make_fake_tensor(cutlass.Int32, (8,), stride=(1,))
    fake_offsets = make_fake_tensor(cutlass.Int32, (free(),), stride=(1,))

    return cute.compile(
        _Tcgen05GroupRowColFused(sr=sr, fast_math=fast_math),
        fake_a,
        fake_b,
        fake_col_fp4,
        fake_col_sf,
        fake_row_fp4,
        fake_row_sf,
        fake_amax,
        fake_amax,
        fake_sr_rng,
        fake_offsets,
        fake_i32_1,
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _cutedsl_group_rht_quantize_row_col_impl(
    A: torch.Tensor,
    offsets: torch.Tensor,
    row_global_amax: torch.Tensor,
    col_global_amax: torch.Tensor,
    num_tensors: int,
    sign_vector=DEFAULT_SIGN_VECTOR,
    logical_packed_length: Optional[torch.Tensor] = None,
    stochastic_rounding: bool = False,
    sr_rng: Optional[torch.Tensor] = None,
    use_fast_math: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grouped fused RHT columnwise + raw rowwise NVFP4 quantization.

    ``A`` is ``(tokens, hidden)`` bfloat16 row-major with ``tokens % 128 == 0``
    and ``hidden % 128 == 0``. Returns ``(col_fp4, col_sf, row_fp4, row_sf)``
    with swizzled scale factors; the wrapper returns views matching torchao's
    ``(qa, sfa, qd, sfd)`` contract. Columnwise scale storage is a flat
    concatenation of independently swizzled group buffers.
    """
    tokens, hidden = A.shape
    dev = A.device
    A = A.detach()

    col_fp4 = torch.empty((hidden, tokens // 8), dtype=torch.uint32, device=dev)
    row_fp4 = torch.empty((tokens, hidden // 8), dtype=torch.uint32, device=dev)
    col_sf = torch.empty(
        (hidden // 128, tokens // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev
    )
    row_sf = torch.empty(
        (tokens // 128, hidden // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev
    )

    rht_nk = _get_rht_buffer(tuple(sign_vector), dev.index)
    sr_rng_t = _get_sr_rng_buffer(dev.index)
    if stochastic_rounding:
        # [col_seed, col_offset, row_seed, row_offset] int64 -> the eight little-endian
        # 32-bit halves Philox keys and counters are built from. One 32-byte D2D copy,
        # so it stays graph-capturable and does no host RNG.
        sr_rng_t.copy_(sr_rng[:4].view(torch.int32))
    if logical_packed_length is None:
        logical_packed_length = offsets[-1:]
    # The CuteDSL entry point requires byte_offset==0, and offsets[-1:] is a
    # nonzero-offset view for every multi-group launch. Cloning is device-side
    # and stays capturable.
    logical_packed_length = logical_packed_length.clone()

    stream = cuda.CUstream(int(torch.cuda.current_stream(dev).cuda_stream))
    compiled = _compile_group_fused_kernel(
        dev.index, bool(stochastic_rounding), bool(use_fast_math)
    )
    compiled(
        A.t().unsqueeze(-1),
        rht_nk,
        col_fp4,
        col_sf.view(torch.uint32).flatten(),
        row_fp4.view(torch.uint64),
        row_sf,
        row_global_amax,
        col_global_amax,
        sr_rng_t,
        offsets,
        logical_packed_length,
        int(hidden),
        int(tokens),
        int(num_tensors),
        stream,
    )
    return col_fp4.view(torch.uint8), col_sf, row_fp4.view(torch.uint8), row_sf


class _Tcgen05GroupRhtAmax(_GroupRhtMainloop):
    """Per-group post-RHT columnwise amax and raw rowwise amax.

    Identical mainloop to the fused kernel; both epilogues reduce to a max-abs
    instead of quantizing. A 128-token tile lies entirely inside one group, so
    the group index is CTA-uniform and each warp contributes one atomic per
    tile. The running max is per tile, not per work item, because consecutive
    tiles can belong to different groups.
    """

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        col_amax_t: cute.Tensor,  # (num_tensors,) f32, pre-zeroed
        row_amax_t: cute.Tensor,  # (num_tensors,) f32, pre-zeroed
        offsets_t: cute.Tensor,
        logical_len_t: cute.Tensor,
        hidden: cutlass.Int32,
        tokens: cutlass.Int32,
        num_tensors: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        (
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            a_clean_layout,
            b_smem_layout_staged,
            acc_fake_layout,
            num_tmem_alloc_cols,
            num_tma_load_bytes,
            num_b_load_bytes,
            tiles_in_n,
            tile_sched_params,
            grid,
        ) = self._setup(mA, mB, hidden, tokens)

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            col_amax_t,
            row_amax_t,
            offsets_t,
            logical_len_t,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            a_clean_layout,
            b_smem_layout_staged,
            acc_fake_layout,
            num_tmem_alloc_cols,
            num_tma_load_bytes,
            num_b_load_bytes,
            tiles_in_n,
            num_tensors,
            tile_sched_params,
        ).launch(grid=grid, block=(TPB, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        col_amax_t: cute.Tensor,
        row_amax_t: cute.Tensor,
        offsets_t: cute.Tensor,
        logical_len_t: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        a_clean_layout: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        acc_fake_layout: cute.Layout,
        num_tmem_alloc_cols: cutlass.Constexpr,
        num_tma_load_bytes: cutlass.Constexpr,
        num_b_load_bytes: cutlass.Constexpr,
        tiles_in_n: cutlass.Int32,
        num_tensors: cutlass.Int32,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % cutlass.Int32(32)

        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, MAINLOOP_STAGES * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
            clc_mbar: cute.struct.MemRange[cutlass.Int64, CLC_STAGES * 2]
            # The scheduler reads each response as one 128-bit load, so the range
            # must carry 16-byte alignment (the struct otherwise aligns to Int64).
            clc_response: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, CLC_STAGES * 4], 16
            ]
            b_mbar: cutlass.Int64
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline = pipeline.PipelineTmaMultiConsumersAsync.create(
            barrier_storage=storage.ab_mbar.data_ptr(),
            num_stages=MAINLOOP_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_umma=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group_async=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, ROW_THREADS
            ),
            tx_count=num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=ACC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, COL_WARP_END - COL_WARP_BEGIN
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbar.data_ptr(),
            num_stages=CLC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, TPB - 32),
            tx_count=16,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_ALLOC_BAR, num_threads=32 + COL_THREADS
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_DEALLOC_BAR, num_threads=COL_THREADS
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=COL_WARP_BEGIN,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        if warp_idx == SCHED_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(storage.b_mbar.ptr, 1)

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        a_cosize = cute.cosize(a_smem_layout_staged.outer)
        raw_a = smem.allocate_array(cutlass.BFloat16, a_cosize, byte_alignment=128)
        swz_ptr = cute.recast_ptr(
            raw_a, a_smem_layout_staged.inner, dtype=cutlass.BFloat16
        )
        sA = cute.make_tensor(swz_ptr, a_smem_layout_staged.outer)
        sA_clean = cute.make_tensor(swz_ptr, a_clean_layout.outer)
        sB = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        thr_mma = tiled_mma.get_slice(0)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_((M_TILE, N_TILE, TOKEN_TILE), (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(MMA_TILER, (0, None, None)), (None, None, None)
        )
        cta_layout = cute.make_layout((1,))
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gA_mkl), 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(thr_mma.partition_B(gB_nkl), 0, 3),
        )
        tBgB = tBgB[(None, 0, None, 0)]

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            storage.clc_response.data_ptr(),
        )
        work_tile = tile_sched.initial_work_tile_info()
        clc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, CLC_STAGES
        )
        tiles_in_n_valid = logical_len_t[0] // cutlass.Int32(TOKEN_TILE)

        # ==================== TMA warp ====================
        if warp_idx == TMA_WARP:
            cute.arch.warpgroup_reg_dealloc(REG_DEALLOC)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    storage.b_mbar.ptr, num_b_load_bytes
                )
            cute.copy(
                tma_atom_b,
                tBgB[(None, 0)],
                tBsB[(None, 0)],
                tma_bar_ptr=storage.b_mbar.ptr,
            )
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, MAINLOOP_STAGES
            )
            while work_tile.is_valid_tile:
                tile_m, tile_n_base = _work_tile_coord(work_tile)
                n_cnt = _valid_tile_count(
                    tile_n_base,
                    _k_tile_count(tile_n_base, tiles_in_n),
                    tiles_in_n_valid,
                )
                for k_tile in cutlass.range(n_cnt, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, tile_m, tile_n_base + k_tile, 0)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                    )
                    ab_producer_state.advance()
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            ab_pipeline.producer_tail(ab_producer_state)

        # ==================== Scheduler warp ====================
        if warp_idx == SCHED_WARP:
            cute.arch.warpgroup_reg_dealloc(REG_DEALLOC)
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, CLC_STAGES
            )
            while work_tile.is_valid_tile:
                clc_pipeline.producer_acquire(clc_producer_state)
                tile_sched.advance_to_next_work(
                    clc_pipeline.producer_get_barrier(clc_producer_state)
                )
                clc_producer_state.advance()
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            clc_pipeline.producer_tail(clc_producer_state)

        # ==================== Idle warp ====================
        if warp_idx == IDLE_WARP:
            cute.arch.warpgroup_reg_dealloc(REG_DEALLOC)

        # ==================== MMA warp ====================
        if warp_idx == MMA_WARP:
            cute.arch.warpgroup_reg_dealloc(REG_DEALLOC)
            tmem.wait_for_alloc()
            tCtAcc_base = cute.make_tensor(
                tmem.retrieve_ptr(cutlass.Float32), acc_fake_layout
            )
            cute.arch.mbarrier_wait(storage.b_mbar.ptr, 0)
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, MAINLOOP_STAGES
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, ACC_STAGES
            )
            while work_tile.is_valid_tile:
                _, tile_n_base = _work_tile_coord(work_tile)
                n_cnt = _valid_tile_count(
                    tile_n_base,
                    _k_tile_count(tile_n_base, tiles_in_n),
                    tiles_in_n_valid,
                )
                for k_tile in cutlass.range(n_cnt, unroll=1):
                    ab_pipeline.consumer_wait(ab_consumer_state)
                    acc_pipeline.producer_acquire(acc_producer_state)
                    for i in cutlass.range_constexpr(EPI_UNROLL):
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        acc = tCtAcc_base[
                            (None, None, None, i, acc_producer_state.index)
                        ]
                        cute.gemm(
                            tiled_mma,
                            acc,
                            tCrA[(None, None, i, ab_consumer_state.index)],
                            tCrB[(None, None, 0, 0)],
                            acc,
                        )
                    acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()
                    ab_pipeline.consumer_release(
                        ab_consumer_state, pipeline.PipelineOp.TCGen05Mma
                    )
                    ab_consumer_state.advance()
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            acc_pipeline.producer_tail(acc_producer_state)

        # ==================== Columnwise amax (post-RHT, from TMEM) ====================
        if warp_idx >= COL_WARP_BEGIN and warp_idx < COL_WARP_END:
            cute.arch.warpgroup_reg_alloc(REG_COL)
            tmem.allocate(num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc_base = cute.make_tensor(tmem_ptr, acc_fake_layout)

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                MMA_TILER,
                self.c_layout,
                cutlass.Float32,
                cutlass.Float32,
                MMA_TILER[:2],
                False,
            )
            tAcc = transform_partitioned_tensor_layout(tCtAcc_base)
            tAcc_epi = cute.flat_divide(tAcc, MMA_TILER[:2])
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
            tTR_rAcc = cute.make_rmem_tensor(((16, 1), 1, 1), cutlass.Float32)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, ACC_STAGES
            )
            while work_tile.is_valid_tile:
                _, tile_n_base = _work_tile_coord(work_tile)
                n_cnt = _valid_tile_count(
                    tile_n_base,
                    _k_tile_count(tile_n_base, tiles_in_n),
                    tiles_in_n_valid,
                )
                if n_cnt > cutlass.Int32(0):
                    g, g_end = _group_at_work_item(tile_n_base, offsets_t, num_tensors)
                    run_max = cutlass.Float32(0.0)
                    for k_tile in cutlass.range(n_cnt, unroll=1):
                        token = (tile_n_base + k_tile) * cutlass.Int32(TOKEN_TILE)
                        if token >= g_end:
                            # Crossing out of the cached group is the only point the
                            # running max has to reach memory: everything before it
                            # belongs to `g`, everything after to the next group.
                            _flush_group_max(run_max, col_amax_t, g, lane)
                            run_max = cutlass.Float32(0.0)
                            g = _group_idx(token, offsets_t, num_tensors)
                            g_end = offsets_t[g]
                        acc_pipeline.consumer_wait(acc_consumer_state)
                        tile_max = cutlass.Float32(0.0)
                        for u in cutlass.range_constexpr(EPI_UNROLL):
                            cute.copy(
                                tiled_copy_t2r,
                                tTR_tAcc[
                                    (
                                        None,
                                        None,
                                        None,
                                        0,
                                        0,
                                        u,
                                        acc_consumer_state.index,
                                    )
                                ],
                                tTR_rAcc,
                            )
                            vals = tTR_rAcc.load().reshape((16,))
                            for i in cutlass.range_constexpr(16):
                                tile_max = _max_f32(tile_max, _abs_f32(vals[i]))
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()

                        run_max = _max_f32(run_max, _round_rht_amax(tile_max))
                    _flush_group_max(run_max, col_amax_t, g, lane)

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ==================== Rowwise amax (raw A, from SMEM) ====================
        if warp_idx >= ROW_WARP_BEGIN and warp_idx < ROW_WARP_END:
            cute.arch.warpgroup_reg_alloc(REG_ROW)
            r_local = tidx - ROW_WARP_BEGIN * cutlass.Int32(32)
            hb = r_local % cutlass.Int32(ROW_HB)
            t0 = r_local // cutlass.Int32(ROW_HB)

            rBlk = cute.make_rmem_tensor((16,), cutlass.BFloat16)
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, MAINLOOP_STAGES
            )
            while work_tile.is_valid_tile:
                _, tile_n_base = _work_tile_coord(work_tile)
                n_cnt = _valid_tile_count(
                    tile_n_base,
                    _k_tile_count(tile_n_base, tiles_in_n),
                    tiles_in_n_valid,
                )
                if n_cnt > cutlass.Int32(0):
                    g, g_end = _group_at_work_item(tile_n_base, offsets_t, num_tensors)
                    run_max = cutlass.Float32(0.0)
                    for k_tile in cutlass.range(n_cnt, unroll=1):
                        token = (tile_n_base + k_tile) * cutlass.Int32(TOKEN_TILE)
                        if token >= g_end:
                            _flush_group_max(run_max, row_amax_t, g, lane)
                            run_max = cutlass.Float32(0.0)
                            g = _group_idx(token, offsets_t, num_tensors)
                            g_end = offsets_t[g]
                        ab_pipeline.consumer_wait(ab_consumer_state)
                        stage = ab_consumer_state.index
                        tile_max = cutlass.Float32(0.0)
                        for p in cutlass.range_constexpr(ROW_PASSES):
                            tok = p * cutlass.Int32(ROW_TOK_PER_PASS) + t0
                            cute.autovec_copy(
                                cute.local_tile(
                                    sA_clean[(None, tok, stage)], (16,), (hb,)
                                ),
                                rBlk,
                            )
                            rWords = cute.recast_tensor(rBlk, cutlass.Uint32)
                            for j in cutlass.range_constexpr(8):
                                tile_max = _max_f32(
                                    tile_max, _abs_f32(_bf16lo_to_f32(rWords[j]))
                                )
                                tile_max = _max_f32(
                                    tile_max, _abs_f32(_bf16hi_to_f32(rWords[j]))
                                )
                        ab_pipeline.consumer_release(
                            ab_consumer_state, pipeline.PipelineOp.AsyncThread
                        )
                        ab_consumer_state.advance()

                        run_max = _max_f32(run_max, tile_max)
                    _flush_group_max(run_max, row_amax_t, g, lane)

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()


@functools.lru_cache(maxsize=None)
def _compile_group_amax_kernel(device_idx: int):
    """Compile the grouped RHT amax kernel with symbolic shapes."""
    free = cute.sym_int
    h_sym = cute.sym_int(divisibility=M_TILE)
    t_sym = cute.sym_int(divisibility=TOKEN_TILE)

    k = _Tcgen05GroupRhtAmax()
    # The TMEM->register op is selected from the row/col-majorness of the
    # columnwise output; the amax kernel has none, so a contiguous stand-in of
    # the same shape picks the same enum.
    dummy = torch.empty(
        (M_TILE, N_TILE), dtype=torch.int32, device=torch.device("cuda", device_idx)
    )
    k.c_layout = utils.LayoutEnum.from_tensor(from_dlpack(dummy))

    return cute.compile(
        k,
        make_fake_tensor(cutlass.BFloat16, (h_sym, t_sym, 1), stride=(1, free(), 1)),
        make_fake_tensor(
            cutlass.BFloat16,
            (HADAMARD_DIM, HADAMARD_DIM, 1),
            stride=(HADAMARD_DIM, 1, 1),
        ),
        make_fake_tensor(cutlass.Float32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Float32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Int32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Int32, (1,), stride=(1,)),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _cutedsl_group_rht_amax_impl(
    A: torch.Tensor,
    offsets: torch.Tensor,
    num_tensors: int,
    sign_vector=DEFAULT_SIGN_VECTOR,
    logical_packed_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-group ``max|RHT(A.t())|`` and ``max|A|``.

    ``A`` is ``(tokens, hidden)`` bfloat16 row-major. Returns
    ``(col_amax, row_amax)``, each ``(num_tensors,)`` float32. The buffers start
    at zero because the epilogues accumulate with atomic max.
    """
    tokens, hidden = A.shape
    dev = A.device
    A = A.detach()

    col_amax = torch.zeros((num_tensors,), dtype=torch.float32, device=dev)
    row_amax = torch.zeros((num_tensors,), dtype=torch.float32, device=dev)
    rht_nk = _get_rht_buffer(tuple(sign_vector), dev.index)
    if logical_packed_length is None:
        logical_packed_length = offsets[-1:]
    # See the fused kernel: the entry point requires byte_offset==0.
    logical_packed_length = logical_packed_length.clone()

    stream = cuda.CUstream(int(torch.cuda.current_stream(dev).cuda_stream))
    _compile_group_amax_kernel(dev.index)(
        A.t().unsqueeze(-1),
        rht_nk,
        col_amax,
        row_amax,
        offsets,
        logical_packed_length,
        int(hidden),
        int(tokens),
        int(num_tensors),
        stream,
    )
    return col_amax, row_amax


def _work_tile_coord(work_tile):
    """(hidden tile, first token tile) for a scheduler work item."""
    coord = work_tile.tile_idx
    return coord[0], coord[1] * cutlass.Int32(K_TILE_MAX)


def _k_tile_count(tile_n_base, tiles_in_n):
    """Token tiles in this work item, clamped to the problem (TE :566)."""
    rem = tiles_in_n - tile_n_base
    return cutlass.select_(
        rem < cutlass.Int32(K_TILE_MAX), rem, cutlass.Int32(K_TILE_MAX)
    )


def _valid_tile_count(tile_n_base, n_all, tiles_in_n_valid):
    """Of this work item's ``n_all`` tiles, how many precede the logical bound."""
    rem = tiles_in_n_valid - tile_n_base
    rem = cutlass.select_(rem > cutlass.Int32(0), rem, cutlass.Int32(0))
    return cutlass.select_(rem < n_all, rem, n_all)


def _store_grouped_col_sf_u32(mSF_u32, rSF, r, c_base, g, offsets_t, hidden):
    """Store 8 columnwise scale bytes in group-local swizzled tiles.

    Each group owns a separately swizzled ``(hidden, group_tokens // 16)``
    scale buffer. ``mSF_u32`` is their flat concatenation, so the address is
    the span of preceding groups plus this hidden block's group-local
    64-token tile. The byte order within a tile matches the standard NVFP4
    swizzle used by ``_store_sf_byte``.
    """
    prev = cutlass.select_(g > cutlass.Int32(0), g - cutlass.Int32(1), 0)
    group_start = cutlass.select_(
        g > cutlass.Int32(0), offsets_t[prev], cutlass.Int32(0)
    )
    group_len = offsets_t[g] - group_start
    r_blk = r // cutlass.Int32(128)
    r_lane = r % cutlass.Int32(32)
    r_grp = (r % cutlass.Int32(128)) // cutlass.Int32(32)
    # A group has ``hidden * group_len / 64`` u32 scale words.  The preceding
    # concatenated groups cover the same expression with ``group_start``.
    # The product is taken in 64 bits: at DeepSeek-V3 671B (hidden 7168) it passes
    # 2^31 once ``group_start`` reaches 299,593 rows, and an Int32 multiply wraps
    # negative there, so the store lands far below the buffer. The quotient is at
    # most ``hidden * tokens / 64``, which is comfortably Int32, so only the
    # multiply needs widening and the index arithmetic below stays 32-bit.
    prefix_words = cutlass.Int32(
        cutlass.Int64(hidden) * cutlass.Int64(group_start) // cutlass.Int64(64)
    )
    words_per_hidden_block = group_len * cutlass.Int32(2)
    c_local = c_base - group_start // cutlass.Int32(16)
    # Plain Python loops: this helper is not AST-preprocessed, so the trace
    # unrolls them the same way cutlass.range_constexpr would inside a kernel.
    for half in range(2):
        packed = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
        for i in range(4):
            packed[i] = rSF[half * 4 + i]
        word = cute.recast_tensor(packed, cutlass.Uint32)[0]
        word_col = c_local // cutlass.Int32(4) + cutlass.Int32(half)
        mSF_u32[
            prefix_words
            + r_blk * words_per_hidden_block
            + word_col * cutlass.Int32(128)
            + r_lane * cutlass.Int32(4)
            + r_grp
        ] = word


def _store_sf_byte(mSF, sf, r, c):
    """Scatter one swizzled scale-factor byte.

    The cutlass NVFP4 layout maps logical ``SF[r, c]`` to
    ``storage[r//128, c//4, r%32, (r%128//32)*4 + c%4]`` over a
    ``(R//128, C//4, 32, 16)`` buffer.
    """
    mSF[
        (
            r // cutlass.Int32(128),
            c // cutlass.Int32(4),
            r % cutlass.Int32(32),
            ((r % cutlass.Int32(128)) // cutlass.Int32(32)) * cutlass.Int32(4)
            + c % cutlass.Int32(4),
        )
    ] = sf


# --- grouped rowwise-RHT + columnwise-RHT amax (RHT-128 on both axes) ---
# The unprefixed constants above belong to the RHT-16 kernels; ``RHT128_`` names
# the distinguishing property of this kernel family.
RHT128_DIM = 128
RHT128_K_BLOCKS = RHT128_DIM // K  # UMMA K steps per transform
RHT128_MMA_TILER_COL = (
    M_TILE,
    RHT128_DIM,
    K,
)  # col chain: M = hidden, N = j, K = tokens
RHT128_MMA_TILER_ROW = (
    TOKEN_TILE,
    RHT128_DIM,
    K,
)  # row chain: M = tokens, N = j, K = hidden
RHT128_CTA_TILE_COL = (M_TILE, RHT128_DIM, TOKEN_TILE)
RHT128_CTA_TILE_ROW = (TOKEN_TILE, RHT128_DIM, M_TILE)
RHT128_EPI_TILE = (M_TILE, N_TILE)  # x16 TMEM load atom
RHT128_B_BYTES = RHT128_DIM * RHT128_DIM * 2  # one resident signed R^T tile
RHT128_MAINLOOP_STAGES = (
    _SMEM_CAPACITY - _SMEM_RESERVE - 2 * RHT128_B_BYTES
) // _A_TILE_BYTES  # 5
RHT128_ACC_STAGES = 2  # per chain; 2 chains x 2 stages x 128 cols = 512 TMEM cols
RHT128_ROW_WARP_END = 12  # warps 0 MMA, 1 TMA, 2-3 idle, 4-7 col, 8-11 row
RHT128_N_WARPS = 12
RHT128_TPB = 32 * RHT128_N_WARPS
RHT128_ACC_CONSUMER_WARPS = (COL_WARP_END - COL_WARP_BEGIN) + (
    RHT128_ROW_WARP_END - ROW_WARP_BEGIN
)
RHT128_ENTRY_BITS = 0x3DB5  # bfloat16(1 / sqrt(128)): |entry| of the normalized H128
RHT128_B_READY_BAR = 3  # both R^T tiles written to shared memory


def _static_tile_range(total, cta, n_ctas):
    """Contiguous balanced chunk ``[begin, end)`` of ``total`` tiles for CTA ``cta``.

    ``begin <= max(total - 1, 0)`` for every ``cta < n_ctas``, so an empty chunk
    still evaluates ``_group_idx`` on an in-range tile and its final zero flush is
    a no-op against the pre-zeroed buffer -- no dynamic guard is needed.
    """
    return (total * cta) // n_ctas, (total * (cta + 1)) // n_ctas


@cute.jit
def _rht128_tile_amax(acc, tidx):
    """max|acc| over one 128x128 f32 TMEM accumulator, per thread (= per lane)."""
    copy_atom_t2r = sm100_utils.get_tmem_load_op(
        RHT128_CTA_TILE_COL,
        utils.LayoutEnum.ROW_MAJOR,
        cutlass.Float32,
        cutlass.Float32,
        RHT128_EPI_TILE,
        False,
    )
    tAcc = transform_partitioned_tensor_layout(acc)
    tAcc_epi = cute.flat_divide(tAcc, RHT128_EPI_TILE)
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
    thr_copy = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc = thr_copy.partition_S(tAcc_epi)
    tTR_rAcc = cute.make_rmem_tensor(((16, 1), 1, 1), cutlass.Float32)
    tile_max = cutlass.Float32(0.0)
    for u in cutlass.range_constexpr(RHT128_DIM // RHT128_EPI_TILE[1]):
        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, u)], tTR_rAcc)
        vals = tTR_rAcc.load().reshape((16,))
        for i in cutlass.range_constexpr(16):
            tile_max = _max_f32(tile_max, _abs_f32(vals[i]))
    return tile_max


@cute.jit
def _rht128_amax_epilogue(
    chain: cutlass.Constexpr,
    amax_t,
    tCtAcc,
    acc_pipeline,
    offsets_t,
    num_tensors,
    t_begin,
    t_end,
    tiles_in_m,
    tidx,
    lane,
):
    """One epilogue body for both chains: consume the acc ring over this CTA's chunk.

    Tiles are enumerated hidden-fastest, so ``token`` is non-decreasing and the
    group cache / crossing flush is ``_Tcgen05GroupRhtAmax``'s.
    """
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, RHT128_ACC_STAGES
    )
    g = _group_idx(
        (t_begin // tiles_in_m) * cutlass.Int32(TOKEN_TILE), offsets_t, num_tensors
    )
    g_end = offsets_t[g]
    run_max = cutlass.Float32(0.0)
    for i in cutlass.range(t_end - t_begin, unroll=1):
        token = ((t_begin + i) // tiles_in_m) * cutlass.Int32(TOKEN_TILE)
        if token >= g_end:
            _flush_group_max(run_max, amax_t, g, lane)
            run_max = cutlass.Float32(0.0)
            g = _group_idx(token, offsets_t, num_tensors)
            g_end = offsets_t[g]
        acc_pipeline.consumer_wait(acc_state)
        tile_max = _rht128_tile_amax(
            tCtAcc[(None, None, None, 2 * acc_state.index + chain)], tidx
        )
        cute.arch.fence_view_async_tmem_load()
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_state)
        acc_state.advance()
        run_max = _max_f32(run_max, _round_rht_amax(tile_max))
    _flush_group_max(run_max, amax_t, g, lane)


@functools.lru_cache(maxsize=None)
def _get_group_amax_reduce_buffers(device_idx):
    """Persistent per-device cross-CTA reduction state of ``_Tcgen05GroupRowRhtColRhtAmax``:
    the ``(1,)`` int32 arrival ticket and the ``(2 * MAX_GROUPS,)`` float32 accumulator the
    epilogues atomic-max into, both zero between launches -- the last CTA copies the
    accumulator's ``2 * num_tensors`` slots into the outputs and re-zeroes them, and
    resets the ticket. Persistent for the reason ``_get_sr_rng_buffer`` is: a fresh
    per-call tensor would be an untracked allocation in the CUDA-graph pool. One state per
    device, so two of these kernels running concurrently on different streams of one device
    would race on it.
    """
    dev = torch.device("cuda", device_idx)
    ticket = torch.zeros((1,), dtype=torch.int32, device=dev)
    acc = torch.zeros((2 * MAX_GROUPS,), dtype=torch.float32, device=dev)
    return ticket, acc


@cute.jit
def _rht128_amax_finalize(
    acc_t, ticket_t, col_amax_t, row_amax_t, flag_t, barrier, num_tensors, gdim, tidx_e
):
    """Hand the accumulated per-group maxes to the outputs, by the last CTA to finish.

    Run by the 256 epilogue threads of every CTA after their last flush and the barrier
    that follows it: thread 0 takes an arrival ticket with an acq_rel RMW at gpu scope
    (the release publishes the CTA's flushes, ordered before it by that barrier) and in
    the CTA whose ticket is ``gdim - 1`` thread ``s < 2 * num_tensors`` -- ordered after
    thread 0's acquire by the barrier -- moves slot ``s`` (``chain * num_tensors + g``,
    chain 0 = col) to its output and re-zeroes it, and thread 0 resets the ticket, so the
    next launch again accumulates into zeros. Until this point the slots of a launch are
    written only by atomics and read by nobody, and the re-zeroing stores are the
    reading threads' own, so no stale L1 line exists.
    """
    if tidx_e == cutlass.Int32(0):
        ticket = cute.arch.atomic_add(
            ticket_t.iterator, cutlass.Int32(1), sem="acq_rel", scope="gpu"
        )
        flag_t[0] = cutlass.Int32(
            cutlass.select_(
                ticket == gdim - cutlass.Int32(1), cutlass.Int32(1), cutlass.Int32(0)
            )
        )
    barrier.arrive_and_wait()
    if flag_t[0] != cutlass.Int32(0):
        if tidx_e < num_tensors:
            col_amax_t[tidx_e] = acc_t[tidx_e]
            acc_t[tidx_e] = cutlass.Float32(0.0)
        elif tidx_e < cutlass.Int32(2) * num_tensors:
            row_amax_t[tidx_e - num_tensors] = acc_t[tidx_e]
            acc_t[tidx_e] = cutlass.Float32(0.0)
        if tidx_e == cutlass.Int32(0):
            ticket_t[0] = cutlass.Int32(0)


@cute.jit
def _rht128_signed_operand_row(sign_t, b_base, j, lane):
    """Write row ``j`` of a resident K-major SW128 ``R^T`` tile from the live int8 signs.

    ``R^T[j, k] = s_k H128[j, k]`` and every ``|H128|`` entry is ``bfloat16(1 / sqrt(128))``
    with sign ``popcount(j & k) & 1`` (Sylvester order), so the element is
    ``RHT128_ENTRY_BITS`` with bit 15 ``= parity(j & k) ^ (s_k < 0)`` -- the bytes of
    ``torch.mul(h128, s[None, :])``. Both sign words are formed once per thread as 128-bit
    vectors over ``k`` (four 32-bit words): the ``s_k < 0`` bits by four warp ballots, the
    ``parity(j & k)`` bits as the XOR of the alternating masks selected by the set bits of
    ``j`` (bits 0-4 of ``k`` alternate inside a word, bits 5-6 pick the word). Per 16 B chunk
    ``c`` (``k = 8 c .. 8 c + 7``) byte ``c`` of that vector is spread onto bits 15, 31, 47,
    63 of two 64-bit words by one multiply per nibble, and stored at byte
    ``(c // 8) * 16384 + 128 j + 16 ((c % 8) ^ x)`` of the tile, where ``x`` is the SW128
    phase of the row's absolute address (bits 7-9), as the TMA that used to write these
    tiles applied it.
    """
    neg0 = cutlass.Uint32(cute.arch.vote_ballot_sync(sign_t[lane] < cutlass.Int8(0)))
    neg1 = cutlass.Uint32(
        cute.arch.vote_ballot_sync(sign_t[lane + cutlass.Int32(32)] < cutlass.Int8(0))
    )
    neg2 = cutlass.Uint32(
        cute.arch.vote_ballot_sync(sign_t[lane + cutlass.Int32(64)] < cutlass.Int8(0))
    )
    neg3 = cutlass.Uint32(
        cute.arch.vote_ballot_sync(sign_t[lane + cutlass.Int32(96)] < cutlass.Int8(0))
    )
    ju = cutlass.Uint32(j)
    zero = cutlass.Uint32(0)
    one = cutlass.Uint32(1)
    had = (
        ((zero - (ju & one)) & cutlass.Uint32(0xAAAAAAAA))
        ^ ((zero - ((ju >> one) & one)) & cutlass.Uint32(0xCCCCCCCC))
        ^ ((zero - ((ju >> cutlass.Uint32(2)) & one)) & cutlass.Uint32(0xF0F0F0F0))
        ^ ((zero - ((ju >> cutlass.Uint32(3)) & one)) & cutlass.Uint32(0xFF00FF00))
        ^ ((zero - ((ju >> cutlass.Uint32(4)) & one)) & cutlass.Uint32(0xFFFF0000))
    )
    j5 = zero - ((ju >> cutlass.Uint32(5)) & one)
    j6 = zero - ((ju >> cutlass.Uint32(6)) & one)
    signs = (had ^ neg0, had ^ j5 ^ neg1, had ^ j6 ^ neg2, had ^ j5 ^ j6 ^ neg3)
    # nibble bit i -> bit 15 + 16 i: the four partial products never overlap or carry.
    spread = cutlass.Uint64(0x1000200040008000)
    sign_bits = cutlass.Uint64(0x8000800080008000)
    entries = cutlass.Uint64(RHT128_ENTRY_BITS * 0x0001000100010001)
    row = cutlass.Int32(b_base) + j * cutlass.Int32(128)
    x = (row >> cutlass.Int32(7)) & cutlass.Int32(7)
    st2 = cute.make_rmem_tensor((2,), cutlass.Uint64)
    for c in cutlass.range_constexpr(RHT128_DIM // 8):
        sign8 = signs[c // 4] >> cutlass.Uint32(8 * (c % 4))
        lo = cutlass.Uint64(sign8 & cutlass.Uint32(0xF))
        hi = cutlass.Uint64((sign8 >> cutlass.Uint32(4)) & cutlass.Uint32(0xF))
        st2[0] = ((lo * spread) & sign_bits) | entries
        st2[1] = ((hi * spread) & sign_bits) | entries
        off = (
            row
            + cutlass.Int32((c // 8) * 16384)
            + ((cutlass.Int32(c % 8) ^ x) << cutlass.Int32(4))
        )
        cute.make_tensor(
            cute.make_ptr(
                cutlass.Uint64, off, cute.AddressSpace.smem, assumed_align=16
            ),
            cute.make_layout((2,)),
        ).store(st2.load())


class _Tcgen05GroupRowRhtColRhtAmax:
    """Per-group ``max|dy @ R_n|`` and ``max|dy.t() @ R_m|`` in one pass over ``dy``.

    Standalone (no ``_GroupRhtMainloop``): every 128x128 tile is TMA'd once and
    feeds two UMMA chains against two resident K-major ``R^T`` tiles, which the
    col / row epilogue warps write into shared memory from the live sign vectors
    (``_rht128_signed_operand_row``) before their first tile -- the col chain reads
    the stage MN-major (as the RHT-16 kernels do) and the row chain reads the same
    bytes through a K-major view. Static persistent grid:
    CTA ``b`` owns the contiguous tile chunk ``_static_tile_range``, tiles below
    ``logical_packed_length`` only, hidden-fastest. The epilogues atomic-max into a
    persistent zeroed accumulator and the last CTA to finish moves it into the
    outputs (``_rht128_amax_finalize``), so the op fills nothing on the host.
    """

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # dy.t().unsqueeze(-1): (hidden, tokens, 1)
        mSignRow: cute.Tensor,  # dgrad_rht (128,) int8: R_n, row chain
        mSignCol: cute.Tensor,  # wgrad_rht (128,) int8: R_m, col chain
        amax_rht_dy_t_: cute.Tensor,  # (num_tensors,) f32, written by the last CTA: row chain
        amax_rht_dy_t_t: cute.Tensor,  # (num_tensors,) f32, written by the last CTA: col chain
        offsets_t: cute.Tensor,
        logical_len_t: cute.Tensor,
        acc_t: cute.Tensor,  # (>= 2 * num_tensors,) f32 accumulator, zero between launches
        ticket_t: cute.Tensor,  # (1,) int32 arrival counter, zero between launches
        hidden: cutlass.Int32,
        num_tensors: cutlass.Int32,
        num_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        k_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        mn_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.MN_SW128, cutlass.BFloat16
        )
        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        mma_col = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            RHT128_MMA_TILER_COL,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.MN,
            OperandMajorMode.K,
        )
        tiled_mma_col = cute.make_tiled_mma(cute.make_mma_atom(mma_col))
        mma_row = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            RHT128_MMA_TILER_ROW,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
        tiled_mma_row = cute.make_tiled_mma(cute.make_mma_atom(mma_row))
        a_shape = tiled_mma_col.partition_shape_A(
            cute.dice(RHT128_CTA_TILE_COL, (1, None, 1))
        )
        a_smem_layout_staged = tcgen05.tile_to_mma_shape(
            mn_atom, cute.append(a_shape, RHT128_MAINLOOP_STAGES), order=(1, 2, 3)
        )
        ar_shape = tiled_mma_row.partition_shape_A(
            cute.dice(RHT128_CTA_TILE_ROW, (1, None, 1))
        )
        a_row_layout_staged = tcgen05.tile_to_mma_shape(
            k_atom, cute.append(ar_shape, RHT128_MAINLOOP_STAGES), order=(2, 1, 3)
        )
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            g2s,
            mA,
            cute.slice_(a_smem_layout_staged, (None, None, None, 0)),
            RHT128_CTA_TILE_COL,
            tiled_mma_col,
            (1, 1, 1, 1),
        )
        b_shape = tiled_mma_col.partition_shape_B(
            cute.dice(RHT128_CTA_TILE_COL, (None, 1, 1))
        )
        b_smem_layout = tcgen05.tile_to_mma_shape(
            k_atom, cute.append(b_shape, 1), order=(1, 2, 3)
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma_col.thr_id.shape,)
        )
        tCtAcc_fake = tiled_mma_row.make_fragment_C(
            cute.append(
                tiled_mma_row.partition_shape_C((TOKEN_TILE, RHT128_DIM)),
                2 * RHT128_ACC_STAGES,
            )
        )
        num_tmem_alloc_cols = sm100_utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        self.kernel(
            tiled_mma_col,
            tiled_mma_row,
            tma_atom_a,
            tma_tensor_a,
            mSignRow,
            mSignCol,
            amax_rht_dy_t_,
            amax_rht_dy_t_t,
            offsets_t,
            logical_len_t,
            acc_t,
            ticket_t,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            a_row_layout_staged,
            b_smem_layout,
            tCtAcc_fake.layout,
            num_tmem_alloc_cols,
            hidden,
            num_tensors,
        ).launch(grid=(num_ctas, 1, 1), block=(RHT128_TPB, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        tiled_mma_col: cute.TiledMma,
        tiled_mma_row: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        mSignRow: cute.Tensor,
        mSignCol: cute.Tensor,
        row_amax_t: cute.Tensor,
        col_amax_t: cute.Tensor,
        offsets_t: cute.Tensor,
        logical_len_t: cute.Tensor,
        acc_t: cute.Tensor,
        ticket_t: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        a_row_layout_staged: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        acc_fake_layout: cute.Layout,
        num_tmem_alloc_cols: cutlass.Constexpr,
        hidden: cutlass.Int32,
        num_tensors: cutlass.Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % cutlass.Int32(32)
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        tiles_in_m = hidden // cutlass.Int32(M_TILE)
        tiles_in_n_valid = logical_len_t[0] // cutlass.Int32(TOKEN_TILE)
        t_begin, t_end = _static_tile_range(tiles_in_m * tiles_in_n_valid, bidx, gdim)
        n_my = t_end - t_begin

        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, RHT128_MAINLOOP_STAGES * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, RHT128_ACC_STAGES * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            last_cta: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        last_cta_t = cute.make_tensor(storage.last_cta.ptr, cute.make_layout((1,)))
        # Accumulator slots (chain, group), chain 0 = col.
        acc_row_t = cute.make_tensor(
            acc_t.iterator + num_tensors, cute.make_layout((num_tensors,))
        )

        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar.data_ptr(),
            num_stages=RHT128_MAINLOOP_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=_A_TILE_BYTES,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        ab_producer, ab_consumer = ab_pipeline.make_participants()
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=RHT128_ACC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, RHT128_ACC_CONSUMER_WARPS
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # MMA warp + both epilogue groups retrieve the TMEM pointer.
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_ALLOC_BAR, num_threads=32 + COL_THREADS + COL_THREADS
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_DEALLOC_BAR, num_threads=COL_THREADS + COL_THREADS
        )
        # Both epilogue groups write an R^T tile; the MMA warp waits for both.
        b_ready_barrier = pipeline.NamedBarrier(
            barrier_id=RHT128_B_READY_BAR, num_threads=32 + COL_THREADS + COL_THREADS
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=COL_WARP_BEGIN,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        raw_a = smem.allocate_array(
            cutlass.BFloat16,
            cute.cosize(a_smem_layout_staged.outer),
            byte_alignment=128,
        )
        sA = cute.make_tensor(
            cute.recast_ptr(raw_a, a_smem_layout_staged.inner, dtype=cutlass.BFloat16),
            a_smem_layout_staged.outer,
        )
        sA_row = cute.make_tensor(
            cute.recast_ptr(raw_a, a_row_layout_staged.inner, dtype=cutlass.BFloat16),
            a_row_layout_staged.outer,
        )
        sBcol = smem.allocate_tensor(
            cutlass.BFloat16, b_smem_layout.outer, 128, swizzle=b_smem_layout.inner
        )
        sBrow = smem.allocate_tensor(
            cutlass.BFloat16, b_smem_layout.outer, 128, swizzle=b_smem_layout.inner
        )

        cta_layout = cute.make_layout((1,))
        thr_col = tiled_mma_col.get_slice(0)
        gA = cute.local_tile(
            mA, cute.slice_(RHT128_CTA_TILE_COL, (None, 0, None)), (None, None, None)
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_col.partition_A(gA), 0, 3),
        )
        tCrA_col = tiled_mma_col.make_fragment_A(sA)
        tCrB_col = tiled_mma_col.make_fragment_B(sBcol)
        tCrA_row = tiled_mma_row.make_fragment_A(sA_row)
        tCrB_row = tiled_mma_row.make_fragment_B(sBrow)

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ==================== TMA warp ====================
        if warp_idx == TMA_WARP:
            for i in cutlass.range(n_my, unroll=1):
                t = t_begin + i
                tile_n = t // tiles_in_m
                tile_m = t - tile_n * tiles_in_m
                handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, tile_m, tile_n, 0)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
            ab_producer.tail()

        # ==================== MMA warp ====================
        if warp_idx == MMA_WARP:
            tmem.wait_for_alloc()
            tCtAcc = cute.make_tensor(
                tmem.retrieve_ptr(cutlass.Float32), acc_fake_layout
            )
            b_ready_barrier.arrive_and_wait()
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, RHT128_ACC_STAGES
            )
            for i in cutlass.range(n_my, unroll=1):
                ab_handle = ab_consumer.wait_and_advance()
                acc_pipeline.producer_acquire(acc_state)
                acc_col = tCtAcc[(None, None, None, 2 * acc_state.index + 0)]
                acc_row = tCtAcc[(None, None, None, 2 * acc_state.index + 1)]
                for kb in cutlass.range_constexpr(RHT128_K_BLOCKS):
                    tiled_mma_col.set(tcgen05.Field.ACCUMULATE, kb > 0)
                    cute.gemm(
                        tiled_mma_col,
                        acc_col,
                        tCrA_col[(None, None, kb, ab_handle.index)],
                        tCrB_col[(None, None, kb, 0)],
                        acc_col,
                    )
                    tiled_mma_row.set(tcgen05.Field.ACCUMULATE, kb > 0)
                    cute.gemm(
                        tiled_mma_row,
                        acc_row,
                        tCrA_row[(None, None, kb, ab_handle.index)],
                        tCrB_row[(None, None, kb, 0)],
                        acc_row,
                    )
                # One elected tcgen05.commit covers both chains' 16 UMMAs.
                acc_pipeline.producer_commit(acc_state)
                acc_state.advance()
                ab_handle.release()
            acc_pipeline.producer_tail(acc_state)

        # ==================== col epilogue: max|dy.t() @ R_m| ====================
        if warp_idx >= COL_WARP_BEGIN and warp_idx < COL_WARP_END:
            _rht128_signed_operand_row(
                mSignCol, sBcol.iterator.toint(), tidx % cutlass.Int32(RHT128_DIM), lane
            )
            # Generic-proxy stores -> visible to the UMMA (async proxy) before the arrive.
            cute.arch.fence_proxy("async.shared", space="cta")
            b_ready_barrier.arrive()
            tmem.allocate(num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc = cute.make_tensor(tmem_ptr, acc_fake_layout)
            _rht128_amax_epilogue(
                0,
                acc_t,
                tCtAcc,
                acc_pipeline,
                offsets_t,
                num_tensors,
                t_begin,
                t_end,
                tiles_in_m,
                tidx,
                lane,
            )
            tmem_dealloc_barrier.arrive_and_wait()
            _rht128_amax_finalize(
                acc_t,
                ticket_t,
                col_amax_t,
                row_amax_t,
                last_cta_t,
                tmem_dealloc_barrier,
                num_tensors,
                gdim,
                tidx - cutlass.Int32(COL_THREADS),
            )
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ==================== row epilogue: max|dy @ R_n| ====================
        if warp_idx >= ROW_WARP_BEGIN and warp_idx < RHT128_ROW_WARP_END:
            _rht128_signed_operand_row(
                mSignRow, sBrow.iterator.toint(), tidx % cutlass.Int32(RHT128_DIM), lane
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            b_ready_barrier.arrive()
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc = cute.make_tensor(tmem_ptr, acc_fake_layout)
            _rht128_amax_epilogue(
                1,
                acc_row_t,
                tCtAcc,
                acc_pipeline,
                offsets_t,
                num_tensors,
                t_begin,
                t_end,
                tiles_in_m,
                tidx,
                lane,
            )
            tmem_dealloc_barrier.arrive_and_wait()
            _rht128_amax_finalize(
                acc_t,
                ticket_t,
                col_amax_t,
                row_amax_t,
                last_cta_t,
                tmem_dealloc_barrier,
                num_tensors,
                gdim,
                tidx - cutlass.Int32(COL_THREADS),
            )


@functools.lru_cache(maxsize=None)
def _compile_group_row_rht_col_rht_amax_kernel(device_idx: int):
    """Compile the grouped row-RHT + col-RHT amax kernel with symbolic shapes."""
    free = cute.sym_int
    h_sym = cute.sym_int(divisibility=M_TILE)
    t_sym = cute.sym_int(divisibility=TOKEN_TILE)
    k = _Tcgen05GroupRowRhtColRhtAmax()
    return cute.compile(
        k,
        make_fake_tensor(cutlass.BFloat16, (h_sym, t_sym, 1), stride=(1, free(), 1)),
        make_fake_tensor(cutlass.Int8, (RHT128_DIM,), stride=(1,)),
        make_fake_tensor(cutlass.Int8, (RHT128_DIM,), stride=(1,)),
        make_fake_tensor(cutlass.Float32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Float32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Int32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Int32, (1,), stride=(1,)),
        make_fake_tensor(cutlass.Float32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Int32, (1,), stride=(1,)),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _cutedsl_group_row_rht_col_rht_amax_impl(
    dy: torch.Tensor,
    dgrad_rht: torch.Tensor,
    wgrad_rht: torch.Tensor,
    offsets: torch.Tensor,
    num_tensors: int,
    logical_packed_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-group ``max|dy @ R_n|`` and ``max|dy.t() @ R_m|``.

    ``dy`` is ``(tokens, hidden)`` bfloat16 row-major; ``dgrad_rht`` / ``wgrad_rht``
    are the live ``(128,)`` int8 sign vectors of ``R_n`` / ``R_m``, from which the
    kernel writes both ``R^T`` tiles into shared memory. Returns
    ``(amax_rht_dy, amax_rht_dy_t)``, each ``(num_tensors,)`` float32, rowwise
    first like the Triton twin, written whole by the kernel's last CTA.
    """
    tokens, hidden = dy.shape
    dev = dy.device
    dy = dy.detach()

    amax_rht_dy = torch.empty((num_tensors,), dtype=torch.float32, device=dev)
    amax_rht_dy_t = torch.empty((num_tensors,), dtype=torch.float32, device=dev)
    ticket, acc = _get_group_amax_reduce_buffers(dev.index)
    if logical_packed_length is None:
        logical_packed_length = offsets[-1:]
    # See the fused kernel: the entry point requires byte_offset==0.
    logical_packed_length = logical_packed_length.clone()
    tiles = (hidden // M_TILE) * (tokens // TOKEN_TILE)
    num_ctas = max(1, min(tiles, _get_num_sms(dev.index)))

    stream = cuda.CUstream(int(torch.cuda.current_stream(dev).cuda_stream))
    _compile_group_row_rht_col_rht_amax_kernel(dev.index)(
        dy.t().unsqueeze(-1),
        dgrad_rht,
        wgrad_rht,
        amax_rht_dy,
        amax_rht_dy_t,
        offsets,
        logical_packed_length,
        acc,
        ticket,
        int(hidden),
        int(num_tensors),
        int(num_ctas),
        stream,
    )
    return amax_rht_dy, amax_rht_dy_t


# --- grouped rowwise-RHT + columnwise-RHT MS-EDEN quantize (RHT-128 on both axes) ---
# The mainloop is ``_Tcgen05GroupRowRhtColRhtAmax``'s; only the epilogues and their warp
# layout differ. Warps: 0 MMA, 1 TMA, 2-3 idle, 4-11 col, 12-19 row -- eight epilogue warps
# per chain, two per TMEM quadrant, splitting a lane's 8 blocks into 0-3 and 4-7. The
# per-block chain is ~200 dependent instructions fed by an in-place tcgen05.ld, so four
# epilogue warps per SM scheduler (not the amax twin's two) are what hides its latency, and
# a 4-block body per warp is what keeps each scheduler's two bodies inside its instruction
# cache.
RHT128_MSEDEN_COL_WARP_BEGIN = 4
RHT128_MSEDEN_COL_WARP_END = 12
RHT128_MSEDEN_ROW_WARP_BEGIN = 12
RHT128_MSEDEN_ROW_WARP_END = 20
RHT128_MSEDEN_N_WARPS = 20
RHT128_MSEDEN_TPB = 32 * RHT128_MSEDEN_N_WARPS
RHT128_MSEDEN_EPI_THREADS = 32 * (
    RHT128_MSEDEN_COL_WARP_END - RHT128_MSEDEN_COL_WARP_BEGIN
)  # per chain
RHT128_MSEDEN_ACC_CONSUMER_WARPS = (
    RHT128_MSEDEN_COL_WARP_END - RHT128_MSEDEN_COL_WARP_BEGIN
) + (RHT128_MSEDEN_ROW_WARP_END - RHT128_MSEDEN_ROW_WARP_BEGIN)
RHT128_MSEDEN_BLOCKS_PER_WARP = (RHT128_DIM // 16) // 2  # 4 of a lane's 8 blocks
RHT128_MSEDEN_SIGN_BAR = 3  # both chains' sign bytes staged in shared memory


def _dot16_tree_rn(a, b):
    """sum a_i * b_i over 16 values in Triton's order (read off the op's PTX):
    scalar RN products, pairs added lane-wise as ((p0+p2)+(p4+p6))+((p8+p10)+(p12+p14))
    and its odd twin in the two f32x2 lanes, then even + odd. Every op rounds to
    nearest and none is fused, so the bits equal Triton's tl.sum; a fused multiply-add
    anywhere here would move the last ulp of the ratio and flip stochastic roundings."""
    p = [
        cute.arch.mul_packed_f32x2((a[2 * j], a[2 * j + 1]), (b[2 * j], b[2 * j + 1]))
        for j in range(8)
    ]
    s = [cute.arch.add_packed_f32x2(p[2 * j], p[2 * j + 1]) for j in range(4)]
    t0 = cute.arch.add_packed_f32x2(s[0], s[1])
    t1 = cute.arch.add_packed_f32x2(s[2], s[3])
    even, odd = cute.arch.add_packed_f32x2(t0, t1)
    return even + odd


def _ms_eden_enc_from_amax(amax, enc_over_fp4max, dec, cap):
    """``_enc_from_amax`` at the ceiling ``cap`` (the MS-EDEN one or ``FP8_E4M3_MAX``),
    returning the stored E4M3 scale as its byte and widened to f32 (the MS-EDEN correction
    multiplies that value, not the byte). ``rcp.rn`` is the same correctly rounded
    reciprocal as ``div.rn`` without the division's slow-path fixups."""
    pvscale = _min_f32(amax * enc_over_fp4max, cap)
    pv_f32 = cute.make_rmem_tensor((4,), cutlass.Float32)
    for i in range(4):
        pv_f32[i] = pvscale
    pv_f8 = cute.make_rmem_tensor((4,), cutlass.Float8E4M3FN)
    pv_f8.store(pv_f32.load().to(cutlass.Float8E4M3FN))
    pv_back = cute.make_rmem_tensor((4,), cutlass.Float32)
    pv_back.store(pv_f8.load().to(cutlass.Float32))
    sf8 = pv_back[0]
    enc = _min_f32(_rcp_rn_f32(sf8 * dec), cutlass.Float32(FP32_MAX))
    return enc, pv_f8[0], sf8


def _ms_eden_block16_corrected(vals, enc_over_fp4max, dec):
    """One 1x16 MS-EDEN block from 16 raw f32 accumulator values -> (w0, w1, corrected
    f32 scale): everything of ``_ms_eden_block16`` ahead of the stochastic rounding, so
    both roundings (software word, hardware ``cvt.rs``) consume the same value."""
    # The bf16-exact values serve both consumers of the rounding: their max is the block
    # amax (what ``_round_rht_amax`` of the raw amax gives: RTNE is monotonic in magnitude,
    # so the max of the rounded values is the rounded max) and they are what the encode
    # multiplier scales.
    zero = cutlass.Float32(0.0)
    e = _bf16round_f32x8(
        vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], zero
    ) + _bf16round_f32x8(
        vals[8],
        vals[9],
        vals[10],
        vals[11],
        vals[12],
        vals[13],
        vals[14],
        vals[15],
        zero,
    )
    enc, _, sf8 = _ms_eden_enc_from_amax(
        _abs_amax16(e), enc_over_fp4max, dec, cutlass.Float32(EDEN_BLOCK_SCALE_MAX)
    )
    v = _mul_clamp_f32x8(*e[0:8], enc) + _mul_clamp_f32x8(*e[8:16], enc)
    w0 = _cvt_rn_e2m1x8_f32(*v[0:8])
    w1 = _cvt_rn_e2m1x8_f32(*v[8:16])
    q = _cvt_e2m1x8_to_f32(w0) + _cvt_e2m1x8_to_f32(w1)
    dot_sq = _dot16_tree_rn(v, v)
    dot_cross = _dot16_tree_rn(v, q)
    ratio = _div_full_f32(dot_sq, dot_cross)
    # False for inf and NaN, as Triton's ``< inf``; a zero ``dot_cross`` makes the ratio
    # inf or NaN, so Triton's ``!= 0`` guard is implied.
    finite = _abs_f32(ratio) <= cutlass.Float32(FP32_MAX)
    corr = cutlass.Float32(cutlass.select_(finite, ratio, cutlass.Float32(1.0)))
    # ONE RN multiply of the widened E4M3 scale (no clamp), as Triton's
    # ``block_scale * correction``.
    return w0, w1, sf8 * corr


def _ms_eden_block16(vals, enc_over_fp4max, dec, rbits):
    """One 1x16 MS-EDEN block from 16 raw f32 accumulator values -> (w0, w1, E4M3 byte).

    RTNE codes against the pre-correction E4M3 scale (cap 256), then the stochastically
    rounded ``sf8 * <v, v> / <v, q>`` with ``rbits`` the block's Triton Philox word. The
    correction reads back the codes it just packed, so it measures that exact rounding.
    """
    w0, w1, corrected = _ms_eden_block16_corrected(vals, enc_over_fp4max, dec)
    return w0, w1, _sr_e4m3_byte(corrected, rbits)


@cute.jit
def _rht128_tile_ms_eden(
    acc,
    tidx,
    enc_over_fp4max,
    dec,
    state,
    idx_base,
    u_base,
    row_addr,
    rSF,
    fast_path: cutlass.Constexpr,
    word,
):
    """MS-EDEN quantize this warp's eighth of one 128x128 f32 TMEM accumulator: this
    thread's lane = one output row, blocks ``u_base .. u_base + 3`` of its 8.

    The codes go straight to global as two 16-byte stores into the lane's 16 code words
    at ``row_addr``; the four scale bytes land in ``rSF`` for the chain's own scatter.
    ``u_base`` is warp-uniform but dynamic, so both warps of a quadrant run one body.

    ``FAST_PATH``: ``idx_base`` is the counter of the row's 16-scale group these blocks
    belong to and ``word`` (0-3, warp-uniform) which of its four Philox words rounds them;
    the four corrected scales go through one ``cvt.rs.satfinite.e4m3x4.f32`` into the
    word ``rSF`` holds.
    """
    copy_atom_t2r = sm100_utils.get_tmem_load_op(
        RHT128_CTA_TILE_COL,
        utils.LayoutEnum.ROW_MAJOR,
        cutlass.Float32,
        cutlass.Float32,
        RHT128_EPI_TILE,
        False,
    )
    tAcc = transform_partitioned_tensor_layout(acc)
    tAcc_epi = cute.flat_divide(tAcc, RHT128_EPI_TILE)
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
    thr_copy = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc = thr_copy.partition_S(tAcc_epi)
    tTR_rAcc = cute.make_rmem_tensor(((16, 1), 1, 1), cutlass.Float32)
    rCodes = cute.make_rmem_tensor((2 * RHT128_MSEDEN_BLOCKS_PER_WARP,), cutlass.Uint32)
    if cutlass.const_expr(fast_path):
        rCorr = cute.make_rmem_tensor((RHT128_MSEDEN_BLOCKS_PER_WARP,), cutlass.Float32)
    for j in cutlass.range_constexpr(RHT128_MSEDEN_BLOCKS_PER_WARP):
        u = u_base + cutlass.Int32(j)
        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, u)], tTR_rAcc)
        vals = tTR_rAcc.load().reshape((16,))
        if cutlass.const_expr(fast_path):
            w0, w1, corrected = _ms_eden_block16_corrected(vals, enc_over_fp4max, dec)
        else:
            # Triton's tl.randint word for this block: Philox4x32-10 at counter
            # (offset_base, linear_idx, 0, 0), word 0.
            rbits = philox_word0(state, cutlass.Uint32(idx_base + u))
            w0, w1, sf = _ms_eden_block16(vals, enc_over_fp4max, dec, rbits)
        rCodes[2 * j] = w0
        rCodes[2 * j + 1] = w1
        if cutlass.const_expr(fast_path):
            rCorr[j] = corrected
        else:
            rSF[j] = cutlass.Uint8(sf)
    if cutlass.const_expr(fast_path):
        # Triton's tl.randint4x for the row's 16-scale group: Philox4x32-10 at counter
        # (offset_base, linear_group16, 0, 0), all four words; this warp's four scales
        # take word ``word`` of it, one 32-bit draw for one hardware cvt.rs of four.
        r0, r1, r2, r3 = philox4_all(state, cutlass.Uint32(idx_base))
        odd = (word & cutlass.Int32(1)) != cutlass.Int32(0)
        lo = cutlass.Uint32(cutlass.select_(odd, r1, r0))
        hi = cutlass.Uint32(cutlass.select_(odd, r3, r2))
        rbits = cutlass.Uint32(cutlass.select_(word >= cutlass.Int32(2), hi, lo))
        cute.recast_tensor(rSF, cutlass.Uint32)[0] = _cvt_rs_e4m3x4_f32(
            rCorr[0], rCorr[1], rCorr[2], rCorr[3], rbits
        )
    # 16-byte aligned by construction: the code row starts on a 64-byte boundary (the u32
    # pitch and the tile column are multiples of 16 words, the buffer is torch.empty's) and
    # ``u_base * 8`` is 0 or 32.
    code_addr = row_addr + cutlass.Int64(u_base) * cutlass.Int64(8)
    _st_global_v4_u32(code_addr, rCodes[0], rCodes[1], rCodes[2], rCodes[3])
    _st_global_v4_u32(
        code_addr + cutlass.Int64(16), rCodes[4], rCodes[5], rCodes[6], rCodes[7]
    )


def _store_grouped_col_sf_word(mSF_u32, rSF, r, c_base, g, offsets_t, hidden):
    """``_store_grouped_col_sf_u32`` for the four scale bytes one MS-EDEN warp holds: the
    u32 word of the group-local swizzled tile that holds columns ``c_base .. c_base + 3``
    (``c_base % 4 == 0``)."""
    prev = cutlass.select_(g > cutlass.Int32(0), g - cutlass.Int32(1), 0)
    group_start = cutlass.select_(
        g > cutlass.Int32(0), offsets_t[prev], cutlass.Int32(0)
    )
    group_len = offsets_t[g] - group_start
    r_blk = r // cutlass.Int32(128)
    r_lane = r % cutlass.Int32(32)
    r_grp = (r % cutlass.Int32(128)) // cutlass.Int32(32)
    prefix_words = cutlass.Int32(
        cutlass.Int64(hidden) * cutlass.Int64(group_start) // cutlass.Int64(64)
    )
    words_per_hidden_block = group_len * cutlass.Int32(2)
    c_local = c_base - group_start // cutlass.Int32(16)
    mSF_u32[
        prefix_words
        + r_blk * words_per_hidden_block
        + (c_local // cutlass.Int32(4)) * cutlass.Int32(128)
        + r_lane * cutlass.Int32(4)
        + r_grp
    ] = cute.recast_tensor(rSF, cutlass.Uint32)[0]


def _ms_eden_col_sf_base(g, g_end, offsets_t, hidden, u_base, lane_word):
    """Group-invariant part of ``_store_grouped_col_sf_word`` for this warp's word: the
    index of its tile ``(0, 0)`` word and the pitch of one hidden block, so the tile
    ``(tile_m, tile_n)`` word sits at ``base + tile_m * pitch + tile_n * 256``. Groups
    are 128-row aligned, so ``group_start // 64`` is exact and the per-tile address
    needs no division."""
    prev = cutlass.select_(g > cutlass.Int32(0), g - cutlass.Int32(1), 0)
    group_start = cutlass.select_(
        g > cutlass.Int32(0), offsets_t[prev], cutlass.Int32(0)
    )
    prefix_words = cutlass.Int32(
        cutlass.Int64(hidden) * cutlass.Int64(group_start) // cutlass.Int64(64)
    )
    base = (
        prefix_words
        - (group_start // cutlass.Int32(64)) * cutlass.Int32(128)
        + u_base * cutlass.Int32(32)
        + lane_word
    )
    return base, (g_end - group_start) * cutlass.Int32(2)


@cute.jit
def _rht128_ms_eden_epilogue(
    chain: cutlass.Constexpr,
    fast_path: cutlass.Constexpr,
    mFP4,
    mSF,
    amax_t,
    sr_rng_t,
    tCtAcc,
    acc_pipeline,
    offsets_t,
    num_tensors,
    t_begin,
    t_end,
    tiles_in_m,
    hidden,
    tokens,
    tidx,
):
    """One epilogue body for both chains: consume the acc ring over this CTA's chunk.

    Tiles are enumerated hidden-fastest, so ``token`` is non-decreasing and the group
    cache is ``_rht128_amax_epilogue``'s; a crossing reloads the group's two-level
    scale instead of flushing. Chain 0 quantizes ``dy.t() @ R_m`` (lane = hidden row,
    blocks along tokens, per-group swizzled scales); chain 1 quantizes ``dy @ R_n``
    (lane = token, blocks along hidden). The chain's second warp group takes blocks
    4-7 of every lane. Tile coordinates advance incrementally, and each warp's four
    scale bytes of a lane go out as the one u32 word they form in the 128x4 swizzle
    atom (``lane_word``: the lane's word within the atom).

    ``FAST_PATH`` counts one Philox counter per 16 scales of a row (``randint4x``):
    ``outer * ceil(INNER_SF / 16) + inner_tile // 2`` -- a ceil pitch, since an odd
    inner tile count leaves the row's last group one tile (8 scales) wide.
    """
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, RHT128_ACC_STAGES
    )
    r_local = tidx % cutlass.Int32(128)
    # Warps 4-7 / 12-15 take blocks 0-3 of their quadrant's lanes, warps 8-11 / 16-19
    # blocks 4-7.
    half = (
        (tidx - cutlass.Int32(32 * RHT128_MSEDEN_COL_WARP_BEGIN)) // cutlass.Int32(128)
    ) % cutlass.Int32(2)
    u_base = half * cutlass.Int32(RHT128_MSEDEN_BLOCKS_PER_WARP)
    lane = r_local % cutlass.Int32(32)
    lane_word = lane * cutlass.Int32(4) + r_local // cutlass.Int32(32)
    if cutlass.const_expr(chain == 0):
        state = philox_prep(
            cutlass.Uint32(sr_rng_t[0]),
            cutlass.Uint32(sr_rng_t[1]),
            cutlass.Uint32(sr_rng_t[2]),
        )
        # Triton's INNER // 16 with INNER = M, the packed capacity.
        inner_blocks = tokens // cutlass.Int32(16)
        inner_groups = (
            tokens // cutlass.Int32(128) + cutlass.Int32(1)
        ) // cutlass.Int32(2)
    else:
        state = philox_prep(
            cutlass.Uint32(sr_rng_t[4]),
            cutlass.Uint32(sr_rng_t[5]),
            cutlass.Uint32(sr_rng_t[6]),
        )
        inner_blocks = hidden // cutlass.Int32(16)
        inner_groups = (
            hidden // cutlass.Int32(128) + cutlass.Int32(1)
        ) // cutlass.Int32(2)
    tile_n = t_begin // tiles_in_m
    tile_m = t_begin - tile_n * tiles_in_m
    g = _group_idx(tile_n * cutlass.Int32(TOKEN_TILE), offsets_t, num_tensors)
    g_end = offsets_t[g]
    _, dec, enc_over_fp4max = _global_scale(amax_t[g], EDEN_BLOCK_SCALE_MAX)
    if cutlass.const_expr(chain == 0):
        sf_base, sf_pitch = _ms_eden_col_sf_base(
            g, g_end, offsets_t, hidden, u_base, lane_word
        )
    rSF = cute.make_rmem_tensor((RHT128_MSEDEN_BLOCKS_PER_WARP,), cutlass.Uint8)
    for i in cutlass.range(t_end - t_begin, unroll=1):
        token = tile_n * cutlass.Int32(TOKEN_TILE)
        if token >= g_end:
            g = _group_idx(token, offsets_t, num_tensors)
            g_end = offsets_t[g]
            _, dec, enc_over_fp4max = _global_scale(amax_t[g], EDEN_BLOCK_SCALE_MAX)
            if cutlass.const_expr(chain == 0):
                sf_base, sf_pitch = _ms_eden_col_sf_base(
                    g, g_end, offsets_t, hidden, u_base, lane_word
                )
        if cutlass.const_expr(chain == 0):
            outer = tile_m * cutlass.Int32(M_TILE) + r_local  # hidden row
            inner_tile = tile_n
            gOut = cute.local_tile(mFP4, (M_TILE, TOKEN_TILE // 8), (tile_m, tile_n))
        else:
            outer = token + r_local  # packed token row
            inner_tile = tile_m
            gOut = cute.local_tile(mFP4, (TOKEN_TILE, M_TILE // 8), (tile_n, tile_m))
        # linear_idx of block 0: the flat index of the scale in the plain
        # (outer, inner // 16) layout, Triton's counter.
        idx_base = outer * inner_blocks + inner_tile * cutlass.Int32(RHT128_DIM // 16)
        word = cutlass.Int32(0)
        if cutlass.const_expr(fast_path):
            # linear_group16 of the row's 16-scale group these two inner tiles form, and
            # this warp's word of the four it yields: tile parity, then half.
            idx_base = outer * inner_groups + inner_tile // cutlass.Int32(2)
            word = (inner_tile % cutlass.Int32(2)) * cutlass.Int32(2) + half
        acc_pipeline.consumer_wait(acc_state)
        _rht128_tile_ms_eden(
            tCtAcc[(None, None, None, 2 * acc_state.index + chain)],
            tidx,
            enc_over_fp4max,
            dec,
            state,
            idx_base,
            u_base,
            gOut[(r_local, None)].iterator.toint(),
            rSF,
            fast_path,
            word,
        )
        cute.arch.fence_view_async_tmem_load()
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_state)
        acc_state.advance()
        sf_word = cute.recast_tensor(rSF, cutlass.Uint32)[0]
        if cutlass.const_expr(chain == 0):
            mSF[sf_base + tile_m * sf_pitch + tile_n * cutlass.Int32(256)] = sf_word
        else:
            # ``_store_sf_byte``'s (r // 128, c // 4, r % 32, (r % 128 // 32) * 4 + c % 4)
            # for the four consecutive columns ``c_base .. c_base + 3``, as one word.
            mSF[
                (
                    tile_n,
                    tile_m * cutlass.Int32(2) + half,
                    lane,
                    r_local // cutlass.Int32(32),
                )
            ] = sf_word
        tile_m = tile_m + cutlass.Int32(1)
        wrap = tile_m == tiles_in_m
        tile_m = cutlass.Int32(cutlass.select_(wrap, cutlass.Int32(0), tile_m))
        tile_n = tile_n + cutlass.Int32(
            cutlass.select_(wrap, cutlass.Int32(1), cutlass.Int32(0))
        )


def _rht128_build_signed_rht(mH, sign_t, sign_base, b_base, p, sign_barrier):
    """Thread ``p`` of 256 builds eight 16 B chunks of a resident K-major ``R^T`` tile:
    chunk ``c = p % 8`` of contracted half ``(p // 8) % 2`` in rows ``j = p // 16 + 16 i``
    of ``H128``, with bit 15 of every bf16 whose contracted index ``k`` has ``s_k < 0``
    flipped -- the exact bf16 sign flip ``torch.mul(h128, signs[None, :])`` performs, so
    the UMMA consumes the same bytes. Consecutive lanes read consecutive chunks of an
    ``H128`` row, so a warp's load covers four 128 B lines instead of thirty-two.

    A chunk of the 128 B row at ``b_base + 16384 * half + 128 * j`` lands at
    ``16 * (c ^ ((row >> 7) & 7))``: the SW128 swizzle is a function of the shared-memory
    address, as the UMMA descriptor applies it. The sign bytes are staged at ``sign_base``:
    thread ``p`` copies byte ``p % 128`` (one byte load per lane, so the vectors need no
    alignment in global memory and the two hot sign lines cost one L2 request per warp),
    the chain barrier publishes them, and the chunk's eight bytes come back as two aligned
    words; a byte is negative iff its bit 7 is set. The ``H128`` chunks are loaded before
    the barrier so the two round trips overlap. Plain function, traced inline."""
    c = p % cutlass.Int32(8)
    half = (p // cutlass.Int32(8)) % cutlass.Int32(2)
    j0 = p // cutlass.Int32(16)
    k = p % cutlass.Int32(RHT128_DIM)
    s_k = sign_t[k]
    h_addr = mH.iterator.toint() + cutlass.Int64(
        j0 * cutlass.Int32(2 * RHT128_DIM) + (p % cutlass.Int32(16)) * cutlass.Int32(16)
    )
    h4 = [cute.make_rmem_tensor((4,), cutlass.Uint32) for _ in range(8)]
    for i in range(8):
        h4[i].store(
            cute.make_tensor(
                cute.make_ptr(
                    cutlass.Uint32,
                    h_addr + cutlass.Int64(i * 16 * 2 * RHT128_DIM),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout((4,)),
            ).load()
        )
    cute.make_tensor(
        cute.make_ptr(
            cutlass.Int8, sign_base + k, cute.AddressSpace.smem, assumed_align=1
        ),
        cute.make_layout((1,)),
    )[0] = s_k
    sign_barrier.arrive_and_wait()
    signs = cute.make_tensor(
        cute.make_ptr(
            cutlass.Uint32,
            sign_base + half * cutlass.Int32(64) + c * cutlass.Int32(8),
            cute.AddressSpace.smem,
            assumed_align=8,
        ),
        cute.make_layout((2,)),
    )
    mask = cute.make_rmem_tensor((4,), cutlass.Uint32)
    for i in range(2):
        s = signs[i]
        mask[2 * i] = ((s & cutlass.Uint32(0x80)) << cutlass.Uint32(8)) | (
            (s & cutlass.Uint32(0x8000)) << cutlass.Uint32(16)
        )
        mask[2 * i + 1] = ((s & cutlass.Uint32(0x800000)) >> cutlass.Uint32(8)) | (
            s & cutlass.Uint32(0x80000000)
        )
    row_addr = (
        b_base + half * cutlass.Int32(RHT128_B_BYTES // 2) + j0 * cutlass.Int32(128)
    )
    st4 = cute.make_rmem_tensor((4,), cutlass.Uint32)
    for i in range(8):
        for w in range(4):
            st4[w] = h4[i][w] ^ mask[w]
        row = row_addr + cutlass.Int32(i * 16 * 128)
        x = (row >> cutlass.Int32(7)) & cutlass.Int32(7)
        cute.make_tensor(
            cute.make_ptr(
                cutlass.Uint32,
                row + cutlass.Int32(16) * (c ^ x),
                cute.AddressSpace.smem,
                assumed_align=16,
            ),
            cute.make_layout((4,)),
        ).store(st4.load())


class _Tcgen05GroupRowRhtColRhtQuantizeMsEden:
    """Per-group MS-EDEN quantize of ``dy @ R_n`` and ``dy.t() @ R_m`` in one pass over ``dy``.

    Standalone (no ``_GroupRhtMainloop``): the mainloop is ``_Tcgen05GroupRowRhtColRhtAmax``'s
    -- every 128x128 tile is TMA'd once and feeds two UMMA chains against two resident
    K-major ``R^T`` tiles the epilogue warps build from ``H128`` and the live sign vectors
    (``_rht128_build_signed_rht``); the col chain reads the stage MN-major and the row
    chain reads the same bytes through a K-major view. Each accumulator is quantized from
    TMEM: RTNE FP4 codes against the group's two-level scale (ceiling 256), then the
    corrected, stochastically rounded E4M3 block scale from Triton's Philox stream. Static
    persistent grid: CTA ``b`` owns the contiguous tile chunk ``_static_tile_range``, tiles
    below ``logical_packed_length`` only, hidden-fastest. Twenty warps (``RHT128_MSEDEN_*``):
    eight epilogue warps per chain, launch-bounded to one CTA per SM (at most 96 registers a
    thread, 65536 / 640 at the 8-register granularity, no ``setmaxnreg``).
    ``fast_path`` selects the ``FAST_PATH`` epilogues: one Philox counter per 16 scales of
    a row and one hardware ``cvt.rs.satfinite.e4m3x4.f32`` per warp per tile -- a different
    stochastic stream from the Triton op's, not bitwise with it.
    """

    def __init__(self, fast_path: bool = False):
        self.fast_path = fast_path

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # dy.t().unsqueeze(-1): (hidden, tokens, 1)
        mH: cute.Tensor,  # H128 (128 j, 128 k) bf16, K-major (symmetric), sign-free
        row_sign_t: cute.Tensor,  # dgrad_rht (128,) int8: row chain
        col_sign_t: cute.Tensor,  # wgrad_rht (128,) int8: col chain
        mRowFP4: cute.Tensor,  # (tokens, hidden // 8) u32 rowwise codes
        mRowSF: cute.Tensor,  # (tokens // 128, hidden // 64, 32, 4) u32 view of the e4m3 scales
        mColFP4: cute.Tensor,  # (hidden, tokens // 8) u32 columnwise codes
        mColSF: cute.Tensor,  # flat u32 view of the per-group swizzled e4m3 scales
        row_amax_t: cute.Tensor,  # amax_rht_dy (num_tensors,) f32: row chain
        col_amax_t: cute.Tensor,  # amax_rht_dy_t (num_tensors,) f32: col chain
        sr_rng_t: cute.Tensor,  # (8,) i32: [col_seed lo/hi, col_off lo/hi, row_seed lo/hi, row_off lo/hi]
        offsets_t: cute.Tensor,
        logical_len_t: cute.Tensor,
        hidden: cutlass.Int32,
        tokens: cutlass.Int32,  # the col-chain Philox pitch is tokens // 16
        num_tensors: cutlass.Int32,
        num_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        k_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        mn_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.MN_SW128, cutlass.BFloat16
        )
        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        mma_col = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            RHT128_MMA_TILER_COL,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.MN,
            OperandMajorMode.K,
        )
        tiled_mma_col = cute.make_tiled_mma(cute.make_mma_atom(mma_col))
        mma_row = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            RHT128_MMA_TILER_ROW,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
        tiled_mma_row = cute.make_tiled_mma(cute.make_mma_atom(mma_row))
        a_shape = tiled_mma_col.partition_shape_A(
            cute.dice(RHT128_CTA_TILE_COL, (1, None, 1))
        )
        a_smem_layout_staged = tcgen05.tile_to_mma_shape(
            mn_atom, cute.append(a_shape, RHT128_MAINLOOP_STAGES), order=(1, 2, 3)
        )
        ar_shape = tiled_mma_row.partition_shape_A(
            cute.dice(RHT128_CTA_TILE_ROW, (1, None, 1))
        )
        a_row_layout_staged = tcgen05.tile_to_mma_shape(
            k_atom, cute.append(ar_shape, RHT128_MAINLOOP_STAGES), order=(2, 1, 3)
        )
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            g2s,
            mA,
            cute.slice_(a_smem_layout_staged, (None, None, None, 0)),
            RHT128_CTA_TILE_COL,
            tiled_mma_col,
            (1, 1, 1, 1),
        )
        b_shape = tiled_mma_col.partition_shape_B(
            cute.dice(RHT128_CTA_TILE_COL, (None, 1, 1))
        )
        b_smem_layout = tcgen05.tile_to_mma_shape(
            k_atom, cute.append(b_shape, 1), order=(1, 2, 3)
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma_col.thr_id.shape,)
        )
        tCtAcc_fake = tiled_mma_row.make_fragment_C(
            cute.append(
                tiled_mma_row.partition_shape_C((TOKEN_TILE, RHT128_DIM)),
                2 * RHT128_ACC_STAGES,
            )
        )
        num_tmem_alloc_cols = sm100_utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        self.kernel(
            tiled_mma_col,
            tiled_mma_row,
            tma_atom_a,
            tma_tensor_a,
            mH,
            row_sign_t,
            col_sign_t,
            mRowFP4,
            mRowSF,
            mColFP4,
            mColSF,
            row_amax_t,
            col_amax_t,
            sr_rng_t,
            offsets_t,
            logical_len_t,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            a_row_layout_staged,
            b_smem_layout,
            tCtAcc_fake.layout,
            num_tmem_alloc_cols,
            hidden,
            tokens,
            num_tensors,
        ).launch(
            grid=(num_ctas, 1, 1),
            block=(RHT128_MSEDEN_TPB, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma_col: cute.TiledMma,
        tiled_mma_row: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        mH: cute.Tensor,
        row_sign_t: cute.Tensor,
        col_sign_t: cute.Tensor,
        mRowFP4: cute.Tensor,
        mRowSF: cute.Tensor,
        mColFP4: cute.Tensor,
        mColSF: cute.Tensor,
        row_amax_t: cute.Tensor,
        col_amax_t: cute.Tensor,
        sr_rng_t: cute.Tensor,
        offsets_t: cute.Tensor,
        logical_len_t: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        a_row_layout_staged: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        acc_fake_layout: cute.Layout,
        num_tmem_alloc_cols: cutlass.Constexpr,
        hidden: cutlass.Int32,
        tokens: cutlass.Int32,
        num_tensors: cutlass.Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        tiles_in_m = hidden // cutlass.Int32(M_TILE)
        tiles_in_n_valid = logical_len_t[0] // cutlass.Int32(TOKEN_TILE)
        t_begin, t_end = _static_tile_range(tiles_in_m * tiles_in_n_valid, bidx, gdim)
        n_my = t_end - t_begin

        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, RHT128_MAINLOOP_STAGES * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, RHT128_ACC_STAGES * 2]
            b_mbar: cutlass.Int64
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar.data_ptr(),
            num_stages=RHT128_MAINLOOP_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=_A_TILE_BYTES,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        ab_producer, ab_consumer = ab_pipeline.make_participants()
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=RHT128_ACC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, RHT128_MSEDEN_ACC_CONSUMER_WARPS
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # MMA warp + both epilogue groups retrieve the TMEM pointer.
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_ALLOC_BAR, num_threads=32 + 2 * RHT128_MSEDEN_EPI_THREADS
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_DEALLOC_BAR, num_threads=2 * RHT128_MSEDEN_EPI_THREADS
        )
        sign_barrier = pipeline.NamedBarrier(
            barrier_id=RHT128_MSEDEN_SIGN_BAR, num_threads=2 * RHT128_MSEDEN_EPI_THREADS
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=RHT128_MSEDEN_COL_WARP_BEGIN,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        if warp_idx == TMA_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(
                    storage.b_mbar.ptr, 2 * RHT128_MSEDEN_EPI_THREADS
                )
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        raw_a = smem.allocate_array(
            cutlass.BFloat16,
            cute.cosize(a_smem_layout_staged.outer),
            byte_alignment=128,
        )
        sA = cute.make_tensor(
            cute.recast_ptr(raw_a, a_smem_layout_staged.inner, dtype=cutlass.BFloat16),
            a_smem_layout_staged.outer,
        )
        sA_row = cute.make_tensor(
            cute.recast_ptr(raw_a, a_row_layout_staged.inner, dtype=cutlass.BFloat16),
            a_row_layout_staged.outer,
        )
        raw_bcol = smem.allocate_array(
            cutlass.BFloat16, cute.cosize(b_smem_layout.outer), byte_alignment=128
        )
        raw_brow = smem.allocate_array(
            cutlass.BFloat16, cute.cosize(b_smem_layout.outer), byte_alignment=128
        )
        raw_sign = smem.allocate_array(cutlass.Int8, 2 * RHT128_DIM, byte_alignment=16)
        sBcol = cute.make_tensor(
            cute.recast_ptr(raw_bcol, b_smem_layout.inner, dtype=cutlass.BFloat16),
            b_smem_layout.outer,
        )
        sBrow = cute.make_tensor(
            cute.recast_ptr(raw_brow, b_smem_layout.inner, dtype=cutlass.BFloat16),
            b_smem_layout.outer,
        )

        cta_layout = cute.make_layout((1,))
        thr_col = tiled_mma_col.get_slice(0)
        gA = cute.local_tile(
            mA, cute.slice_(RHT128_CTA_TILE_COL, (None, 0, None)), (None, None, None)
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_col.partition_A(gA), 0, 3),
        )
        tCrA_col = tiled_mma_col.make_fragment_A(sA)
        tCrB_col = tiled_mma_col.make_fragment_B(sBcol)
        tCrA_row = tiled_mma_row.make_fragment_A(sA_row)
        tCrB_row = tiled_mma_row.make_fragment_B(sBrow)

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ==================== TMA warp ====================
        if warp_idx == TMA_WARP:
            for i in cutlass.range(n_my, unroll=1):
                t = t_begin + i
                tile_n = t // tiles_in_m
                tile_m = t - tile_n * tiles_in_m
                handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, tile_m, tile_n, 0)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
            ab_producer.tail()

        # ==================== MMA warp ====================
        if warp_idx == MMA_WARP:
            tmem.wait_for_alloc()
            tCtAcc = cute.make_tensor(
                tmem.retrieve_ptr(cutlass.Float32), acc_fake_layout
            )
            cute.arch.mbarrier_wait(storage.b_mbar.ptr, 0)
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, RHT128_ACC_STAGES
            )
            for i in cutlass.range(n_my, unroll=1):
                ab_handle = ab_consumer.wait_and_advance()
                acc_pipeline.producer_acquire(acc_state)
                acc_col = tCtAcc[(None, None, None, 2 * acc_state.index + 0)]
                acc_row = tCtAcc[(None, None, None, 2 * acc_state.index + 1)]
                for kb in cutlass.range_constexpr(RHT128_K_BLOCKS):
                    tiled_mma_col.set(tcgen05.Field.ACCUMULATE, kb > 0)
                    cute.gemm(
                        tiled_mma_col,
                        acc_col,
                        tCrA_col[(None, None, kb, ab_handle.index)],
                        tCrB_col[(None, None, kb, 0)],
                        acc_col,
                    )
                    tiled_mma_row.set(tcgen05.Field.ACCUMULATE, kb > 0)
                    cute.gemm(
                        tiled_mma_row,
                        acc_row,
                        tCrA_row[(None, None, kb, ab_handle.index)],
                        tCrB_row[(None, None, kb, 0)],
                        acc_row,
                    )
                # One elected tcgen05.commit covers both chains' 16 UMMAs.
                acc_pipeline.producer_commit(acc_state)
                acc_state.advance()
                ab_handle.release()
            acc_pipeline.producer_tail(acc_state)

        # ==================== col epilogue: ms_eden(dy.t() @ R_m) -> (col_fp4, col_sf) ====================
        if (
            warp_idx >= RHT128_MSEDEN_COL_WARP_BEGIN
            and warp_idx < RHT128_MSEDEN_COL_WARP_END
        ):
            _rht128_build_signed_rht(
                mH,
                col_sign_t,
                raw_sign.toint(),
                raw_bcol.toint(),
                tidx - cutlass.Int32(32 * RHT128_MSEDEN_COL_WARP_BEGIN),
                sign_barrier,
            )
            # Generic-proxy stores -> visible to the UMMA (async proxy) before the arrive.
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.mbarrier_arrive(storage.b_mbar.ptr)
            tmem.allocate(num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc = cute.make_tensor(tmem_ptr, acc_fake_layout)
            _rht128_ms_eden_epilogue(
                0,
                self.fast_path,
                mColFP4,
                mColSF,
                col_amax_t,
                sr_rng_t,
                tCtAcc,
                acc_pipeline,
                offsets_t,
                num_tensors,
                t_begin,
                t_end,
                tiles_in_m,
                hidden,
                tokens,
                tidx,
            )
            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ==================== row epilogue: ms_eden(dy @ R_n) -> (row_fp4, row_sf) ====================
        if (
            warp_idx >= RHT128_MSEDEN_ROW_WARP_BEGIN
            and warp_idx < RHT128_MSEDEN_ROW_WARP_END
        ):
            _rht128_build_signed_rht(
                mH,
                row_sign_t,
                raw_sign.toint() + cutlass.Int32(RHT128_DIM),
                raw_brow.toint(),
                tidx - cutlass.Int32(32 * RHT128_MSEDEN_ROW_WARP_BEGIN),
                sign_barrier,
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.mbarrier_arrive(storage.b_mbar.ptr)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc = cute.make_tensor(tmem_ptr, acc_fake_layout)
            _rht128_ms_eden_epilogue(
                1,
                self.fast_path,
                mRowFP4,
                mRowSF,
                row_amax_t,
                sr_rng_t,
                tCtAcc,
                acc_pipeline,
                offsets_t,
                num_tensors,
                t_begin,
                t_end,
                tiles_in_m,
                hidden,
                tokens,
                tidx,
            )
            tmem_dealloc_barrier.arrive_and_wait()


@functools.lru_cache(maxsize=None)
def _compile_group_row_rht_col_rht_quantize_ms_eden_kernel(
    device_idx: int, fast_path: bool
):
    """Compile the grouped row-RHT + col-RHT MS-EDEN quantize kernel with symbolic shapes
    (cached per device+flag)."""
    free = cute.sym_int
    h_sym = cute.sym_int(divisibility=M_TILE)
    t_sym = cute.sym_int(divisibility=TOKEN_TILE)
    k = _Tcgen05GroupRowRhtColRhtQuantizeMsEden(fast_path=fast_path)
    return cute.compile(
        k,
        make_fake_tensor(cutlass.BFloat16, (h_sym, t_sym, 1), stride=(1, free(), 1)),
        make_fake_tensor(
            cutlass.BFloat16,
            (RHT128_DIM, RHT128_DIM),
            stride=(RHT128_DIM, 1),
            assumed_align=16,
        ),
        make_fake_tensor(cutlass.Int8, (RHT128_DIM,), stride=(1,)),
        make_fake_tensor(cutlass.Int8, (RHT128_DIM,), stride=(1,)),
        # Each epilogue thread stores its warp's four blocks of a lane's 16 code words as
        # two 16-byte ``st.global.v4`` (``_rht128_tile_ms_eden``); assumed_align=16 and the
        # divisibilities record why that is aligned: hidden % 128 makes the rowwise u32
        # pitch hidden // 8 a multiple of 16, tokens % 128 the columnwise one, so every
        # lane's 64-byte code row and its 16-byte quarters start 16-byte aligned.
        make_fake_tensor(
            cutlass.Uint32,
            (t_sym, cute.sym_int(divisibility=M_TILE // 8)),
            stride=(cute.sym_int(divisibility=M_TILE // 8), 1),
            assumed_align=16,
        ),
        make_fake_tensor(
            cutlass.Uint32, (free(), free(), 32, 4), stride=(free(), 128, 4, 1)
        ),
        make_fake_tensor(
            cutlass.Uint32,
            (h_sym, cute.sym_int(divisibility=TOKEN_TILE // 8)),
            stride=(cute.sym_int(divisibility=TOKEN_TILE // 8), 1),
            assumed_align=16,
        ),
        make_fake_tensor(cutlass.Uint32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Float32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Float32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Int32, (8,), stride=(1,)),
        make_fake_tensor(cutlass.Int32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Int32, (1,), stride=(1,)),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _cutedsl_group_row_rht_col_rht_quantize_ms_eden_impl(
    dy: torch.Tensor,
    h128: torch.Tensor,
    dgrad_rht: torch.Tensor,
    wgrad_rht: torch.Tensor,
    amax_rht_dy: torch.Tensor,
    amax_rht_dy_t: torch.Tensor,
    offsets: torch.Tensor,
    num_tensors: int,
    rng_state: torch.Tensor,
    logical_packed_length: Optional[torch.Tensor] = None,
    fast_path: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MS-EDEN quantize of ``dy @ R_n`` (rowwise) and ``dy.t() @ R_m`` (columnwise), per group.

    ``h128`` is the cached sign-free ``(128, 128)`` bfloat16 Hadamard, ``dgrad_rht`` /
    ``wgrad_rht`` the live ``(128,)`` int8 sign vectors of the row / col chain; the kernel
    folds the signs into its resident ``R^T`` tiles.
    Returns ``(row_fp4, row_sf, col_fp4, col_sf)`` rowwise first: uint8 code views of the
    u32 buffers and the 4-D swizzled e4m3 scale storage the wrapper returns as 2-D views.
    Rows at or after ``logical_packed_length`` are never read and their outputs are left
    as allocated. ``fast_path`` selects the ``FAST_PATH`` kernel variant (hardware
    stochastic rounding, one Philox draw per 16 scales; not bitwise with the Triton op).
    """
    tokens, hidden = dy.shape
    dev = dy.device
    dy = dy.detach()

    row_fp4 = torch.empty((tokens, hidden // 8), dtype=torch.uint32, device=dev)
    col_fp4 = torch.empty((hidden, tokens // 8), dtype=torch.uint32, device=dev)
    row_sf = torch.empty(
        (tokens // 128, hidden // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev
    )
    col_sf = torch.empty(
        (hidden // 128, tokens // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev
    )

    sr_rng_t = _get_sr_rng_buffer(dev.index)
    # [col_seed, col_offset, row_seed, row_offset] int64 -> the eight little-endian
    # 32-bit halves (see the fused kernel).
    sr_rng_t.copy_(rng_state[:4].view(torch.int32))
    if logical_packed_length is None:
        logical_packed_length = offsets[-1:]
    # See the fused kernel: the entry point requires byte_offset==0.
    logical_packed_length = logical_packed_length.clone()
    tiles = (hidden // M_TILE) * (tokens // TOKEN_TILE)
    num_ctas = max(1, min(tiles, _get_num_sms(dev.index)))

    stream = cuda.CUstream(int(torch.cuda.current_stream(dev).cuda_stream))
    _compile_group_row_rht_col_rht_quantize_ms_eden_kernel(dev.index, bool(fast_path))(
        dy.t().unsqueeze(-1),
        h128,
        dgrad_rht,
        wgrad_rht,
        row_fp4,
        row_sf.view(torch.uint32),
        col_fp4,
        col_sf.view(torch.uint32).flatten(),
        amax_rht_dy,
        amax_rht_dy_t,
        sr_rng_t,
        offsets,
        logical_packed_length,
        int(hidden),
        int(tokens),
        int(num_tensors),
        int(num_ctas),
        stream,
    )
    return row_fp4.view(torch.uint8), row_sf, col_fp4.view(torch.uint8), col_sf


# --- grouped columnwise-RHT weight requantization (RHT-128 on the transposed weight) ---
# Flavor-B siblings of the two kernels above: an expert-uniform ``(E, M, N)`` weight
# stack, the expert derived from the tile index (no offsets, no group cap). Warps of the
# amax kernel: 0 MMA (+ the one-shot TMA of the resident ``H128``), 1-4 epilogue (lane =
# TMEM lane = output row n, quadrant-aligned via ``tidx % 128``), 5-12 one 256-thread
# producer group that dequantizes the codes into the MN-major SW128 A stage; the
# requantize kernel has eight epilogue warps (1-8) and the producers at 9-16
# (``RHT128_COLRHT_REQ_*``). Four A stages: the RHT-16 reserve-based
# ``RHT128_MAINLOOP_STAGES`` (5) is the TMA kernels' budget, not this one's.
RHT128_COLRHT_STAGES = 4
RHT128_COLRHT_EPI_WARP_BEGIN = 1
RHT128_COLRHT_EPI_WARP_END = 5
RHT128_COLRHT_PROD_WARP_BEGIN = 5
RHT128_COLRHT_N_WARPS = 13
RHT128_COLRHT_TPB = 32 * RHT128_COLRHT_N_WARPS
RHT128_COLRHT_PROD_THREADS = 32 * (
    RHT128_COLRHT_N_WARPS - RHT128_COLRHT_PROD_WARP_BEGIN
)
RHT128_COLRHT_EPI_WARPS = RHT128_COLRHT_EPI_WARP_END - RHT128_COLRHT_EPI_WARP_BEGIN
RHT128_COLRHT_A_STAGE_BYTES = RHT128_DIM * RHT128_DIM * 2
RHT128_COLRHT_ACC_ZERO_BAR = 3
# The producers stage each tile's codes and scale words in shared memory through
# ``cp.async``, ``RHT128_COLRHT_LOAD_STAGES - 1`` tiles ahead of the decode.
RHT128_COLRHT_LOAD_STAGES = 4
RHT128_COLRHT_LOAD_TILE_BYTES = RHT128_COLRHT_PROD_THREADS * (32 + 4)
# The requantize class runs eight epilogue warps (1-8), two per TMEM quadrant, each
# taking four of its lanes' eight blocks -- the MS-EDEN class's layout, for the same
# reason: a lane's RTNE block chain is latency-bound with one warp per scheduler.
RHT128_COLRHT_REQ_EPI_WARP_END = 9
RHT128_COLRHT_REQ_PROD_WARP_BEGIN = 9
RHT128_COLRHT_REQ_N_WARPS = 17
RHT128_COLRHT_REQ_TPB = 32 * RHT128_COLRHT_REQ_N_WARPS
RHT128_COLRHT_REQ_EPI_WARPS = (
    RHT128_COLRHT_REQ_EPI_WARP_END - RHT128_COLRHT_EPI_WARP_BEGIN
)
RHT128_COLRHT_REQ_EPI_THREADS = 32 * RHT128_COLRHT_REQ_EPI_WARPS
RHT128_COLRHT_REQ_BLOCKS_PER_WARP = (RHT128_DIM // 16) // 2


def _rht128_colrht_tile(t, tiles_per_expert, tiles_n):
    """Expert-major tile ``t`` -> ``(e, pid_m, pid_n)``; ``pid_n`` fastest."""
    e = t // tiles_per_expert
    r = t - e * tiles_per_expert
    pid_m = r // tiles_n
    pid_n = r - pid_m * tiles_n
    return e, pid_m, pid_n


@dsl_user_op
def _cp_async_16(dst: cutlass.Uint32, src: cutlass.Int64, *, loc=None, ip=None):
    """One 16-byte ``cp.async`` (L2-cached, L1-bypassing) from global byte address
    ``src`` to shared byte address ``dst``, both 16-byte aligned; joins the thread's
    open commit group."""
    llvm.inline_asm(
        None,
        [dst.ir_value(loc=loc, ip=ip), src.ir_value(loc=loc, ip=ip)],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _cp_async_4(dst: cutlass.Uint32, src: cutlass.Int64, *, loc=None, ip=None):
    """One 4-byte ``cp.async`` from global byte address ``src`` to shared byte address
    ``dst``; joins the thread's open commit group."""
    llvm.inline_asm(
        None,
        [dst.ir_value(loc=loc, ip=ip), src.ir_value(loc=loc, ip=ip)],
        "cp.async.ca.shared.global [$0], [$1], 4;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _rht128_colrht_prefetch(
    mCodes, mSF, dst_codes, dst_sf, e, pid_m, pid_n, M, N, m, h
):
    """``cp.async`` producer thread ``(m, h)``'s 32 B of codes and its scale word of tile
    ``(e, pid_m, pid_n)`` into its bytes of a staging slot (one commit group): two 16 B
    copies from the code row and one u32 of the swizzled scale atom ``2 * pid_n + h``
    (row ``m``'s word is ``(m % 32) * 4 + m // 32``). Plain function: traced inline by the
    producer, with 64-bit expert bases."""
    word = (
        cutlass.Int64(e) * cutlass.Int64(M)
        + cutlass.Int64(pid_m * cutlass.Int32(RHT128_DIM) + m)
    ) * cutlass.Int64(N // cutlass.Int32(8)) + cutlass.Int64(
        pid_n * cutlass.Int32(16) + h * cutlass.Int32(8)
    )
    addr = mCodes.iterator.toint() + word * cutlass.Int64(4)
    _cp_async_16(dst_codes, addr)
    _cp_async_16(dst_codes + cutlass.Uint32(16), addr + cutlass.Int64(16))
    atom = (
        cutlass.Int64(e) * cutlass.Int64(M // cutlass.Int32(RHT128_DIM))
        + cutlass.Int64(pid_m)
    ) * cutlass.Int64(N // cutlass.Int32(64)) + cutlass.Int64(
        pid_n * cutlass.Int32(2) + h
    )
    _cp_async_4(
        dst_sf,
        mSF.iterator.toint()
        + (
            atom * cutlass.Int64(128)
            + cutlass.Int64(
                (m % cutlass.Int32(32)) * cutlass.Int32(4) + m // cutlass.Int32(32)
            )
        )
        * cutlass.Int64(4),
    )
    cute.arch.cp_async_commit_group()


def _rht128_colrht_next_tile(e, pid_m, pid_n, tiles_m, tiles_n, more):
    """``(e, pid_m, pid_n)`` of the next tile (``pid_n`` fastest, expert-major), or the
    same tile when ``more`` is false: loads past a chunk's last tile re-load it rather
    than branching."""
    pn = pid_n + cutlass.Int32(1)
    wrap_n = pn == tiles_n
    pn = cutlass.Int32(cutlass.select_(wrap_n, cutlass.Int32(0), pn))
    pm = pid_m + cutlass.Int32(
        cutlass.select_(wrap_n, cutlass.Int32(1), cutlass.Int32(0))
    )
    wrap_m = pm == tiles_m
    pm = cutlass.Int32(cutlass.select_(wrap_m, cutlass.Int32(0), pm))
    en = e + cutlass.Int32(cutlass.select_(wrap_m, cutlass.Int32(1), cutlass.Int32(0)))
    return (
        cutlass.Int32(cutlass.select_(more, en, e)),
        cutlass.Int32(cutlass.select_(more, pm, pid_m)),
        cutlass.Int32(cutlass.select_(more, pn, pid_n)),
    )


@dsl_user_op
def _rht128_colrht_dequant_word(
    word: cutlass.Uint32,
    sf: cutlass.Float32,
    gds: cutlass.Float32,
    *,
    loc=None,
    ip=None,
):
    """``_dequant_e2m1x8_bf16x2x4`` with its sixteen ``mul.rn.f32`` as eight ``mul.f32x2``
    (per lane the same correctly rounded product, as the epilogue's packs): exact e2m1
    decode, the exact product by the block scale, the one rounding by the decode scale,
    ``cvt.rn.bf16x2.f32``. Word k holds elements ``2k`` (low half) and ``2k + 1``."""
    rst = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [
            word.ir_value(loc=loc, ip=ip),
            sf.ir_value(loc=loc, ip=ip),
            gds.ir_value(loc=loc, ip=ip),
        ],
        (
            "{\n"
            ".reg .b8 b0, b1, b2, b3;\n"
            ".reg .b32 p0, p1, p2, p3;\n"
            ".reg .b16 l0, h0, l1, h1, l2, h2, l3, h3;\n"
            ".reg .f32 q0, q1, q2, q3, q4, q5, q6, q7;\n"
            ".reg .b64 s2, g2, r01, r23, r45, r67;\n"
            "mov.b32 {b0, b1, b2, b3}, $4;\n"
            "cvt.rn.f16x2.e2m1x2 p0, b0;\n"
            "cvt.rn.f16x2.e2m1x2 p1, b1;\n"
            "cvt.rn.f16x2.e2m1x2 p2, b2;\n"
            "cvt.rn.f16x2.e2m1x2 p3, b3;\n"
            "mov.b32 {l0, h0}, p0;\n"
            "mov.b32 {l1, h1}, p1;\n"
            "mov.b32 {l2, h2}, p2;\n"
            "mov.b32 {l3, h3}, p3;\n"
            "cvt.f32.f16 q0, l0;\n"
            "cvt.f32.f16 q1, h0;\n"
            "cvt.f32.f16 q2, l1;\n"
            "cvt.f32.f16 q3, h1;\n"
            "cvt.f32.f16 q4, l2;\n"
            "cvt.f32.f16 q5, h2;\n"
            "cvt.f32.f16 q6, l3;\n"
            "cvt.f32.f16 q7, h3;\n"
            "mov.b64 s2, {$5, $5};\n"
            "mov.b64 g2, {$6, $6};\n"
            "mov.b64 r01, {q0, q1};\n"
            "mov.b64 r23, {q2, q3};\n"
            "mov.b64 r45, {q4, q5};\n"
            "mov.b64 r67, {q6, q7};\n"
            "mul.f32x2 r01, r01, s2;\n"
            "mul.f32x2 r23, r23, s2;\n"
            "mul.f32x2 r45, r45, s2;\n"
            "mul.f32x2 r67, r67, s2;\n"
            "mul.f32x2 r01, r01, g2;\n"
            "mul.f32x2 r23, r23, g2;\n"
            "mul.f32x2 r45, r45, g2;\n"
            "mul.f32x2 r67, r67, g2;\n"
            "mov.b64 {q0, q1}, r01;\n"
            "mov.b64 {q2, q3}, r23;\n"
            "mov.b64 {q4, q5}, r45;\n"
            "mov.b64 {q6, q7}, r67;\n"
            "cvt.rn.bf16x2.f32 $0, q1, q0;\n"
            "cvt.rn.bf16x2.f32 $1, q3, q2;\n"
            "cvt.rn.bf16x2.f32 $2, q5, q4;\n"
            "cvt.rn.bf16x2.f32 $3, q7, q6;\n"
            "}"
        ),
        "=r,=r,=r,=r,r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(
        cutlass.Uint32(llvm.extractvalue(T.i32(), rst, [k], loc=loc, ip=ip))
        for k in range(4)
    )


def _rht128_colrht_put_word(word, sf, gds_m, st4, off):
    """Dequantize one code word (eight columns of the tile row) and store its four
    bf16x2 words as one 16 B chunk at swizzled stage byte offset ``off``. Plain
    function, traced inline."""
    st4[0], st4[1], st4[2], st4[3] = _rht128_colrht_dequant_word(word, sf, gds_m)
    cute.make_tensor(
        cute.make_ptr(cutlass.Uint32, off, cute.AddressSpace.smem, assumed_align=16),
        cute.make_layout((4,)),
    ).store(st4.load())


@cute.jit
def _rht128_dequant_a_producer(
    mCodes,
    mSF,
    amax_t,
    sign_t,
    a_base,
    stg_base,
    ab_pipeline,
    t_begin,
    n_my,
    tiles_per_expert,
    tiles_n,
    M,
    N,
    p,
):
    """Producer thread ``p`` of 256: rebuild its 128 B of ``W_qdq`` per tile straight into
    the MN-major SW128 A stage, for this CTA's whole tile chunk.

    Thread ``p`` owns weight row ``m = p % 128`` (the contracted index) and half
    ``h = p // 128`` of its columns: code words ``8h .. 8h + 7`` (16 B each in the tile
    row) and the one u32 of scale atom ``2 * pid_n + h`` that holds their four E4M3
    scales. The codes and scale word of a tile are ``cp.async``-staged in shared memory
    ``RHT128_COLRHT_LOAD_STAGES - 1`` tiles ahead (this thread's 36 B of each slot), so
    the DRAM round trip overlaps several tiles' decode without rotating registers.

    Numerics are the Triton reconstruction's, op for op (``_rht128_colrht_dequant_word``),
    with the sign vector folded into the per-row decode scale ``gds_m = s_m * gds`` (an
    exact sign flip) and an invalid (NaN/inf) expert amax filling the row with the
    sign-folded zero ``s_m * (+0)``: every A element is Triton's times ``s_m``, so every
    UMMA product -- including the sign of each zero product -- equals Triton's product of
    the unsigned element with the signed ``R_n`` row.

    Each code word becomes one 16 B store at the closed-form swizzled offset
    ``2048 * (m // 8) + 128 * (m % 8) + 1024 * h + 16 * (k ^ (m % 8))`` of a 1 KB-aligned
    stage; tile coordinates advance incrementally and the per-expert row constants are
    refreshed only when the expert changes.
    """
    m = p % cutlass.Int32(RHT128_DIM)
    h = p // cutlass.Int32(RHT128_DIM)
    row_off = (
        a_base
        + cutlass.Int32(2048) * (m // cutlass.Int32(8))
        + cutlass.Int32(128) * (m % cutlass.Int32(8))
        + cutlass.Int32(1024) * h
    )
    x = m % cutlass.Int32(8)
    inf = cutlass.Float32(float("inf"))

    pre_lo = cute.make_rmem_tensor((4,), cutlass.Uint32)
    pre_hi = cute.make_rmem_tensor((4,), cutlass.Uint32)
    st4 = cute.make_rmem_tensor((4,), cutlass.Uint32)
    # This thread's bytes of a staging slot: 32 B of codes, then its scale word.
    my_codes = stg_base + cutlass.Uint32(p) * cutlass.Uint32(32)
    my_sf = (
        stg_base
        + cutlass.Uint32(RHT128_COLRHT_PROD_THREADS * 32)
        + cutlass.Uint32(p) * cutlass.Uint32(4)
    )

    ab_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, RHT128_COLRHT_STAGES
    )
    tiles_m = tiles_per_expert // tiles_n
    # Tile coordinates advance incrementally (pid_n fastest, expert-major): the only
    # integer divisions are the chunk's first tile's. The load coordinates run
    # ``RHT128_COLRHT_LOAD_STAGES - 1`` tiles ahead of the decode's.
    e, pid_m, pid_n = _rht128_colrht_tile(t_begin, tiles_per_expert, tiles_n)
    e_l = e
    pm_l = pid_m
    pn_l = pid_n
    for d in cutlass.range_constexpr(RHT128_COLRHT_LOAD_STAGES - 1):
        off_d = cutlass.Uint32(d * RHT128_COLRHT_LOAD_TILE_BYTES)
        _rht128_colrht_prefetch(
            mCodes, mSF, my_codes + off_d, my_sf + off_d, e_l, pm_l, pn_l, M, N, m, h
        )
        e_l, pm_l, pn_l = _rht128_colrht_next_tile(
            e_l, pm_l, pn_l, tiles_m, tiles_n, cutlass.Int32(d + 1) < n_my
        )
    neg = sign_t[m] < cutlass.Int8(0)
    # Per-expert row constants, refreshed on an expert change only.
    e_cur = cutlass.Int32(-1)
    valid = cutlass.Int32(0)
    gds_m = cutlass.Float32(0.0)
    zero_m = cutlass.Uint32(0)
    # Slot loaded / decoded this iteration: a load reuses the slot decoded last time.
    s_load = cutlass.Int32(RHT128_COLRHT_LOAD_STAGES - 1)
    s_read = cutlass.Int32(0)
    for i in cutlass.range(n_my, unroll=1):
        off_l = cutlass.Uint32(s_load) * cutlass.Uint32(RHT128_COLRHT_LOAD_TILE_BYTES)
        _rht128_colrht_prefetch(
            mCodes, mSF, my_codes + off_l, my_sf + off_l, e_l, pm_l, pn_l, M, N, m, h
        )
        e_l, pm_l, pn_l = _rht128_colrht_next_tile(
            e_l,
            pm_l,
            pn_l,
            tiles_m,
            tiles_n,
            i + cutlass.Int32(RHT128_COLRHT_LOAD_STAGES) < n_my,
        )

        if e != e_cur:
            amax_e = amax_t[e]
            _, gds, _ = _global_scale(amax_e)
            valid = cutlass.Int32(
                cutlass.select_(
                    _abs_f32(amax_e) < inf, cutlass.Int32(1), cutlass.Int32(0)
                )
            )
            gds_m = cutlass.Float32(cutlass.select_(neg, -gds, gds))
            zero_m = cutlass.Uint32(
                cutlass.select_(neg, cutlass.Uint32(0x80008000), cutlass.Uint32(0))
            )
            e_cur = e

        # Tile i's copies have landed once only the newer groups may be pending.
        cute.arch.cp_async_wait_group(RHT128_COLRHT_LOAD_STAGES - 1)
        off_r = cutlass.Uint32(s_read) * cutlass.Uint32(RHT128_COLRHT_LOAD_TILE_BYTES)
        pre_lo.store(
            cute.make_tensor(
                cute.make_ptr(
                    cutlass.Uint32,
                    my_codes + off_r,
                    cute.AddressSpace.smem,
                    assumed_align=16,
                ),
                cute.make_layout((4,)),
            ).load()
        )
        pre_hi.store(
            cute.make_tensor(
                cute.make_ptr(
                    cutlass.Uint32,
                    my_codes + off_r + cutlass.Uint32(16),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                ),
                cute.make_layout((4,)),
            ).load()
        )
        sfw = cute.make_tensor(
            cute.make_ptr(
                cutlass.Uint32, my_sf + off_r, cute.AddressSpace.smem, assumed_align=4
            ),
            cute.make_layout((1,)),
        )[0]
        c0 = pre_lo[0]
        c1 = pre_lo[1]
        c2 = pre_lo[2]
        c3 = pre_lo[3]
        c4 = pre_hi[0]
        c5 = pre_hi[1]
        c6 = pre_hi[2]
        c7 = pre_hi[3]

        ab_pipeline.producer_acquire(ab_state)
        stage_off = row_off + ab_state.index * cutlass.Int32(
            RHT128_COLRHT_A_STAGE_BYTES
        )
        if valid != cutlass.Int32(0):
            for j in cutlass.range_constexpr(2):
                sfp = _e4m3x2_to_f16x2(sfw >> cutlass.Uint32(16 * j))
                sf_ab = _f16lo_to_f32(sfp)
                sf_cd = _f16hi_to_f32(sfp)
                words = (c0, c1, c2, c3) if j == 0 else (c4, c5, c6, c7)
                for k in cutlass.range_constexpr(4):
                    _rht128_colrht_put_word(
                        words[k],
                        sf_ab if k < 2 else sf_cd,
                        gds_m,
                        st4,
                        stage_off + cutlass.Int32(16) * (cutlass.Int32(4 * j + k) ^ x),
                    )
        else:
            # An invalid expert amax: the row is the sign-folded zero in every word.
            for k in cutlass.range_constexpr(4):
                st4[k] = zero_m
            for k in cutlass.range_constexpr(8):
                cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Uint32,
                        stage_off + cutlass.Int32(16) * (cutlass.Int32(k) ^ x),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                    cute.make_layout((4,)),
                ).store(st4.load())
        # Generic-proxy stores -> visible to the UMMA (async proxy) before the arrive.
        cute.arch.fence_proxy("async.shared", space="cta")
        ab_pipeline.producer_commit(ab_state)
        ab_state.advance()
        s_load = s_read
        s_read = cutlass.Int32(
            cutlass.select_(
                s_read + cutlass.Int32(1) == cutlass.Int32(RHT128_COLRHT_LOAD_STAGES),
                cutlass.Int32(0),
                s_read + cutlass.Int32(1),
            )
        )
        e, pid_m, pid_n = _rht128_colrht_next_tile(
            e, pid_m, pid_n, tiles_m, tiles_n, i + cutlass.Int32(1) < n_my
        )
    cute.arch.cp_async_wait_group(0)
    ab_pipeline.producer_tail(ab_state)


def _store_expert_col_sf_word(mSF_u32, rSF, expert_words, r, c_base, m_blocks64):
    """``_store_grouped_col_sf_word`` for an expert-uniform stack: the u32 word of expert
    ``e``'s swizzled ``(N//128, M//64, 32, 16)`` scale tile that holds row ``r``'s scale
    bytes for columns ``c_base .. c_base + 3`` (``c_base % 4 == 0``); the expert's tile
    starts ``expert_words`` into the flat buffer."""
    r_blk = r // cutlass.Int32(128)
    r_lane = r % cutlass.Int32(32)
    r_grp = (r % cutlass.Int32(128)) // cutlass.Int32(32)
    mSF_u32[
        expert_words
        + r_blk * (m_blocks64 * cutlass.Int32(128))
        + (c_base // cutlass.Int32(4)) * cutlass.Int32(128)
        + r_lane * cutlass.Int32(4)
        + r_grp
    ] = cute.recast_tensor(rSF, cutlass.Uint32)[0]


@cute.jit
def _rht128_colrht_amax_epilogue(
    amax_t, tCtAcc, acc_pipeline, t_begin, t_end, tiles_per_expert, tidx, lane
):
    """Consume the acc ring over this CTA's chunk: ``max|bf16(acc)|`` per expert.

    Tiles are expert-major, so the running max flushes on an expert change and once at
    the end, as ``_rht128_amax_epilogue`` does per group.
    """
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, RHT128_ACC_STAGES
    )
    g = t_begin // tiles_per_expert
    run_max = cutlass.Float32(0.0)
    for i in cutlass.range(t_end - t_begin, unroll=1):
        e = (t_begin + i) // tiles_per_expert
        if e != g:
            _flush_group_max(run_max, amax_t, g, lane)
            run_max = cutlass.Float32(0.0)
            g = e
        acc_pipeline.consumer_wait(acc_state)
        tile_max = _rht128_tile_amax(tCtAcc[(None, None, None, acc_state.index)], tidx)
        cute.arch.fence_view_async_tmem_load()
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_state)
        acc_state.advance()
        run_max = _max_f32(run_max, _round_rht_amax(tile_max))
    _flush_group_max(run_max, amax_t, g, lane)


@cute.jit
def _rht128_zero_acc(acc, tidx, u_base):
    """``tcgen05.st`` zeros over this warp's four 16-column chunks (``u_base .. u_base + 3``)
    of one 128x128 f32 TMEM accumulator, per thread (= per lane): the next chain
    accumulates onto ``+0`` exactly as Triton's zero-filled ``tl.dot``."""
    st_atom = cute.make_copy_atom(
        tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition.x16), cutlass.Float32
    )
    tAcc = transform_partitioned_tensor_layout(acc)
    tAcc_epi = cute.flat_divide(tAcc, RHT128_EPI_TILE)
    tiled_copy_r2t = tcgen05.make_tmem_copy(st_atom, tAcc_epi[(None, None, 0, 0)])
    thr_copy = tiled_copy_r2t.get_slice(tidx)
    tRT_tAcc = thr_copy.partition_D(tAcc_epi)
    zeros = cute.make_rmem_tensor(((16, 1), 1, 1), cutlass.Float32)
    for i in cutlass.range_constexpr(16):
        zeros[i] = cutlass.Float32(0.0)
    for j in cutlass.range_constexpr(RHT128_COLRHT_REQ_BLOCKS_PER_WARP):
        u = u_base + cutlass.Int32(j)
        cute.copy(tiled_copy_r2t, zeros, tRT_tAcc[(None, None, None, 0, u)])


@cute.jit
def _rht128_tile_requantize(acc, tidx, u_base, enc_over_fp4max, dec, row_addr, rSF):
    """RTNE-quantize this warp's half of one 128x128 f32 TMEM accumulator: this thread's
    lane = one output row, blocks ``u_base .. u_base + 3`` of its eight along the
    transformed index.

    The codes go straight to global as two 16-byte stores into the lane's code words
    ``8 * (u_base // 4) ..`` at ``row_addr``; the four scale bytes land in ``rSF`` for the
    swizzled store. After its loads the thread zero-fills its chunks of the stage
    (``_rht128_zero_acc``) so the next tile's chain accumulates onto ``+0``.
    """
    copy_atom_t2r = sm100_utils.get_tmem_load_op(
        RHT128_CTA_TILE_COL,
        utils.LayoutEnum.ROW_MAJOR,
        cutlass.Float32,
        cutlass.Float32,
        RHT128_EPI_TILE,
        False,
    )
    tAcc = transform_partitioned_tensor_layout(acc)
    tAcc_epi = cute.flat_divide(tAcc, RHT128_EPI_TILE)
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
    thr_copy = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc = thr_copy.partition_S(tAcc_epi)
    rCodes = cute.make_rmem_tensor(
        (2 * RHT128_COLRHT_REQ_BLOCKS_PER_WARP,), cutlass.Uint32
    )
    # The next block's TMEM load is in flight while this block's chain runs: ptxas
    # serialises the loads otherwise (each chain's reciprocal slow path is a
    # reconvergence region), and more than two live fragments spill around it.
    tTR_rAcc = [
        cute.make_rmem_tensor(((16, 1), 1, 1), cutlass.Float32) for _ in range(2)
    ]
    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, u_base)], tTR_rAcc[0])
    zero = cutlass.Float32(0.0)
    for j in cutlass.range_constexpr(RHT128_COLRHT_REQ_BLOCKS_PER_WARP):
        if cutlass.const_expr(j + 1 < RHT128_COLRHT_REQ_BLOCKS_PER_WARP):
            u = u_base + cutlass.Int32(j + 1)
            cute.copy(
                tiled_copy_t2r,
                tTR_tAcc[(None, None, None, 0, u)],
                tTR_rAcc[(j + 1) % 2],
            )
        vals = tTR_rAcc[j % 2].load().reshape((16,))
        # The bf16-exact values serve both the block amax and the encode multiply, as
        # in ``_ms_eden_block16``.
        e = _bf16round_f32x8(
            vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], zero
        ) + _bf16round_f32x8(
            vals[8],
            vals[9],
            vals[10],
            vals[11],
            vals[12],
            vals[13],
            vals[14],
            vals[15],
            zero,
        )
        enc, rSF[j], _ = _ms_eden_enc_from_amax(
            _abs_amax16(e), enc_over_fp4max, dec, cutlass.Float32(FP8_E4M3_MAX)
        )
        rCodes[2 * j], rCodes[2 * j + 1] = _pack16_rn_from_enc(e, enc)
    cute.arch.fence_view_async_tmem_load()
    _rht128_zero_acc(acc, tidx, u_base)
    cute.arch.fence_view_async_tmem_store()
    # 16-byte aligned by construction (64-byte code rows; ``u_base * 8`` is 0 or 32).
    code_addr = row_addr + cutlass.Int64(u_base) * cutlass.Int64(8)
    _st_global_v4_u32(code_addr, rCodes[0], rCodes[1], rCodes[2], rCodes[3])
    _st_global_v4_u32(
        code_addr + cutlass.Int64(16), rCodes[4], rCodes[5], rCodes[6], rCodes[7]
    )


@cute.jit
def _rht128_colrht_requantize_epilogue(
    mOutCodes,
    mOutSF,
    amax_t,
    tCtAcc,
    acc_pipeline,
    t_begin,
    t_end,
    tiles_per_expert,
    tiles_n,
    M,
    N,
    tidx,
    u_base,
):
    """Consume the acc ring over this CTA's chunk: quantize every tile into the expert's
    ``(N, M//8)`` u32 code rows and its swizzled ``(N//128, M//64, 32, 16)`` scales; this
    warp takes blocks ``u_base .. u_base + 3`` of its lanes."""
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, RHT128_ACC_STAGES
    )
    codes_base = mOutCodes.iterator.toint()
    words_per_row = M // cutlass.Int32(8)
    m_blocks64 = M // cutlass.Int32(64)
    # Int32 like the read side: E * M * N / 64 words in the flat scale buffer.
    sf_words_per_expert = (N // cutlass.Int32(128)) * m_blocks64 * cutlass.Int32(128)
    rSF = cute.make_rmem_tensor(
        (RHT128_COLRHT_REQ_BLOCKS_PER_WARP,), cutlass.Float8E4M3FN
    )
    tiles_m = tiles_per_expert // tiles_n
    # Tile coordinates advance incrementally and the expert's scale scalars are refreshed
    # on an expert change only, as in the producer.
    e, pid_m, pid_n = _rht128_colrht_tile(t_begin, tiles_per_expert, tiles_n)
    e_cur = cutlass.Int32(-1)
    dec = cutlass.Float32(0.0)
    enc_over_fp4max = cutlass.Float32(0.0)
    for i in cutlass.range(t_end - t_begin, unroll=1):
        if e != e_cur:
            _, dec, enc_over_fp4max = _global_scale(amax_t[e])
            e_cur = e
        n_glob = pid_n * cutlass.Int32(RHT128_DIM) + tidx
        row_addr = (
            codes_base
            + (cutlass.Int64(e) * cutlass.Int64(N) + cutlass.Int64(n_glob))
            * cutlass.Int64(words_per_row)
            * cutlass.Int64(4)
            + cutlass.Int64(pid_m) * cutlass.Int64(64)
        )
        acc_pipeline.consumer_wait(acc_state)
        _rht128_tile_requantize(
            tCtAcc[(None, None, None, acc_state.index)],
            tidx,
            u_base,
            enc_over_fp4max,
            dec,
            row_addr,
            rSF,
        )
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_state)
        acc_state.advance()
        _store_expert_col_sf_word(
            mOutSF,
            rSF,
            e * sf_words_per_expert,
            n_glob,
            pid_m * cutlass.Int32(RHT128_DIM // 16) + u_base,
            m_blocks64,
        )
        e, pid_m, pid_n = _rht128_colrht_next_tile(
            e, pid_m, pid_n, tiles_m, tiles_n, i + cutlass.Int32(1) < t_end - t_begin
        )


class _Tcgen05GroupColRhtRequantAmax:
    """Per-expert ``max|W_qdq.bf16().t() @ R_n|`` from the rowwise NVFP4 weight.

    Expert-uniform ``(E, M, N)`` stack, expert = tile-derived, no group cap. Standalone:
    the producer warps rebuild each 128x128 ``W_qdq`` tile into the MN-major A stage
    with the signs folded in (``_rht128_dequant_a_producer``), the MMA warp runs one
    UMMA chain against the resident TMA'd ``H128``, the epilogue warps reduce
    ``max|bf16(acc)|`` and atomic-max it into the pre-zeroed per-expert slot. Static
    persistent grid: CTA ``b`` owns the contiguous tile chunk ``_static_tile_range``,
    tiles expert-major.
    """

    @cute.jit
    def __call__(
        self,
        mCodes: cute.Tensor,  # row_fp4_w as flat u32 (E*M*N//8,)
        mSF: cute.Tensor,  # row_sf_w as flat u32 (E*M*N//64,)
        mAmax: cute.Tensor,  # global_amax (E,) f32
        mSign: cute.Tensor,  # dgrad_rht (128,) int8
        mB: cute.Tensor,  # H128 (128 j, 128 k, 1), K-major (symmetric)
        amax_out: cute.Tensor,  # (E,) f32, pre-zeroed
        M: cutlass.Int32,
        N: cutlass.Int32,
        E: cutlass.Int32,
        num_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        k_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        mn_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.MN_SW128, cutlass.BFloat16
        )
        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        mma_col = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            RHT128_MMA_TILER_COL,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.MN,
            OperandMajorMode.K,
        )
        tiled_mma_col = cute.make_tiled_mma(cute.make_mma_atom(mma_col))
        a_shape = tiled_mma_col.partition_shape_A(
            cute.dice(RHT128_CTA_TILE_COL, (1, None, 1))
        )
        a_smem_layout_staged = tcgen05.tile_to_mma_shape(
            mn_atom, cute.append(a_shape, RHT128_COLRHT_STAGES), order=(1, 2, 3)
        )
        b_shape = tiled_mma_col.partition_shape_B(
            cute.dice(RHT128_CTA_TILE_COL, (None, 1, 1))
        )
        b_smem_layout = tcgen05.tile_to_mma_shape(
            k_atom, cute.append(b_shape, 1), order=(1, 2, 3)
        )
        tma_atom_b_col, tma_tensor_b_col = cute.nvgpu.make_tiled_tma_atom_B(
            g2s,
            mB,
            cute.slice_(b_smem_layout, (None, None, None, 0)),
            RHT128_CTA_TILE_COL,
            tiled_mma_col,
            (1, 1, 1, 1),
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma_col.thr_id.shape,)
        )
        tCtAcc_fake = tiled_mma_col.make_fragment_C(
            cute.append(
                tiled_mma_col.partition_shape_C((M_TILE, RHT128_DIM)),
                RHT128_ACC_STAGES,
            )
        )
        num_tmem_alloc_cols = sm100_utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        self.kernel(
            tiled_mma_col,
            tma_atom_b_col,
            tma_tensor_b_col,
            mCodes,
            mSF,
            mAmax,
            mSign,
            amax_out,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            b_smem_layout,
            tCtAcc_fake.layout,
            num_tmem_alloc_cols,
            M,
            N,
            E,
        ).launch(grid=(num_ctas, 1, 1), block=(RHT128_COLRHT_TPB, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        tiled_mma_col: cute.TiledMma,
        tma_atom_b_col: cute.CopyAtom,
        mBcol: cute.Tensor,
        mCodes: cute.Tensor,
        mSF: cute.Tensor,
        mAmax: cute.Tensor,
        mSign: cute.Tensor,
        amax_out: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        acc_fake_layout: cute.Layout,
        num_tmem_alloc_cols: cutlass.Constexpr,
        M: cutlass.Int32,
        N: cutlass.Int32,
        E: cutlass.Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % cutlass.Int32(32)
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        tiles_n = N // cutlass.Int32(RHT128_DIM)
        tiles_per_expert = (M // cutlass.Int32(RHT128_DIM)) * tiles_n
        t_begin, t_end = _static_tile_range(E * tiles_per_expert, bidx, gdim)
        n_my = t_end - t_begin

        if warp_idx == MMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_b_col)

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, RHT128_COLRHT_STAGES * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, RHT128_ACC_STAGES * 2]
            b_mbar: cutlass.Int64
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.ab_mbar.data_ptr(),
            num_stages=RHT128_COLRHT_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, RHT128_COLRHT_PROD_THREADS
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=RHT128_ACC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, RHT128_COLRHT_EPI_WARPS
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # MMA warp + the epilogue group retrieve the TMEM pointer.
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_ALLOC_BAR, num_threads=32 + COL_THREADS
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_DEALLOC_BAR, num_threads=COL_THREADS
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=RHT128_COLRHT_EPI_WARP_BEGIN,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        if warp_idx == MMA_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(storage.b_mbar.ptr, 1)
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # 1 KB alignment: the producer's closed-form swizzle assumes it of the stage base.
        raw_a = smem.allocate_array(
            cutlass.BFloat16,
            cute.cosize(a_smem_layout_staged.outer),
            byte_alignment=1024,
        )
        sA = cute.make_tensor(
            cute.recast_ptr(raw_a, a_smem_layout_staged.inner, dtype=cutlass.BFloat16),
            a_smem_layout_staged.outer,
        )
        sBcol = smem.allocate_tensor(
            cutlass.BFloat16, b_smem_layout.outer, 128, swizzle=b_smem_layout.inner
        )
        # Staging ring of the producers' codes and scale words (``cp.async`` targets).
        raw_stg = smem.allocate_array(
            cutlass.Uint32,
            RHT128_COLRHT_LOAD_STAGES * RHT128_COLRHT_LOAD_TILE_BYTES // 4,
            byte_alignment=16,
        )

        cta_layout = cute.make_layout((1,))
        thr_col = tiled_mma_col.get_slice(0)
        gBcol = cute.local_tile(
            mBcol, cute.slice_(RHT128_CTA_TILE_COL, (0, None, None)), (None, None, None)
        )
        tBsBcol, tBgBcol = cpasync.tma_partition(
            tma_atom_b_col,
            0,
            cta_layout,
            cute.group_modes(sBcol, 0, 3),
            cute.group_modes(thr_col.partition_B(gBcol), 0, 3),
        )
        tCrA_col = tiled_mma_col.make_fragment_A(sA)
        tCrB_col = tiled_mma_col.make_fragment_B(sBcol)

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ==================== MMA warp (+ one-shot B TMA) ====================
        if warp_idx == MMA_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    storage.b_mbar.ptr, RHT128_B_BYTES
                )
            cute.copy(
                tma_atom_b_col,
                tBgBcol[(None, 0, 0, 0)],
                tBsBcol[(None, 0)],
                tma_bar_ptr=storage.b_mbar.ptr,
            )
            tmem.wait_for_alloc()
            tCtAcc = cute.make_tensor(
                tmem.retrieve_ptr(cutlass.Float32), acc_fake_layout
            )
            cute.arch.mbarrier_wait(storage.b_mbar.ptr, 0)
            ab_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, RHT128_COLRHT_STAGES
            )
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, RHT128_ACC_STAGES
            )
            for i in cutlass.range(n_my, unroll=1):
                ab_pipeline.consumer_wait(ab_state)
                acc_pipeline.producer_acquire(acc_state)
                acc_col = tCtAcc[(None, None, None, acc_state.index)]
                for kb in cutlass.range_constexpr(RHT128_K_BLOCKS):
                    tiled_mma_col.set(tcgen05.Field.ACCUMULATE, kb > 0)
                    cute.gemm(
                        tiled_mma_col,
                        acc_col,
                        tCrA_col[(None, None, kb, ab_state.index)],
                        tCrB_col[(None, None, kb, 0)],
                        acc_col,
                    )
                acc_pipeline.producer_commit(acc_state)
                acc_state.advance()
                ab_pipeline.consumer_release(ab_state)
                ab_state.advance()
            acc_pipeline.producer_tail(acc_state)

        # ==================== epilogue: max|W_qdq.t() @ R_n| per expert ====================
        if (
            warp_idx >= RHT128_COLRHT_EPI_WARP_BEGIN
            and warp_idx < RHT128_COLRHT_EPI_WARP_END
        ):
            tmem.allocate(num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc = cute.make_tensor(tmem_ptr, acc_fake_layout)
            _rht128_colrht_amax_epilogue(
                amax_out,
                tCtAcc,
                acc_pipeline,
                t_begin,
                t_end,
                tiles_per_expert,
                tidx % cutlass.Int32(RHT128_DIM),
                lane,
            )
            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ==================== producers: W_qdq tiles into the A ring ====================
        if warp_idx >= RHT128_COLRHT_PROD_WARP_BEGIN:
            _rht128_dequant_a_producer(
                mCodes,
                mSF,
                mAmax,
                mSign,
                raw_a.toint(),
                cutlass.Uint32(raw_stg.toint()),
                ab_pipeline,
                t_begin,
                n_my,
                tiles_per_expert,
                tiles_n,
                M,
                N,
                tidx - cutlass.Int32(32 * RHT128_COLRHT_PROD_WARP_BEGIN),
            )


class _Tcgen05GroupColRhtRequantize:
    """Per-expert columnwise NVFP4 requantization of ``W_qdq.bf16().t() @ R_n``.

    Expert-uniform ``(E, M, N)`` stack, expert = tile-derived, no group cap. The mainloop
    is ``_Tcgen05GroupColRhtRequantAmax``'s; eight epilogue warps (two per TMEM quadrant,
    four of a lane's eight blocks each) RTNE-quantize each tile from TMEM into the
    expert's ``(N, M//2)`` codes and swizzled scales. The accumulator
    discipline is Triton's: the epilogue zero-fills a TMEM stage after reading it (and
    both stages once before the first chain), and every UMMA accumulates, so an
    exact-zero sum keeps the sign Triton's zero-initialised ``tl.dot`` gives it.
    """

    @cute.jit
    def __call__(
        self,
        mCodes: cute.Tensor,  # row_fp4_w as flat u32 (E*M*N//8,)
        mSF: cute.Tensor,  # row_sf_w as flat u32 (E*M*N//64,)
        mAmax: cute.Tensor,  # global_amax (E,) f32
        mSign: cute.Tensor,  # dgrad_rht (128,) int8
        mB: cute.Tensor,  # H128 (128 j, 128 k, 1), K-major (symmetric)
        mAmaxT: cute.Tensor,  # amax_rht_w_qdq_t (E,) f32
        mOutCodes: cute.Tensor,  # (E*N*M//8,) u32 columnwise codes
        mOutSF: cute.Tensor,  # flat u32 view of the (E, N//128, M//64, 32, 16) e4m3 scales
        M: cutlass.Int32,
        N: cutlass.Int32,
        E: cutlass.Int32,
        num_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        k_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        mn_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.MN_SW128, cutlass.BFloat16
        )
        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        mma_col = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            RHT128_MMA_TILER_COL,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.MN,
            OperandMajorMode.K,
        )
        tiled_mma_col = cute.make_tiled_mma(cute.make_mma_atom(mma_col))
        a_shape = tiled_mma_col.partition_shape_A(
            cute.dice(RHT128_CTA_TILE_COL, (1, None, 1))
        )
        a_smem_layout_staged = tcgen05.tile_to_mma_shape(
            mn_atom, cute.append(a_shape, RHT128_COLRHT_STAGES), order=(1, 2, 3)
        )
        b_shape = tiled_mma_col.partition_shape_B(
            cute.dice(RHT128_CTA_TILE_COL, (None, 1, 1))
        )
        b_smem_layout = tcgen05.tile_to_mma_shape(
            k_atom, cute.append(b_shape, 1), order=(1, 2, 3)
        )
        tma_atom_b_col, tma_tensor_b_col = cute.nvgpu.make_tiled_tma_atom_B(
            g2s,
            mB,
            cute.slice_(b_smem_layout, (None, None, None, 0)),
            RHT128_CTA_TILE_COL,
            tiled_mma_col,
            (1, 1, 1, 1),
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma_col.thr_id.shape,)
        )
        tCtAcc_fake = tiled_mma_col.make_fragment_C(
            cute.append(
                tiled_mma_col.partition_shape_C((M_TILE, RHT128_DIM)),
                RHT128_ACC_STAGES,
            )
        )
        num_tmem_alloc_cols = sm100_utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        self.kernel(
            tiled_mma_col,
            tma_atom_b_col,
            tma_tensor_b_col,
            mCodes,
            mSF,
            mAmax,
            mSign,
            mAmaxT,
            mOutCodes,
            mOutSF,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            b_smem_layout,
            tCtAcc_fake.layout,
            num_tmem_alloc_cols,
            M,
            N,
            E,
        ).launch(
            grid=(num_ctas, 1, 1), block=(RHT128_COLRHT_REQ_TPB, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma_col: cute.TiledMma,
        tma_atom_b_col: cute.CopyAtom,
        mBcol: cute.Tensor,
        mCodes: cute.Tensor,
        mSF: cute.Tensor,
        mAmax: cute.Tensor,
        mSign: cute.Tensor,
        mAmaxT: cute.Tensor,
        mOutCodes: cute.Tensor,
        mOutSF: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        acc_fake_layout: cute.Layout,
        num_tmem_alloc_cols: cutlass.Constexpr,
        M: cutlass.Int32,
        N: cutlass.Int32,
        E: cutlass.Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        tiles_n = N // cutlass.Int32(RHT128_DIM)
        tiles_per_expert = (M // cutlass.Int32(RHT128_DIM)) * tiles_n
        t_begin, t_end = _static_tile_range(E * tiles_per_expert, bidx, gdim)
        n_my = t_end - t_begin

        if warp_idx == MMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_b_col)

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[cutlass.Int64, RHT128_COLRHT_STAGES * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, RHT128_ACC_STAGES * 2]
            b_mbar: cutlass.Int64
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.ab_mbar.data_ptr(),
            num_stages=RHT128_COLRHT_STAGES,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, RHT128_COLRHT_PROD_THREADS
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=RHT128_ACC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, RHT128_COLRHT_REQ_EPI_WARPS
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # MMA warp + the epilogue group retrieve the TMEM pointer; the same 160 threads
        # meet again once the epilogue has zero-filled both accumulator stages.
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_ALLOC_BAR, num_threads=32 + RHT128_COLRHT_REQ_EPI_THREADS
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_DEALLOC_BAR, num_threads=RHT128_COLRHT_REQ_EPI_THREADS
        )
        acc_zero_barrier = pipeline.NamedBarrier(
            barrier_id=RHT128_COLRHT_ACC_ZERO_BAR,
            num_threads=32 + RHT128_COLRHT_REQ_EPI_THREADS,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=RHT128_COLRHT_EPI_WARP_BEGIN,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        if warp_idx == MMA_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(storage.b_mbar.ptr, 1)
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # 1 KB alignment: the producer's closed-form swizzle assumes it of the stage base.
        raw_a = smem.allocate_array(
            cutlass.BFloat16,
            cute.cosize(a_smem_layout_staged.outer),
            byte_alignment=1024,
        )
        sA = cute.make_tensor(
            cute.recast_ptr(raw_a, a_smem_layout_staged.inner, dtype=cutlass.BFloat16),
            a_smem_layout_staged.outer,
        )
        sBcol = smem.allocate_tensor(
            cutlass.BFloat16, b_smem_layout.outer, 128, swizzle=b_smem_layout.inner
        )
        # Staging ring of the producers' codes and scale words (``cp.async`` targets).
        raw_stg = smem.allocate_array(
            cutlass.Uint32,
            RHT128_COLRHT_LOAD_STAGES * RHT128_COLRHT_LOAD_TILE_BYTES // 4,
            byte_alignment=16,
        )

        cta_layout = cute.make_layout((1,))
        thr_col = tiled_mma_col.get_slice(0)
        gBcol = cute.local_tile(
            mBcol, cute.slice_(RHT128_CTA_TILE_COL, (0, None, None)), (None, None, None)
        )
        tBsBcol, tBgBcol = cpasync.tma_partition(
            tma_atom_b_col,
            0,
            cta_layout,
            cute.group_modes(sBcol, 0, 3),
            cute.group_modes(thr_col.partition_B(gBcol), 0, 3),
        )
        tCrA_col = tiled_mma_col.make_fragment_A(sA)
        tCrB_col = tiled_mma_col.make_fragment_B(sBcol)

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ==================== MMA warp (+ one-shot B TMA) ====================
        if warp_idx == MMA_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    storage.b_mbar.ptr, RHT128_B_BYTES
                )
            cute.copy(
                tma_atom_b_col,
                tBgBcol[(None, 0, 0, 0)],
                tBsBcol[(None, 0)],
                tma_bar_ptr=storage.b_mbar.ptr,
            )
            tmem.wait_for_alloc()
            tCtAcc = cute.make_tensor(
                tmem.retrieve_ptr(cutlass.Float32), acc_fake_layout
            )
            acc_zero_barrier.arrive_and_wait()
            cute.arch.mbarrier_wait(storage.b_mbar.ptr, 0)
            ab_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, RHT128_COLRHT_STAGES
            )
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, RHT128_ACC_STAGES
            )
            for i in cutlass.range(n_my, unroll=1):
                ab_pipeline.consumer_wait(ab_state)
                acc_pipeline.producer_acquire(acc_state)
                acc_col = tCtAcc[(None, None, None, acc_state.index)]
                for kb in cutlass.range_constexpr(RHT128_K_BLOCKS):
                    # Every step accumulates onto the epilogue's zero fill (Triton's
                    # ``tl.dot`` discipline); never ``kb > 0`` without that fill.
                    tiled_mma_col.set(tcgen05.Field.ACCUMULATE, True)
                    cute.gemm(
                        tiled_mma_col,
                        acc_col,
                        tCrA_col[(None, None, kb, ab_state.index)],
                        tCrB_col[(None, None, kb, 0)],
                        acc_col,
                    )
                acc_pipeline.producer_commit(acc_state)
                acc_state.advance()
                ab_pipeline.consumer_release(ab_state)
                ab_state.advance()
            acc_pipeline.producer_tail(acc_state)

        # ==================== epilogue: quantize W_qdq.t() @ R_n per expert ====================
        if (
            warp_idx >= RHT128_COLRHT_EPI_WARP_BEGIN
            and warp_idx < RHT128_COLRHT_REQ_EPI_WARP_END
        ):
            tmem.allocate(num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc = cute.make_tensor(tmem_ptr, acc_fake_layout)
            # Warps 1-4 take blocks 0-3 of their quadrant's lanes, warps 5-8 blocks 4-7.
            u_base = (
                (warp_idx - cutlass.Int32(RHT128_COLRHT_EPI_WARP_BEGIN))
                // cutlass.Int32(4)
            ) * cutlass.Int32(RHT128_COLRHT_REQ_BLOCKS_PER_WARP)
            for s in cutlass.range_constexpr(RHT128_ACC_STAGES):
                _rht128_zero_acc(
                    tCtAcc[(None, None, None, s)],
                    tidx % cutlass.Int32(RHT128_DIM),
                    u_base,
                )
            cute.arch.fence_view_async_tmem_store()
            acc_zero_barrier.arrive_and_wait()
            _rht128_colrht_requantize_epilogue(
                mOutCodes,
                mOutSF,
                mAmaxT,
                tCtAcc,
                acc_pipeline,
                t_begin,
                t_end,
                tiles_per_expert,
                tiles_n,
                M,
                N,
                tidx % cutlass.Int32(RHT128_DIM),
                u_base,
            )
            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ==================== producers: W_qdq tiles into the A ring ====================
        if warp_idx >= RHT128_COLRHT_REQ_PROD_WARP_BEGIN:
            _rht128_dequant_a_producer(
                mCodes,
                mSF,
                mAmax,
                mSign,
                raw_a.toint(),
                cutlass.Uint32(raw_stg.toint()),
                ab_pipeline,
                t_begin,
                n_my,
                tiles_per_expert,
                tiles_n,
                M,
                N,
                tidx - cutlass.Int32(32 * RHT128_COLRHT_REQ_PROD_WARP_BEGIN),
            )


@functools.lru_cache(maxsize=None)
def _compile_group_col_rht_requant_amax_kernel(device_idx: int):
    """Compile the grouped columnwise-RHT weight requantization amax kernel.

    The codes and scales arrive as flat u32 views with ``assumed_align=16`` (every code
    row is a multiple of 64 B, so a thread's two 16 B code loads are aligned); ``H128``
    as the ``(128, 128, 1)`` K-major operand the amax kernel above takes.
    """
    free = cute.sym_int
    k = _Tcgen05GroupColRhtRequantAmax()
    return cute.compile(
        k,
        make_fake_tensor(cutlass.Uint32, (free(),), stride=(1,), assumed_align=16),
        make_fake_tensor(cutlass.Uint32, (free(),), stride=(1,), assumed_align=16),
        make_fake_tensor(cutlass.Float32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Int8, (RHT128_DIM,), stride=(1,)),
        make_fake_tensor(
            cutlass.BFloat16, (RHT128_DIM, RHT128_DIM, 1), stride=(RHT128_DIM, 1, 1)
        ),
        make_fake_tensor(cutlass.Float32, (free(),), stride=(1,)),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _cutedsl_group_col_rht_requant_amax_impl(
    row_fp4_w: torch.Tensor,
    row_sf_w: torch.Tensor,
    global_amax: torch.Tensor,
    dgrad_rht: torch.Tensor,
    h128: torch.Tensor,
    num_tensors: int,
) -> torch.Tensor:
    """Per-expert ``max|W_qdq.bf16().t() @ R_n|`` from the rowwise NVFP4 weight.

    ``row_fp4_w`` is ``(E, M, N//2)`` uint8, ``row_sf_w`` the ``(E, M//128, N//64, 32, 16)``
    swizzled e4m3 scales, ``global_amax`` the ``(E,)`` decode amax, ``dgrad_rht`` the
    ``(128,)`` int8 sign vector and ``h128`` the cached ``(128, 128)`` bfloat16 Hadamard
    (symmetric, so it is the K-major ``(N, K)`` UMMA operand as is; the signs reach the
    kernel through the decode scale). Returns ``(E,)`` float32, zero-initialised because
    the epilogue accumulates with atomic max.
    """
    E, M, packed_N = row_fp4_w.shape
    N = packed_N * 2
    dev = row_fp4_w.device

    amax_rht_w_qdq_t = torch.zeros((E,), dtype=torch.float32, device=dev)
    # An empty stack passes validation and has nothing to reduce: a zero grid is a
    # launch error here.
    if row_fp4_w.numel() == 0:
        return amax_rht_w_qdq_t
    tiles = E * (M // RHT128_DIM) * (N // RHT128_DIM)
    num_ctas = max(1, min(tiles, _get_num_sms(dev.index)))

    stream = cuda.CUstream(int(torch.cuda.current_stream(dev).cuda_stream))
    _compile_group_col_rht_requant_amax_kernel(dev.index)(
        row_fp4_w.view(torch.uint32).view(-1),
        row_sf_w.view(torch.uint32).view(-1),
        global_amax,
        dgrad_rht,
        h128.unsqueeze(-1),
        amax_rht_w_qdq_t,
        int(M),
        int(N),
        int(E),
        int(num_ctas),
        stream,
    )
    return amax_rht_w_qdq_t


@functools.lru_cache(maxsize=None)
def _compile_group_col_rht_requantize_kernel(device_idx: int):
    """Compile the grouped columnwise-RHT weight requantize kernel (see the amax
    compile above for the operand forms; the codes out are the same flat u32 view)."""
    free = cute.sym_int
    k = _Tcgen05GroupColRhtRequantize()
    return cute.compile(
        k,
        make_fake_tensor(cutlass.Uint32, (free(),), stride=(1,), assumed_align=16),
        make_fake_tensor(cutlass.Uint32, (free(),), stride=(1,), assumed_align=16),
        make_fake_tensor(cutlass.Float32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Int8, (RHT128_DIM,), stride=(1,)),
        make_fake_tensor(
            cutlass.BFloat16, (RHT128_DIM, RHT128_DIM, 1), stride=(RHT128_DIM, 1, 1)
        ),
        make_fake_tensor(cutlass.Float32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Uint32, (free(),), stride=(1,), assumed_align=16),
        make_fake_tensor(cutlass.Uint32, (free(),), stride=(1,), assumed_align=16),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _cutedsl_group_col_rht_requantize_impl(
    row_fp4_w: torch.Tensor,
    row_sf_w: torch.Tensor,
    global_amax: torch.Tensor,
    amax_rht_w_qdq_t: torch.Tensor,
    dgrad_rht: torch.Tensor,
    h128: torch.Tensor,
    num_tensors: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-expert columnwise NVFP4 requantization of ``W_qdq.bf16().t() @ R_n``.

    Operands as ``_cutedsl_group_col_rht_requant_amax_impl`` plus ``amax_rht_w_qdq_t``,
    that op's ``(E,)`` output. Returns ``(E, N, M//2)`` uint8 codes (a view of the u32
    buffer the kernel writes) and ``(E, N//128, M//64, 32, 16)`` float8_e4m3fn swizzled
    scales.
    """
    E, M, packed_N = row_fp4_w.shape
    N = packed_N * 2
    dev = row_fp4_w.device

    col_fp4 = torch.empty((E, N, M // 8), dtype=torch.uint32, device=dev)
    col_sf = torch.empty(
        (E, N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev
    )
    # An empty stack passes validation and has nothing to quantize: a zero grid is a
    # launch error here.
    if row_fp4_w.numel() == 0:
        return col_fp4.view(torch.uint8), col_sf
    tiles = E * (M // RHT128_DIM) * (N // RHT128_DIM)
    num_ctas = max(1, min(tiles, _get_num_sms(dev.index)))

    stream = cuda.CUstream(int(torch.cuda.current_stream(dev).cuda_stream))
    _compile_group_col_rht_requantize_kernel(dev.index)(
        row_fp4_w.view(torch.uint32).view(-1),
        row_sf_w.view(torch.uint32).view(-1),
        global_amax,
        dgrad_rht,
        h128.unsqueeze(-1),
        amax_rht_w_qdq_t,
        col_fp4.view(-1),
        col_sf.view(torch.uint32).view(-1),
        int(M),
        int(N),
        int(E),
        int(num_ctas),
        stream,
    )
    # uint32 -> uint8 quadruples the last extent: (E, N, M//8) -> (E, N, M//2).
    return col_fp4.view(torch.uint8), col_sf


# --- grouped rowwise cast + columnwise-RHT amax (RHT-128 on the token axis) ---
# The dynamic-RHT path of ``cutedsl_group_rht_amax``: ``_Tcgen05GroupRowRhtColRhtAmax``'s
# col chain against one resident ``R^T`` tile -- the cached ``H128`` TMA'd once and
# sign-flipped in SMEM by warps 2-3 -- with the RHT-16 amax kernel's
# SMEM row epilogue reading the same A stage through the plain (hidden, token, stage)
# view -- so the stage has two consumer groups (``PipelineTmaMultiConsumersAsync``), as
# in the RHT-16 kernels. Warps: 0 MMA, 1 TMA, 2-3 sign, 4-7 col (TMEM), 8-15 row (SMEM).
RHT128_ROWCAST_MAINLOOP_STAGES = (
    _SMEM_CAPACITY - _SMEM_RESERVE - RHT128_B_BYTES
) // _A_TILE_BYTES  # 6
RHT128_ROWCAST_SIGN_WARP_BEGIN = 2
RHT128_ROWCAST_SIGN_WARP_END = 4
RHT128_ROWCAST_B_SIGN_BAR = 3
RHT128_ROWCAST_COL_WARP_BEGIN = 4
RHT128_ROWCAST_COL_WARP_END = 8
RHT128_ROWCAST_ROW_WARP_BEGIN = 8
RHT128_ROWCAST_ROW_WARP_END = 16
RHT128_ROWCAST_N_WARPS = 16
RHT128_ROWCAST_TPB = 32 * RHT128_ROWCAST_N_WARPS
RHT128_ROWCAST_COL_THREADS = 32 * (
    RHT128_ROWCAST_COL_WARP_END - RHT128_ROWCAST_COL_WARP_BEGIN
)
RHT128_ROWCAST_ROW_THREADS = 32 * (
    RHT128_ROWCAST_ROW_WARP_END - RHT128_ROWCAST_ROW_WARP_BEGIN
)
RHT128_ROWCAST_SIGN_THREADS = 32 * (
    RHT128_ROWCAST_SIGN_WARP_END - RHT128_ROWCAST_SIGN_WARP_BEGIN
)
RHT128_ROWCAST_ACC_CONSUMER_WARPS = (
    RHT128_ROWCAST_COL_WARP_END - RHT128_ROWCAST_COL_WARP_BEGIN
)


@dsl_user_op
def _max_abs_bf16x2_x8(
    w0: cutlass.Uint32,
    w1: cutlass.Uint32,
    w2: cutlass.Uint32,
    w3: cutlass.Uint32,
    w4: cutlass.Uint32,
    w5: cutlass.Uint32,
    w6: cutlass.Uint32,
    w7: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint32:
    """Per-half ``max|.|`` of eight packed bf16 pairs, as a packed pair with junk signs.

    ``max.NaN.xorsign.abs.bf16x2`` keeps the larger magnitude of each half exactly (a
    comparison, no rounding), makes any NaN input a NaN as ``max.NaN.f32`` does, and
    sets the result's sign to the XOR of the input signs -- junk that
    ``_bf16x2_amax_to_f32`` masks off. One ``HMNMX2`` per word replaces the widen pair
    plus the f32 max of the scalar path.
    """
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [w.ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3, w4, w5, w6, w7)],
            (
                "{\n"
                ".reg .b32 m0, m1, m2, m3;\n"
                "max.NaN.xorsign.abs.bf16x2 m0, $1, $2;\n"
                "max.NaN.xorsign.abs.bf16x2 m1, $3, $4;\n"
                "max.NaN.xorsign.abs.bf16x2 m2, $5, $6;\n"
                "max.NaN.xorsign.abs.bf16x2 m3, $7, $8;\n"
                "max.NaN.xorsign.abs.bf16x2 m0, m0, m1;\n"
                "max.NaN.xorsign.abs.bf16x2 m2, m2, m3;\n"
                "max.NaN.xorsign.abs.bf16x2 $0, m0, m2;\n"
                "}"
            ),
            "=r,r,r,r,r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _bf16x2_amax_to_f32(
    m0: cutlass.Uint32,
    m1: cutlass.Uint32,
    m2: cutlass.Uint32,
    m3: cutlass.Uint32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Float32:
    """Fold four packed magnitude pairs into the f32 ``max|.|`` the scalar path's
    ``max.NaN.f32`` chain yields over the same values: the mask clears the junk signs,
    the shift and the mask widen both halves exactly (see ``_bf16lo_to_f32``)."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [m.ir_value(loc=loc, ip=ip) for m in (m0, m1, m2, m3)],
            (
                "{\n"
                ".reg .b32 a, b, lo, hi;\n"
                ".reg .f32 fl, fh;\n"
                "max.NaN.xorsign.abs.bf16x2 a, $1, $2;\n"
                "max.NaN.xorsign.abs.bf16x2 b, $3, $4;\n"
                "max.NaN.xorsign.abs.bf16x2 a, a, b;\n"
                "and.b32 a, a, 0x7fff7fff;\n"
                "shl.b32 lo, a, 16;\n"
                "and.b32 hi, a, 0xffff0000;\n"
                "mov.b32 fl, lo;\n"
                "mov.b32 fh, hi;\n"
                "max.NaN.f32 $0, fl, fh;\n"
                "}"
            ),
            "=f,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _rowcast_tile_amax(rWords):
    """``max|A|`` over a row thread's share of one tile -- ``ROW_PASSES`` tokens, two 16 B
    chunks each, held as u32 word views -- as f32."""
    m = []
    for p in range(ROW_PASSES):
        a, b = rWords[2 * p], rWords[2 * p + 1]
        m.append(_max_abs_bf16x2_x8(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]))
    return _bf16x2_amax_to_f32(m[0], m[1], m[2], m[3])


@dsl_user_op
def _st_release_gpu_u32(
    addr: cutlass.Pointer, val: cutlass.Uint32, *, loc=None, ip=None
):
    """Release-store one u32 at GPU scope: every prior store of this thread is visible to
    a thread that acquire-loads the value."""
    llvm.inline_asm(
        None,
        [addr.llvm_ptr, val.ir_value(loc=loc, ip=ip)],
        "st.release.gpu.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _ld_acquire_gpu_u32(addr: cutlass.Pointer, *, loc=None, ip=None) -> cutlass.Uint32:
    """Acquire-load one u32 at GPU scope (the pair of ``_st_release_gpu_u32``)."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [addr.llvm_ptr],
            "ld.acquire.gpu.global.b32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _bf16x2_sign_mask(sign_t, k):
    """XOR mask that negates the bf16 pair ``(k, k + 1)`` where ``sign_t`` is negative."""
    lo = sign_t[k] < cutlass.Int8(0)
    hi = sign_t[k + cutlass.Int32(1)] < cutlass.Int8(0)
    return cutlass.Uint32(
        cutlass.select_(lo, cutlass.Uint32(0x8000), cutlass.Uint32(0))
    ) | cutlass.Uint32(
        cutlass.select_(hi, cutlass.Uint32(0x80000000), cutlass.Uint32(0))
    )


class _Tcgen05GroupRowCastColRhtAmax:
    """Per-group ``max|A.t() @ R|`` and ``max|A|`` in one pass over ``A``.

    Standalone (no ``_GroupRhtMainloop``): every 128x128 tile is TMA'd once and
    feeds one UMMA chain against the resident K-major ``R^T`` tile -- the col chain
    of ``_Tcgen05GroupRowRhtColRhtAmax``, its operand the TMA'd ``H128`` whose
    columns the sign warps negate in SMEM (``R^T[j, k] = H[j, k] * s_k``, a bf16
    sign-bit flip) -- while the row warps read the same bytes through the RHT-16
    kernels' plain ``(hidden, token, stage)`` view and reduce the raw ``max|A|``.
    Static persistent grid: CTA ``b`` owns the contiguous tile chunk
    ``_static_tile_range``, tiles below ``logical_packed_length`` only,
    hidden-fastest.
    """

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # A.t().unsqueeze(-1): (hidden, tokens, 1)
        mB: cute.Tensor,  # H128 (128 j, 128 k, 1), K-major (symmetric)
        mSign: cute.Tensor,  # wgrad_rht (128,) int8
        col_amax_t: cute.Tensor,  # (num_tensors + 1,) f32, pre-zeroed; [-1] = row flag
        row_amax_t: cute.Tensor,  # (num_tensors,) f32, zeroed by CTA 0 behind the flag
        offsets_t: cute.Tensor,
        logical_len_t: cute.Tensor,
        hidden: cutlass.Int32,
        num_tensors: cutlass.Int32,
        num_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        k_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        mn_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.MN_SW128, cutlass.BFloat16
        )
        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        mma_col = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            RHT128_MMA_TILER_COL,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.MN,
            OperandMajorMode.K,
        )
        tiled_mma_col = cute.make_tiled_mma(cute.make_mma_atom(mma_col))
        a_shape = tiled_mma_col.partition_shape_A(
            cute.dice(RHT128_CTA_TILE_COL, (1, None, 1))
        )
        a_smem_layout_staged = tcgen05.tile_to_mma_shape(
            mn_atom,
            cute.append(a_shape, RHT128_ROWCAST_MAINLOOP_STAGES),
            order=(1, 2, 3),
        )
        # Same bytes, plain (hidden, token, stage) grouping for the row warps.
        a_clean_layout = cute.tile_to_shape(
            mn_atom,
            (M_TILE, TOKEN_TILE, RHT128_ROWCAST_MAINLOOP_STAGES),
            order=(0, 1, 2),
        )
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            g2s,
            mA,
            cute.slice_(a_smem_layout_staged, (None, None, None, 0)),
            RHT128_CTA_TILE_COL,
            tiled_mma_col,
            (1, 1, 1, 1),
        )
        b_shape = tiled_mma_col.partition_shape_B(
            cute.dice(RHT128_CTA_TILE_COL, (None, 1, 1))
        )
        b_smem_layout = tcgen05.tile_to_mma_shape(
            k_atom, cute.append(b_shape, 1), order=(1, 2, 3)
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            g2s,
            mB,
            cute.slice_(b_smem_layout, (None, None, None, 0)),
            RHT128_CTA_TILE_COL,
            tiled_mma_col,
            (1, 1, 1, 1),
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma_col.thr_id.shape,)
        )
        tCtAcc_fake = tiled_mma_col.make_fragment_C(
            cute.append(
                tiled_mma_col.partition_shape_C((M_TILE, RHT128_DIM)),
                2 * RHT128_ACC_STAGES,
            )
        )
        num_tmem_alloc_cols = sm100_utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        self.kernel(
            tiled_mma_col,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            mSign,
            col_amax_t,
            row_amax_t,
            offsets_t,
            logical_len_t,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            a_clean_layout,
            b_smem_layout,
            tCtAcc_fake.layout,
            num_tmem_alloc_cols,
            hidden,
            num_tensors,
        ).launch(grid=(num_ctas, 1, 1), block=(RHT128_ROWCAST_TPB, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        tiled_mma_col: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        mSign: cute.Tensor,
        col_amax_t: cute.Tensor,
        row_amax_t: cute.Tensor,
        offsets_t: cute.Tensor,
        logical_len_t: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        a_clean_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        acc_fake_layout: cute.Layout,
        num_tmem_alloc_cols: cutlass.Constexpr,
        hidden: cutlass.Int32,
        num_tensors: cutlass.Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % cutlass.Int32(32)
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        tiles_in_m = hidden // cutlass.Int32(M_TILE)
        tiles_in_n_valid = logical_len_t[0] // cutlass.Int32(TOKEN_TILE)
        t_begin, t_end = _static_tile_range(tiles_in_m * tiles_in_n_valid, bidx, gdim)
        n_my = t_end - t_begin

        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[
                cutlass.Int64, RHT128_ROWCAST_MAINLOOP_STAGES * 2
            ]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, RHT128_ACC_STAGES * 2]
            b_mbar: cutlass.Int64
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mainloop: one TMA producer, two heterogeneous consumer groups on the
        # same stage -- the UMMA (released via tcgen05.commit) and the 256 row
        # threads (released via a plain mbarrier arrive).
        ab_pipeline = pipeline.PipelineTmaMultiConsumersAsync.create(
            barrier_storage=storage.ab_mbar.data_ptr(),
            num_stages=RHT128_ROWCAST_MAINLOOP_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_umma=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group_async=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, RHT128_ROWCAST_ROW_THREADS
            ),
            tx_count=_A_TILE_BYTES,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=RHT128_ACC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, RHT128_ROWCAST_ACC_CONSUMER_WARPS
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # MMA warp + the col epilogue group retrieve the TMEM pointer.
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_ALLOC_BAR, num_threads=32 + RHT128_ROWCAST_COL_THREADS
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_DEALLOC_BAR, num_threads=RHT128_ROWCAST_COL_THREADS
        )
        # The sign warps hand the negated B tile to the MMA warp.
        b_sign_barrier = pipeline.NamedBarrier(
            barrier_id=RHT128_ROWCAST_B_SIGN_BAR,
            num_threads=32 + RHT128_ROWCAST_SIGN_THREADS,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=RHT128_ROWCAST_COL_WARP_BEGIN,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        if warp_idx == TMA_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(storage.b_mbar.ptr, 1)
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # The UMMA reads the A stage through its swizzled MN-major layout; the
        # row warps see a plain (hidden, token, stage) view of the same bytes.
        raw_a = smem.allocate_array(
            cutlass.BFloat16,
            cute.cosize(a_smem_layout_staged.outer),
            byte_alignment=128,
        )
        swz_ptr = cute.recast_ptr(
            raw_a, a_smem_layout_staged.inner, dtype=cutlass.BFloat16
        )
        sA = cute.make_tensor(swz_ptr, a_smem_layout_staged.outer)
        sA_clean = cute.make_tensor(swz_ptr, a_clean_layout.outer)
        sB = smem.allocate_tensor(
            cutlass.BFloat16, b_smem_layout.outer, 128, swizzle=b_smem_layout.inner
        )
        # The same bytes as a plain (j, k % 64, k // 64) view for the sign warps: the
        # K-major SW128 tile is two 16 KB halves of 128 rows x 64 k.
        sB_clean = cute.make_tensor(
            sB.iterator,
            cute.make_layout(
                (RHT128_DIM, RHT128_DIM // 2, 2), stride=(RHT128_DIM // 2, 1, 8192)
            ),
        )

        cta_layout = cute.make_layout((1,))
        thr_col = tiled_mma_col.get_slice(0)
        gA = cute.local_tile(
            mA, cute.slice_(RHT128_CTA_TILE_COL, (None, 0, None)), (None, None, None)
        )
        gB = cute.local_tile(
            mB, cute.slice_(RHT128_CTA_TILE_COL, (0, None, None)), (None, None, None)
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_col.partition_A(gA), 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(thr_col.partition_B(gB), 0, 3),
        )
        tCrA_col = tiled_mma_col.make_fragment_A(sA)
        tCrB_col = tiled_mma_col.make_fragment_B(sB)

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ==================== TMA warp ====================
        if warp_idx == TMA_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    storage.b_mbar.ptr, RHT128_B_BYTES
                )
            cute.copy(
                tma_atom_b,
                tBgB[(None, 0, 0, 0)],
                tBsB[(None, 0)],
                tma_bar_ptr=storage.b_mbar.ptr,
            )
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, RHT128_ROWCAST_MAINLOOP_STAGES
            )
            for i in cutlass.range(n_my, unroll=1):
                t = t_begin + i
                tile_n = t // tiles_in_m
                tile_m = t - tile_n * tiles_in_m
                ab_pipeline.producer_acquire(ab_producer_state)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, tile_m, tile_n, 0)],
                    tAsA[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                )
                ab_producer_state.advance()
            ab_pipeline.producer_tail(ab_producer_state)

        # ==================== sign warps: R^T = H128 * s_k in SMEM ====================
        if (
            warp_idx >= RHT128_ROWCAST_SIGN_WARP_BEGIN
            and warp_idx < RHT128_ROWCAST_SIGN_WARP_END
        ):
            # Thread ``t`` owns one 16 B chunk of 32 rows: k in [8 c, 8 c + 8) of rows
            # j0 .. j0 + 31, so a quarter-warp reads eight distinct chunks of one row
            # (conflict-free) and needs only its eight signs, which fold into one XOR
            # mask per bf16 pair; a negative sign flips the bf16 sign bit, exactly
            # ``H[j, k] * -1``. The masks are built before the B tile lands.
            t = tidx - RHT128_ROWCAST_SIGN_WARP_BEGIN * cutlass.Int32(32)
            # CTA 0 zeroes the row buffer and publishes it behind the flag that shares
            # the col buffer's fill; the row warps of every CTA acquire it before their
            # first flush. CTA 0 is dispatched no later than any other CTA and waits on
            # nothing here, so the grid makes progress.
            if bidx == cutlass.Int32(0) and t == cutlass.Int32(0):
                for e in cutlass.range(num_tensors, unroll=1):
                    row_amax_t[e] = cutlass.Float32(0.0)
                _st_release_gpu_u32(
                    col_amax_t.iterator + num_tensors, cutlass.Uint32(1)
                )
            c = t % cutlass.Int32(16)
            j0 = (t // cutlass.Int32(16)) * cutlass.Int32(32)
            k0 = c * cutlass.Int32(8)
            m0 = _bf16x2_sign_mask(mSign, k0)
            m1 = _bf16x2_sign_mask(mSign, k0 + cutlass.Int32(2))
            m2 = _bf16x2_sign_mask(mSign, k0 + cutlass.Int32(4))
            m3 = _bf16x2_sign_mask(mSign, k0 + cutlass.Int32(6))
            half = c // cutlass.Int32(8)
            chunk = c % cutlass.Int32(8)
            rRows = []
            rWords = []
            for r in cutlass.range_constexpr(8):
                rRows.append(cute.make_rmem_tensor((8,), cutlass.BFloat16))
                rWords.append(cute.recast_tensor(rRows[r], cutlass.Uint32))
            cute.arch.mbarrier_wait(storage.b_mbar.ptr, 0)
            for b in cutlass.range_constexpr(4):
                for r in cutlass.range_constexpr(8):
                    j = j0 + cutlass.Int32(b * 8 + r)
                    cute.autovec_copy(
                        cute.local_tile(sB_clean[(j, None, half)], (8,), (chunk,)),
                        rRows[r],
                    )
                for r in cutlass.range_constexpr(8):
                    rWords[r][0] = rWords[r][0] ^ m0
                    rWords[r][1] = rWords[r][1] ^ m1
                    rWords[r][2] = rWords[r][2] ^ m2
                    rWords[r][3] = rWords[r][3] ^ m3
                for r in cutlass.range_constexpr(8):
                    j = j0 + cutlass.Int32(b * 8 + r)
                    cute.autovec_copy(
                        rRows[r],
                        cute.local_tile(sB_clean[(j, None, half)], (8,), (chunk,)),
                    )
            # Generic-proxy stores -> visible to the UMMA (async proxy) before the arrive.
            cute.arch.fence_proxy("async.shared", space="cta")
            b_sign_barrier.arrive_and_wait()

        # ==================== MMA warp ====================
        if warp_idx == MMA_WARP:
            tmem.wait_for_alloc()
            tCtAcc = cute.make_tensor(
                tmem.retrieve_ptr(cutlass.Float32), acc_fake_layout
            )
            b_sign_barrier.arrive_and_wait()
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, RHT128_ROWCAST_MAINLOOP_STAGES
            )
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, RHT128_ACC_STAGES
            )
            for i in cutlass.range(n_my, unroll=1):
                ab_pipeline.consumer_wait(ab_consumer_state)
                acc_pipeline.producer_acquire(acc_state)
                acc_col = tCtAcc[(None, None, None, 2 * acc_state.index + 0)]
                for kb in cutlass.range_constexpr(RHT128_K_BLOCKS):
                    tiled_mma_col.set(tcgen05.Field.ACCUMULATE, kb > 0)
                    cute.gemm(
                        tiled_mma_col,
                        acc_col,
                        tCrA_col[(None, None, kb, ab_consumer_state.index)],
                        tCrB_col[(None, None, kb, 0)],
                        acc_col,
                    )
                acc_pipeline.producer_commit(acc_state)
                acc_state.advance()
                ab_pipeline.consumer_release(
                    ab_consumer_state, pipeline.PipelineOp.TCGen05Mma
                )
                ab_consumer_state.advance()
            acc_pipeline.producer_tail(acc_state)

        # ==================== col epilogue: max|A.t() @ R| ====================
        if (
            warp_idx >= RHT128_ROWCAST_COL_WARP_BEGIN
            and warp_idx < RHT128_ROWCAST_COL_WARP_END
        ):
            tmem.allocate(num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc = cute.make_tensor(tmem_ptr, acc_fake_layout)
            _rht128_amax_epilogue(
                0,
                col_amax_t,
                tCtAcc,
                acc_pipeline,
                offsets_t,
                num_tensors,
                t_begin,
                t_end,
                tiles_in_m,
                tidx,
                lane,
            )
            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ==================== Rowwise amax (raw A, from SMEM) ====================
        if (
            warp_idx >= RHT128_ROWCAST_ROW_WARP_BEGIN
            and warp_idx < RHT128_ROWCAST_ROW_WARP_END
        ):
            r_local = tidx - RHT128_ROWCAST_ROW_WARP_BEGIN * cutlass.Int32(32)
            hb = r_local % cutlass.Int32(ROW_HB)
            t0 = r_local // cutlass.Int32(ROW_HB)

            # Lane ``hb`` reads 16 B chunk ``hb`` of both 128 B swizzle rows of its token
            # (hidden [8 hb, 8 hb + 8) and [64 + 8 hb, 64 + 8 hb + 8)), so a quarter-warp
            # covers one whole row per LDS.128, conflict-free; all eight loads of a tile
            # issue before the first max and the stage is released before the reduction.
            rBlk = []
            rWords = []
            for i in cutlass.range_constexpr(2 * ROW_PASSES):
                rBlk.append(cute.make_rmem_tensor((8,), cutlass.BFloat16))
                rWords.append(cute.recast_tensor(rBlk[i], cutlass.Uint32))
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, RHT128_ROWCAST_MAINLOOP_STAGES
            )
            g = _group_idx(
                (t_begin // tiles_in_m) * cutlass.Int32(TOKEN_TILE),
                offsets_t,
                num_tensors,
            )
            g_end = offsets_t[g]
            run_max = cutlass.Float32(0.0)
            ready = _ld_acquire_gpu_u32(col_amax_t.iterator + num_tensors)
            while ready == cutlass.Uint32(0):
                ready = _ld_acquire_gpu_u32(col_amax_t.iterator + num_tensors)
            for i in cutlass.range(n_my, unroll=1):
                token = ((t_begin + i) // tiles_in_m) * cutlass.Int32(TOKEN_TILE)
                if token >= g_end:
                    _flush_group_max(run_max, row_amax_t, g, lane)
                    run_max = cutlass.Float32(0.0)
                    g = _group_idx(token, offsets_t, num_tensors)
                    g_end = offsets_t[g]
                ab_pipeline.consumer_wait(ab_consumer_state)
                stage = ab_consumer_state.index
                for p in cutlass.range_constexpr(ROW_PASSES):
                    tok = p * cutlass.Int32(ROW_TOK_PER_PASS) + t0
                    for r in cutlass.range_constexpr(2):
                        cute.autovec_copy(
                            cute.local_tile(
                                sA_clean[(None, tok, stage)], (8,), (r * ROW_HB + hb,)
                            ),
                            rBlk[2 * p + r],
                        )
                ab_pipeline.consumer_release(
                    ab_consumer_state, pipeline.PipelineOp.AsyncThread
                )
                ab_consumer_state.advance()

                run_max = _max_f32(run_max, _rowcast_tile_amax(rWords))
            _flush_group_max(run_max, row_amax_t, g, lane)


@functools.lru_cache(maxsize=None)
def _compile_group_row_cast_col_rht_amax_kernel(device_idx: int):
    """Compile the grouped row-cast + col-RHT amax kernel with symbolic shapes."""
    free = cute.sym_int
    h_sym = cute.sym_int(divisibility=M_TILE)
    t_sym = cute.sym_int(divisibility=TOKEN_TILE)
    k = _Tcgen05GroupRowCastColRhtAmax()
    return cute.compile(
        k,
        make_fake_tensor(cutlass.BFloat16, (h_sym, t_sym, 1), stride=(1, free(), 1)),
        make_fake_tensor(
            cutlass.BFloat16, (RHT128_DIM, RHT128_DIM, 1), stride=(RHT128_DIM, 1, 1)
        ),
        make_fake_tensor(cutlass.Int8, (RHT128_DIM,), stride=(1,)),
        make_fake_tensor(cutlass.Float32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Float32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Int32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Int32, (1,), stride=(1,)),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _cutedsl_group_row_cast_col_rht_amax_impl(
    A: torch.Tensor,
    wgrad_rht: torch.Tensor,
    h128: torch.Tensor,
    offsets: torch.Tensor,
    num_tensors: int,
    logical_packed_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-group ``max|A.t() @ R|`` and ``max|A|``.

    ``A`` is ``(tokens, hidden)`` bfloat16 row-major; ``wgrad_rht`` is the ``(128,)``
    int8 sign vector and ``h128`` the cached ``(128, 128)`` bfloat16 Hadamard
    (symmetric, so it is the K-major ``(N, K)`` UMMA operand as is; the signs reach
    the kernel's SMEM copy of it). Returns ``(col_amax, row_amax)``, each
    ``(num_tensors,)`` float32, in ``cutedsl_group_rht_amax``'s order. The buffers
    start at zero because the epilogues accumulate with atomic max: one fill zeroes
    the col buffer together with the flag behind which CTA 0 zeroes the row buffer.
    """
    tokens, hidden = A.shape
    dev = A.device
    A = A.detach()

    col_amax_flag = torch.zeros((num_tensors + 1,), dtype=torch.float32, device=dev)
    col_amax = col_amax_flag[:num_tensors]
    row_amax = torch.empty((num_tensors,), dtype=torch.float32, device=dev)
    if logical_packed_length is None:
        logical_packed_length = offsets[-1:]
    # See the fused kernel: the entry point requires byte_offset==0.
    logical_packed_length = logical_packed_length.clone()
    tiles = (hidden // M_TILE) * (tokens // TOKEN_TILE)
    num_ctas = max(1, min(tiles, _get_num_sms(dev.index)))

    stream = cuda.CUstream(int(torch.cuda.current_stream(dev).cuda_stream))
    _compile_group_row_cast_col_rht_amax_kernel(dev.index)(
        A.t().unsqueeze(-1),
        h128.unsqueeze(-1),
        wgrad_rht,
        col_amax_flag,
        row_amax,
        offsets,
        logical_packed_length,
        int(hidden),
        int(num_tensors),
        int(num_ctas),
        stream,
    )
    return col_amax, row_amax


# --- grouped rowwise cast + columnwise-RHT quantize (RHT-128 on the token axis) ---
# The dynamic-RHT path of ``cutedsl_group_rht_quantize_row_col``: the amax kernel above
# with the MS-EDEN warp layout (eight col warps, two per TMEM quadrant, splitting a lane's
# 8 blocks into 0-3 and 4-7) and the requantize kernel's accumulator discipline (the
# epilogue zero-fills TMEM, every UMMA accumulates, so an exact-zero column sum keeps
# Triton's +0). Warps: 0 MMA, 1 TMA, 2-3 idle, 4-11 col (TMEM), 12-19 row (SMEM).
RHT128_ROWCAST_QUANT_COL_WARP_BEGIN = 4
RHT128_ROWCAST_QUANT_COL_WARP_END = 12
RHT128_ROWCAST_QUANT_ROW_WARP_BEGIN = 12
RHT128_ROWCAST_QUANT_ROW_WARP_END = 20
RHT128_ROWCAST_QUANT_N_WARPS = 20
RHT128_ROWCAST_QUANT_TPB = 32 * RHT128_ROWCAST_QUANT_N_WARPS
RHT128_ROWCAST_QUANT_COL_THREADS = 32 * (
    RHT128_ROWCAST_QUANT_COL_WARP_END - RHT128_ROWCAST_QUANT_COL_WARP_BEGIN
)
RHT128_ROWCAST_QUANT_ROW_THREADS = 32 * (
    RHT128_ROWCAST_QUANT_ROW_WARP_END - RHT128_ROWCAST_QUANT_ROW_WARP_BEGIN
)
RHT128_ROWCAST_QUANT_ACC_CONSUMER_WARPS = (
    RHT128_ROWCAST_QUANT_COL_WARP_END - RHT128_ROWCAST_QUANT_COL_WARP_BEGIN
)
RHT128_ROWCAST_QUANT_BLOCKS_PER_WARP = (RHT128_DIM // 16) // 2
RHT128_ROWCAST_QUANT_ACC_ZERO_BAR = 3


@cute.jit
def _rht128_tile_quantize(
    acc, tidx, u_base, enc_over_fp4max, dec, row_addr, rSF, fast_math: cutlass.Constexpr
):
    """``_rht128_tile_requantize`` for the activation path: RTNE-quantize this warp's half
    of one 128x128 f32 TMEM accumulator -- this thread's lane = one output (hidden) row,
    blocks ``u_base .. u_base + 3`` of its eight along the transformed token index --
    under the op's ``use_fast_math`` arithmetic.

    The codes go straight to global as two 16-byte stores into the lane's code words
    ``8 * (u_base // 4) ..`` at ``row_addr``; the four scale bytes land in ``rSF`` for the
    swizzled store. After its loads the thread zero-fills its chunks of the stage
    (``_rht128_zero_acc``) so the next tile's chain accumulates onto ``+0``.
    """
    copy_atom_t2r = sm100_utils.get_tmem_load_op(
        RHT128_CTA_TILE_COL,
        utils.LayoutEnum.ROW_MAJOR,
        cutlass.Float32,
        cutlass.Float32,
        RHT128_EPI_TILE,
        False,
    )
    tAcc = transform_partitioned_tensor_layout(acc)
    tAcc_epi = cute.flat_divide(tAcc, RHT128_EPI_TILE)
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
    thr_copy = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc = thr_copy.partition_S(tAcc_epi)
    tTR_rAcc = cute.make_rmem_tensor(((16, 1), 1, 1), cutlass.Float32)
    rCodes = cute.make_rmem_tensor(
        (2 * RHT128_ROWCAST_QUANT_BLOCKS_PER_WARP,), cutlass.Uint32
    )
    for j in cutlass.range_constexpr(RHT128_ROWCAST_QUANT_BLOCKS_PER_WARP):
        u = u_base + cutlass.Int32(j)
        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, u)], tTR_rAcc)
        vals = tTR_rAcc.load().reshape((16,))
        w0, w1, pvscale_fp8 = _quant16(
            vals, enc_over_fp4max, dec, rht_acc=True, fast_math=fast_math
        )
        rCodes[2 * j] = w0
        rCodes[2 * j + 1] = w1
        rSF[j] = pvscale_fp8
    cute.arch.fence_view_async_tmem_load()
    _rht128_zero_acc(acc, tidx, u_base)
    cute.arch.fence_view_async_tmem_store()
    # 16-byte aligned by construction (64-byte code rows; ``u_base * 8`` is 0 or 32).
    code_addr = row_addr + cutlass.Int64(u_base) * cutlass.Int64(8)
    _st_global_v4_u32(code_addr, rCodes[0], rCodes[1], rCodes[2], rCodes[3])
    _st_global_v4_u32(
        code_addr + cutlass.Int64(16), rCodes[4], rCodes[5], rCodes[6], rCodes[7]
    )


def _rht128_rowcast_group_words(g, g_end, offsets_t, hidden):
    """``_store_grouped_col_sf_word``'s per-group constants: the u32 words of the groups
    before ``g``, the words per 128-row hidden block of ``g``'s swizzled scale tile, and
    ``g``'s first 16-token column. Plain function, traced inline at a group crossing."""
    prev = cutlass.select_(g > cutlass.Int32(0), g - cutlass.Int32(1), 0)
    group_start = cutlass.select_(
        g > cutlass.Int32(0), offsets_t[prev], cutlass.Int32(0)
    )
    prefix_words = cutlass.Int32(
        cutlass.Int64(hidden) * cutlass.Int64(group_start) // cutlass.Int64(64)
    )
    return (
        prefix_words,
        (g_end - group_start) * cutlass.Int32(2),
        group_start // cutlass.Int32(16),
    )


@cute.jit
def _rht128_rowcast_quantize_epilogue(
    mColFP4,
    mColSF,
    col_amax_t,
    tCtAcc,
    acc_pipeline,
    offsets_t,
    num_tensors,
    t_begin,
    t_end,
    tiles_in_m,
    hidden,
    tidx,
    u_base,
    fast_math: cutlass.Constexpr,
):
    """Consume the acc ring over this CTA's chunk: quantize every ``A_g.t() @ R`` tile into
    the ``(hidden, tokens // 8)`` u32 code rows and the per-group swizzled scales; this
    warp takes blocks ``u_base .. u_base + 3`` of its lanes.

    Tiles are enumerated hidden-fastest, so ``token`` is non-decreasing and the group
    cache is ``_rht128_amax_epilogue``'s; a crossing reloads the group's two-level scale
    and the constants of its swizzled scale tile (``_store_grouped_col_sf_word``'s
    arithmetic, with the per-group and per-lane terms hoisted out of the tile loop).
    Tile coordinates advance incrementally.
    """
    acc_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, RHT128_ACC_STAGES
    )
    tile_n = t_begin // tiles_in_m
    tile_m = t_begin - tile_n * tiles_in_m
    g = _group_idx(tile_n * cutlass.Int32(TOKEN_TILE), offsets_t, num_tensors)
    g_end = offsets_t[g]
    _, dec, enc_over_fp4max = _global_scale(col_amax_t[g])
    prefix_words, words_per_hidden_block, c_local_base = _rht128_rowcast_group_words(
        g, g_end, offsets_t, hidden
    )
    lane_word = (tidx % cutlass.Int32(32)) * cutlass.Int32(4) + tidx // cutlass.Int32(
        32
    )
    rSF = cute.make_rmem_tensor(
        (RHT128_ROWCAST_QUANT_BLOCKS_PER_WARP,), cutlass.Float8E4M3FN
    )
    for i in cutlass.range(t_end - t_begin, unroll=1):
        token = tile_n * cutlass.Int32(TOKEN_TILE)
        if token >= g_end:
            g = _group_idx(token, offsets_t, num_tensors)
            g_end = offsets_t[g]
            _, dec, enc_over_fp4max = _global_scale(col_amax_t[g])
            prefix_words, words_per_hidden_block, c_local_base = (
                _rht128_rowcast_group_words(g, g_end, offsets_t, hidden)
            )
        gCol = cute.local_tile(mColFP4, (M_TILE, TOKEN_TILE // 8), (tile_m, tile_n))
        acc_pipeline.consumer_wait(acc_state)
        _rht128_tile_quantize(
            tCtAcc[(None, None, None, acc_state.index)],
            tidx,
            u_base,
            enc_over_fp4max,
            dec,
            gCol[(tidx, None)].iterator.toint(),
            rSF,
            fast_math,
        )
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_state)
        acc_state.advance()
        c_local = tile_n * cutlass.Int32(RHT128_DIM // 16) + u_base - c_local_base
        mColSF[
            prefix_words
            + tile_m * words_per_hidden_block
            + (c_local // cutlass.Int32(4)) * cutlass.Int32(128)
            + lane_word
        ] = cute.recast_tensor(rSF, cutlass.Uint32)[0]
        tile_m = tile_m + cutlass.Int32(1)
        wrap = tile_m == tiles_in_m
        tile_m = cutlass.Int32(cutlass.select_(wrap, cutlass.Int32(0), tile_m))
        tile_n = tile_n + cutlass.Int32(
            cutlass.select_(wrap, cutlass.Int32(1), cutlass.Int32(0))
        )


@cute.jit
def _rht128_rowcast_sign_b(b_base, b_mbar, sign_t, u):
    """Sign the resident unsigned ``H128`` stage into ``R^T`` in place once the TMA behind
    ``b_mbar`` has landed: col thread ``u`` of 256 flips bit 15 of every bfloat16 in K chunk
    ``u % 16`` (tokens ``8 * (u % 16) ..``) of rows ``8 * (u // 16) ..`` whose token sign is
    negative -- ``torch.mul(h128, signs[None, :])`` byte for byte, since negating a bfloat16
    is exact for ``+-1`` signs. Every chunk is rewritten through the generic proxy and fenced
    for the UMMA; the caller's ``acc_zero_barrier`` hands the stage to the MMA warp.

    A 16-byte chunk of the K-major SW128 stage sits at ``row + 16 * (chunk ^ (row >> 7) % 8)``
    with ``row = b_base + 16384 * (k // 64) + 1024 * (n // 8) + 128 * (n % 8)``: the TMA
    swizzle is keyed on the absolute shared-memory address, so a stage that is not 1 KB
    aligned rotates the chunk order (probed, not assumed).
    """
    kc = u % cutlass.Int32(16)
    k0 = kc * cutlass.Int32(8)
    mask = cute.make_rmem_tensor((4,), cutlass.Uint32)
    for j in cutlass.range_constexpr(4):
        lo = sign_t[k0 + cutlass.Int32(2 * j)] < cutlass.Int8(0)
        hi = sign_t[k0 + cutlass.Int32(2 * j + 1)] < cutlass.Int8(0)
        mask[j] = cutlass.Uint32(
            cutlass.select_(lo, cutlass.Uint32(0x8000), cutlass.Uint32(0))
        ) | cutlass.Uint32(
            cutlass.select_(hi, cutlass.Uint32(0x80000000), cutlass.Uint32(0))
        )
    row0 = (
        b_base
        + cutlass.Int32(16384) * (kc // cutlass.Int32(8))
        + cutlass.Int32(1024) * (u // cutlass.Int32(16))
    )
    c = kc % cutlass.Int32(8)
    chunks = []
    words = []
    for i in cutlass.range_constexpr(8):
        row = row0 + cutlass.Int32(128 * i)
        chunks.append(
            cute.make_tensor(
                cute.make_ptr(
                    cutlass.Uint32,
                    row
                    + cutlass.Int32(16)
                    * (c ^ ((row >> cutlass.Int32(7)) & cutlass.Int32(7))),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                ),
                cute.make_layout((4,)),
            )
        )
        words.append(cute.make_rmem_tensor((4,), cutlass.Uint32))
    cute.arch.mbarrier_wait(b_mbar, 0)
    # All eight loads in flight before the first flip: the stage is on the MMA warp's
    # critical path once per CTA.
    for i in cutlass.range_constexpr(8):
        words[i].store(chunks[i].load())
    for i in cutlass.range_constexpr(8):
        for j in cutlass.range_constexpr(4):
            words[i][j] = words[i][j] ^ mask[j]
        chunks[i].store(words[i].load())
    # Generic-proxy stores -> visible to the UMMA (async proxy) before the barrier.
    cute.arch.fence_proxy("async.shared", space="cta")


class _Tcgen05GroupRowCastColRhtQuantize:
    """Per-group NVFP4 quantize of ``A.t() @ R`` (columnwise) and ``A`` (rowwise) in one
    pass over ``A``.

    Standalone (no ``_GroupRhtMainloop``): the mainloop is ``_Tcgen05GroupRowCastColRhtAmax``'s
    -- every 128x128 tile is TMA'd once, feeds one UMMA chain against the resident
    K-major ``R^T`` tile and is read raw by the row warps through the plain
    ``(hidden, token, stage)`` view. ``R^T`` is the unsigned ``H128`` TMA'd once and
    signed in shared memory by the col warps (``_rht128_rowcast_sign_b``) inside their
    zero-fill handshake with the MMA warp, so the op issues no torch glue. The
    accumulator is quantized from TMEM under the requantize kernel's discipline: the
    epilogue zero-fills its chunks and every UMMA accumulates, so an exact-zero column
    sum keeps Triton's ``+0``. Static persistent grid: CTA ``b`` owns the contiguous tile
    chunk ``_static_tile_range``, tiles below ``logical_packed_length`` only,
    hidden-fastest. Twenty warps
    (``RHT128_ROWCAST_QUANT_*``): eight col warps, two per TMEM quadrant, eight row warps.
    """

    def __init__(self, fast_math: bool = False):
        self.fast_math = fast_math

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # A.t().unsqueeze(-1): (hidden, tokens, 1)
        mB: cute.Tensor,  # H128 (128 j, 128 k, 1); signed in SMEM into the col chain's R^T
        mSign: cute.Tensor,  # (128,) int8 token signs (wgrad)
        mColFP4: cute.Tensor,  # (hidden, tokens // 8) u32 columnwise codes
        mColSF: cute.Tensor,  # flat u32 concatenation of per-group swizzled scales
        mRowFP4: cute.Tensor,  # (tokens, hidden // 16) u64: the row code pair per store
        mRowSF: cute.Tensor,  # (tokens // 128, hidden // 64, 32, 16) e4m3
        row_amax_t: cute.Tensor,  # (num_tensors,) f32
        col_amax_t: cute.Tensor,  # (num_tensors,) f32
        offsets_t: cute.Tensor,
        logical_len_t: cute.Tensor,
        hidden: cutlass.Int32,
        num_tensors: cutlass.Int32,
        num_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        k_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        mn_atom = tcgen05.make_smem_layout_atom(
            tcgen05.SmemLayoutAtomKind.MN_SW128, cutlass.BFloat16
        )
        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        mma_col = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            RHT128_MMA_TILER_COL,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.MN,
            OperandMajorMode.K,
        )
        tiled_mma_col = cute.make_tiled_mma(cute.make_mma_atom(mma_col))
        a_shape = tiled_mma_col.partition_shape_A(
            cute.dice(RHT128_CTA_TILE_COL, (1, None, 1))
        )
        a_smem_layout_staged = tcgen05.tile_to_mma_shape(
            mn_atom,
            cute.append(a_shape, RHT128_ROWCAST_MAINLOOP_STAGES),
            order=(1, 2, 3),
        )
        # Same bytes, plain (hidden, token, stage) grouping for the row warps.
        a_clean_layout = cute.tile_to_shape(
            mn_atom,
            (M_TILE, TOKEN_TILE, RHT128_ROWCAST_MAINLOOP_STAGES),
            order=(0, 1, 2),
        )
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            g2s,
            mA,
            cute.slice_(a_smem_layout_staged, (None, None, None, 0)),
            RHT128_CTA_TILE_COL,
            tiled_mma_col,
            (1, 1, 1, 1),
        )
        b_shape = tiled_mma_col.partition_shape_B(
            cute.dice(RHT128_CTA_TILE_COL, (None, 1, 1))
        )
        b_smem_layout = tcgen05.tile_to_mma_shape(
            k_atom, cute.append(b_shape, 1), order=(1, 2, 3)
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            g2s,
            mB,
            cute.slice_(b_smem_layout, (None, None, None, 0)),
            RHT128_CTA_TILE_COL,
            tiled_mma_col,
            (1, 1, 1, 1),
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma_col.thr_id.shape,)
        )
        tCtAcc_fake = tiled_mma_col.make_fragment_C(
            cute.append(
                tiled_mma_col.partition_shape_C((M_TILE, RHT128_DIM)),
                RHT128_ACC_STAGES,
            )
        )
        num_tmem_alloc_cols = sm100_utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        self.kernel(
            tiled_mma_col,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            mSign,
            mColFP4,
            mColSF,
            mRowFP4,
            mRowSF,
            row_amax_t,
            col_amax_t,
            offsets_t,
            logical_len_t,
            cluster_layout_vmnk,
            a_smem_layout_staged,
            a_clean_layout,
            b_smem_layout,
            tCtAcc_fake.layout,
            num_tmem_alloc_cols,
            hidden,
            num_tensors,
        ).launch(
            grid=(num_ctas, 1, 1), block=(RHT128_ROWCAST_QUANT_TPB, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma_col: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        mSign: cute.Tensor,
        mColFP4: cute.Tensor,
        mColSF: cute.Tensor,
        mRowFP4: cute.Tensor,
        mRowSF: cute.Tensor,
        row_amax_t: cute.Tensor,
        col_amax_t: cute.Tensor,
        offsets_t: cute.Tensor,
        logical_len_t: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        a_clean_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        acc_fake_layout: cute.Layout,
        num_tmem_alloc_cols: cutlass.Constexpr,
        hidden: cutlass.Int32,
        num_tensors: cutlass.Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        tiles_in_m = hidden // cutlass.Int32(M_TILE)
        tiles_in_n_valid = logical_len_t[0] // cutlass.Int32(TOKEN_TILE)
        t_begin, t_end = _static_tile_range(tiles_in_m * tiles_in_n_valid, bidx, gdim)
        n_my = t_end - t_begin

        if warp_idx == TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        @cute.struct
        class SharedStorage:
            ab_mbar: cute.struct.MemRange[
                cutlass.Int64, RHT128_ROWCAST_MAINLOOP_STAGES * 2
            ]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, RHT128_ACC_STAGES * 2]
            b_mbar: cutlass.Int64
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mainloop: one TMA producer, two heterogeneous consumer groups on the
        # same stage -- the UMMA (released via tcgen05.commit) and the 256 row
        # threads (released via a plain mbarrier arrive).
        ab_pipeline = pipeline.PipelineTmaMultiConsumersAsync.create(
            barrier_storage=storage.ab_mbar.data_ptr(),
            num_stages=RHT128_ROWCAST_MAINLOOP_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group_umma=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group_async=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, RHT128_ROWCAST_QUANT_ROW_THREADS
            ),
            tx_count=_A_TILE_BYTES,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=RHT128_ACC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, RHT128_ROWCAST_QUANT_ACC_CONSUMER_WARPS
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        # MMA warp + the col epilogue group retrieve the TMEM pointer.
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_ALLOC_BAR,
            num_threads=32 + RHT128_ROWCAST_QUANT_COL_THREADS,
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=TMEM_DEALLOC_BAR, num_threads=RHT128_ROWCAST_QUANT_COL_THREADS
        )
        acc_zero_barrier = pipeline.NamedBarrier(
            barrier_id=RHT128_ROWCAST_QUANT_ACC_ZERO_BAR,
            num_threads=32 + RHT128_ROWCAST_QUANT_COL_THREADS,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=RHT128_ROWCAST_QUANT_COL_WARP_BEGIN,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        if warp_idx == TMA_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(storage.b_mbar.ptr, 1)
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # The UMMA reads the A stage through its swizzled MN-major layout; the
        # row warps see a plain (hidden, token, stage) view of the same bytes.
        raw_a = smem.allocate_array(
            cutlass.BFloat16,
            cute.cosize(a_smem_layout_staged.outer),
            byte_alignment=128,
        )
        swz_ptr = cute.recast_ptr(
            raw_a, a_smem_layout_staged.inner, dtype=cutlass.BFloat16
        )
        sA = cute.make_tensor(swz_ptr, a_smem_layout_staged.outer)
        sA_clean = cute.make_tensor(swz_ptr, a_clean_layout.outer)
        raw_b = smem.allocate_array(
            cutlass.BFloat16, cute.cosize(b_smem_layout.outer), byte_alignment=128
        )
        sB = cute.make_tensor(
            cute.recast_ptr(raw_b, b_smem_layout.inner, dtype=cutlass.BFloat16),
            b_smem_layout.outer,
        )

        cta_layout = cute.make_layout((1,))
        thr_col = tiled_mma_col.get_slice(0)
        gA = cute.local_tile(
            mA, cute.slice_(RHT128_CTA_TILE_COL, (None, 0, None)), (None, None, None)
        )
        gB = cute.local_tile(
            mB, cute.slice_(RHT128_CTA_TILE_COL, (0, None, None)), (None, None, None)
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_col.partition_A(gA), 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(thr_col.partition_B(gB), 0, 3),
        )
        tCrA_col = tiled_mma_col.make_fragment_A(sA)
        tCrB_col = tiled_mma_col.make_fragment_B(sB)

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ==================== TMA warp ====================
        if warp_idx == TMA_WARP:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    storage.b_mbar.ptr, RHT128_B_BYTES
                )
            cute.copy(
                tma_atom_b,
                tBgB[(None, 0, 0, 0)],
                tBsB[(None, 0)],
                tma_bar_ptr=storage.b_mbar.ptr,
            )
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, RHT128_ROWCAST_MAINLOOP_STAGES
            )
            tile_n = t_begin // tiles_in_m
            tile_m = t_begin - tile_n * tiles_in_m
            for i in cutlass.range(n_my, unroll=1):
                ab_pipeline.producer_acquire(ab_producer_state)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, tile_m, tile_n, 0)],
                    tAsA[(None, ab_producer_state.index)],
                    tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                )
                ab_producer_state.advance()
                tile_m = tile_m + cutlass.Int32(1)
                wrap = tile_m == tiles_in_m
                tile_m = cutlass.Int32(cutlass.select_(wrap, cutlass.Int32(0), tile_m))
                tile_n = tile_n + cutlass.Int32(
                    cutlass.select_(wrap, cutlass.Int32(1), cutlass.Int32(0))
                )
            ab_pipeline.producer_tail(ab_producer_state)

        # ==================== MMA warp ====================
        if warp_idx == MMA_WARP:
            tmem.wait_for_alloc()
            tCtAcc = cute.make_tensor(
                tmem.retrieve_ptr(cutlass.Float32), acc_fake_layout
            )
            acc_zero_barrier.arrive_and_wait()
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, RHT128_ROWCAST_MAINLOOP_STAGES
            )
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, RHT128_ACC_STAGES
            )
            for i in cutlass.range(n_my, unroll=1):
                ab_pipeline.consumer_wait(ab_consumer_state)
                acc_pipeline.producer_acquire(acc_state)
                acc_col = tCtAcc[(None, None, None, acc_state.index)]
                for kb in cutlass.range_constexpr(RHT128_K_BLOCKS):
                    # Every step accumulates onto the epilogue's zero fill (Triton's
                    # ``tl.dot`` discipline); never ``kb > 0`` without that fill.
                    tiled_mma_col.set(tcgen05.Field.ACCUMULATE, True)
                    cute.gemm(
                        tiled_mma_col,
                        acc_col,
                        tCrA_col[(None, None, kb, ab_consumer_state.index)],
                        tCrB_col[(None, None, kb, 0)],
                        acc_col,
                    )
                acc_pipeline.producer_commit(acc_state)
                acc_state.advance()
                ab_pipeline.consumer_release(
                    ab_consumer_state, pipeline.PipelineOp.TCGen05Mma
                )
                ab_consumer_state.advance()
            acc_pipeline.producer_tail(acc_state)

        # ==================== col epilogue: quantize A.t() @ R per group ====================
        if (
            warp_idx >= RHT128_ROWCAST_QUANT_COL_WARP_BEGIN
            and warp_idx < RHT128_ROWCAST_QUANT_COL_WARP_END
        ):
            tmem.allocate(num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc = cute.make_tensor(tmem_ptr, acc_fake_layout)
            # Warps 4-7 take blocks 0-3 of their quadrant's lanes, warps 8-11 blocks 4-7.
            u_base = (
                (warp_idx - cutlass.Int32(RHT128_ROWCAST_QUANT_COL_WARP_BEGIN))
                // cutlass.Int32(4)
            ) * cutlass.Int32(RHT128_ROWCAST_QUANT_BLOCKS_PER_WARP)
            for s in cutlass.range_constexpr(RHT128_ACC_STAGES):
                _rht128_zero_acc(
                    tCtAcc[(None, None, None, s)],
                    tidx % cutlass.Int32(RHT128_DIM),
                    u_base,
                )
            cute.arch.fence_view_async_tmem_store()
            _rht128_rowcast_sign_b(
                raw_b.toint(),
                storage.b_mbar.ptr,
                mSign,
                tidx - cutlass.Int32(32 * RHT128_ROWCAST_QUANT_COL_WARP_BEGIN),
            )
            acc_zero_barrier.arrive_and_wait()
            _rht128_rowcast_quantize_epilogue(
                mColFP4,
                mColSF,
                col_amax_t,
                tCtAcc,
                acc_pipeline,
                offsets_t,
                num_tensors,
                t_begin,
                t_end,
                tiles_in_m,
                hidden,
                tidx % cutlass.Int32(RHT128_DIM),
                u_base,
                self.fast_math,
            )
            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ==================== Rowwise epilogue (SMEM -> NVFP4) ====================
        if (
            warp_idx >= RHT128_ROWCAST_QUANT_ROW_WARP_BEGIN
            and warp_idx < RHT128_ROWCAST_QUANT_ROW_WARP_END
        ):
            r_local = tidx - RHT128_ROWCAST_QUANT_ROW_WARP_BEGIN * cutlass.Int32(32)
            hb = r_local % cutlass.Int32(ROW_HB)  # 16-hidden block
            hb4 = hb // cutlass.Int32(4)
            # Token within a pass. Lanes hb 4-7 take the neighbouring token: their hidden
            # atom sits 1 KB above lanes 0-3's on the same swizzle row, so reading the same
            # token would put a quarter-warp's two 64-byte reads on the same banks; the
            # neighbouring token's row flips the chunk swizzle and the reads are conflict
            # free. A permutation of (token, hb) over the lanes -- every block is
            # quantized once and stored to its own address.
            t0 = (r_local // cutlass.Int32(ROW_HB)) ^ hb4
            # The lane's swizzled scale bytes: ``_store_sf_byte``'s ``(r // 128, c // 4,
            # r % 32, (r % 128 // 32) * 4 + c % 4)`` at ``r = token + 32 p + t0``,
            # ``c = 8 tile_m + hb`` is ``(tile_n, 2 tile_m + hb // 4, t0, 4 p + hb % 4)``.
            sf_lane = hb % cutlass.Int32(4)

            blk = cute.make_rmem_tensor((16,), cutlass.Float32)
            rBlk = cute.make_rmem_tensor((16,), cutlass.BFloat16)
            rPair = cute.make_rmem_tensor((2,), cutlass.Uint32)
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, RHT128_ROWCAST_MAINLOOP_STAGES
            )
            tile_n = t_begin // tiles_in_m
            tile_m = t_begin - tile_n * tiles_in_m
            g = _group_idx(tile_n * cutlass.Int32(TOKEN_TILE), offsets_t, num_tensors)
            g_end = offsets_t[g]
            _, r_dec, r_enc_over_fp4max = _global_scale(row_amax_t[g])
            for i in cutlass.range(n_my, unroll=1):
                token = tile_n * cutlass.Int32(TOKEN_TILE)
                if token >= g_end:
                    g = _group_idx(token, offsets_t, num_tensors)
                    g_end = offsets_t[g]
                    _, r_dec, r_enc_over_fp4max = _global_scale(row_amax_t[g])

                ab_pipeline.consumer_wait(ab_consumer_state)
                stage = ab_consumer_state.index
                gRow = cute.local_tile(
                    mRowFP4, (TOKEN_TILE, M_TILE // 16), (tile_n, tile_m)
                )[(None, hb)]
                gSF = mRowSF[(tile_n, tile_m * cutlass.Int32(2) + hb4, t0, None)]
                for p in cutlass.range_constexpr(ROW_PASSES):
                    tok = p * cutlass.Int32(ROW_TOK_PER_PASS) + t0
                    cute.autovec_copy(
                        cute.local_tile(sA_clean[(None, tok, stage)], (16,), (hb,)),
                        rBlk,
                    )
                    rWords = cute.recast_tensor(rBlk, cutlass.Uint32)
                    for j in cutlass.range_constexpr(8):
                        blk[2 * j] = _bf16lo_to_f32(rWords[j])
                        blk[2 * j + 1] = _bf16hi_to_f32(rWords[j])
                    w0, w1, sf = _quant16(
                        blk,
                        r_enc_over_fp4max,
                        r_dec,
                        False,
                        None,
                        fast_math=self.fast_math,
                    )
                    # One 64-bit store, not two 32-bit ones (see the RHT-16 fused kernel).
                    rPair[0] = w0
                    rPair[1] = w1
                    pair64 = cute.recast_tensor(rPair, cutlass.Uint64)
                    gRow[tok] = pair64[0]
                    gSF[cutlass.Int32(4 * p) + sf_lane] = sf
                ab_pipeline.consumer_release(
                    ab_consumer_state, pipeline.PipelineOp.AsyncThread
                )
                ab_consumer_state.advance()
                tile_m = tile_m + cutlass.Int32(1)
                wrap = tile_m == tiles_in_m
                tile_m = cutlass.Int32(cutlass.select_(wrap, cutlass.Int32(0), tile_m))
                tile_n = tile_n + cutlass.Int32(
                    cutlass.select_(wrap, cutlass.Int32(1), cutlass.Int32(0))
                )


@functools.lru_cache(maxsize=None)
def _compile_group_row_cast_col_rht_quantize_kernel(device_idx: int, fast_math: bool):
    """Compile the grouped row-cast + col-RHT quantize kernel with symbolic shapes (cached
    per device+flag); see ``_compile_group_fused_kernel`` for the alignment facts the
    ``assumed_align`` / divisibilities record."""
    free = cute.sym_int
    h_sym = cute.sym_int(divisibility=M_TILE)
    t_sym = cute.sym_int(divisibility=TOKEN_TILE)
    k = _Tcgen05GroupRowCastColRhtQuantize(fast_math=fast_math)
    return cute.compile(
        k,
        make_fake_tensor(cutlass.BFloat16, (h_sym, t_sym, 1), stride=(1, free(), 1)),
        make_fake_tensor(
            cutlass.BFloat16, (RHT128_DIM, RHT128_DIM, 1), stride=(RHT128_DIM, 1, 1)
        ),
        make_fake_tensor(cutlass.Int8, (RHT128_DIM,), stride=(1,)),
        make_fake_tensor(
            cutlass.Uint32,
            (h_sym, cute.sym_int(divisibility=TOKEN_TILE // 8)),
            stride=(cute.sym_int(divisibility=TOKEN_TILE // 8), 1),
            assumed_align=16,
        ),
        make_fake_tensor(cutlass.Uint32, (free(),), stride=(1,)),
        make_fake_tensor(
            cutlass.Uint64,
            (t_sym, cute.sym_int(divisibility=M_TILE // 16)),
            stride=(free(), 1),
            assumed_align=16,
        ),
        make_fake_tensor(
            cutlass.Float8E4M3FN, (free(), free(), 32, 16), stride=(free(), 512, 16, 1)
        ),
        make_fake_tensor(cutlass.Float32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Float32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Int32, (free(),), stride=(1,)),
        make_fake_tensor(cutlass.Int32, (1,), stride=(1,)),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def _cutedsl_group_row_cast_col_rht_quantize_impl(
    A: torch.Tensor,
    offsets: torch.Tensor,
    row_global_amax: torch.Tensor,
    col_global_amax: torch.Tensor,
    num_tensors: int,
    wgrad_rht: torch.Tensor,
    h128: torch.Tensor,
    logical_packed_length: Optional[torch.Tensor] = None,
    use_fast_math: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grouped RHT-128 columnwise + raw rowwise NVFP4 quantization (RTNE).

    ``A`` is ``(tokens, hidden)`` bfloat16 row-major; ``wgrad_rht`` is the live ``(128,)``
    int8 sign vector and ``h128`` the unsigned ``(128, 128)`` bfloat16 Hadamard, signed
    into the K-major ``R^T`` operand on chip. Returns ``(col_fp4, col_sf, row_fp4, row_sf)``
    as ``_cutedsl_group_rht_quantize_row_col_impl`` does; rows at or after
    ``logical_packed_length`` are never read and their outputs are left as allocated.
    """
    tokens, hidden = A.shape
    dev = A.device
    A = A.detach()

    col_fp4 = torch.empty((hidden, tokens // 8), dtype=torch.uint32, device=dev)
    row_fp4 = torch.empty((tokens, hidden // 8), dtype=torch.uint32, device=dev)
    col_sf = torch.empty(
        (hidden // 128, tokens // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev
    )
    row_sf = torch.empty(
        (tokens // 128, hidden // 64, 32, 16), dtype=torch.float8_e4m3fn, device=dev
    )

    if logical_packed_length is None:
        logical_packed_length = offsets[-1:]
    # See the fused kernel: the entry point requires byte_offset==0.
    logical_packed_length = logical_packed_length.clone()
    tiles = (hidden // M_TILE) * (tokens // TOKEN_TILE)
    num_ctas = max(1, min(tiles, _get_num_sms(dev.index)))

    stream = cuda.CUstream(int(torch.cuda.current_stream(dev).cuda_stream))
    _compile_group_row_cast_col_rht_quantize_kernel(dev.index, bool(use_fast_math))(
        A.t().unsqueeze(-1),
        h128.unsqueeze(-1),
        wgrad_rht,
        col_fp4,
        col_sf.view(torch.uint32).flatten(),
        row_fp4.view(torch.uint64),
        row_sf,
        row_global_amax,
        col_global_amax,
        offsets,
        logical_packed_length,
        int(hidden),
        int(num_tensors),
        int(num_ctas),
        stream,
    )
    return col_fp4.view(torch.uint8), col_sf, row_fp4.view(torch.uint8), row_sf
