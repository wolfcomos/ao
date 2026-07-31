# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Unified SwiGLU + MXFP8 CuTe DSL kernel for SM100.

One pass computes the SwiGLU activation and its MXFP8 cast, so the activation
never round-trips through global memory as bfloat16:

    forward:   h     = silu(gate) * up
    backward:  dGate = grad_h * up * d_silu(gate),  dUp = grad_h * silu(gate)

Input is a packed ``gated_input`` of shape (M, 2K) holding ``gate`` in the first
K columns and ``up`` in the last K. Forward outputs are K wide; backward outputs
are 2K wide and hold the concatenated ``[dGate | dUp]``. Rowwise (1x32) scales,
colwise (32x1) scales, or both are produced from that single read, with
``IS_BWD`` / ``ROWWISE`` / ``COLWISE`` as ``Constexpr`` flags so each combination
specializes with no runtime branching.

Requires M and K to be multiples of 128, and ``2*K*M <= INT32_MAX`` because the
gmem layouts are built from Python ints, so CuTe emits 32-bit index math.
Scaling is RCEIL; scales use the blocked tcgen05 layouts.
"""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute import AddressSpace
from cutlass.cutlass_dsl import T, dsl_user_op

try:
    from cuda.bindings.driver import CUstream
except Exception:
    from cuda.cuda import CUstream

F32 = cutlass.Float32
BF16 = cutlass.BFloat16
I32 = cutlass.Int32
U8 = cutlass.Uint8

EVICT_FIRST = cute.nvgpu.common.CacheEvictionPriority.EVICT_FIRST

# CTA tile geometry. Each CTA owns a TR x TJ tile of the logical [M, K] output.
TR = 32  # rows per CTA tile
TJ = 128  # j-columns per CTA tile

# Transposed smem cache used only by the colwise path. Rows are padded from 32
# to SP words so that the colwise read pattern is bank-conflict free.
SP = 36  # padded row-stride, in 32-bit words
SH = 64 * SP  # words per output half ([dGate | dUp] is two halves)

_INT32_MAX = 2**31 - 1


@dsl_user_op
def _qpack(w0, w1, s, *, loc=None, ip=None):
    """Scale two bf16x2 words by bf16x2 `s`, convert to 4 packed e4m3 bytes."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(w0).ir_value(loc=loc, ip=ip),
                cutlass.Int32(w1).ir_value(loc=loc, ip=ip),
                cutlass.Int32(s).ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .b16 a, b;\n"
            ".reg .b32 t0, t1;\n"
            "mul.rn.bf16x2 t0, $1, $3;\n"
            "mul.rn.bf16x2 t1, $2, $3;\n"
            "cvt.rn.satfinite.e4m3x2.bf16x2 a, t0;\n"
            "cvt.rn.satfinite.e4m3x2.bf16x2 b, t1;\n"
            "mov.b32 $0, {a, b}; }",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _binary_b32_op(name, asm, doc):
    """Wrap a two-operand, one-result b32 PTX instruction as a dsl_user_op."""

    @dsl_user_op
    def op(a, b, *, loc=None, ip=None):
        return cutlass.Int32(
            llvm.inline_asm(
                T.i32(),
                [
                    cutlass.Int32(a).ir_value(loc=loc, ip=ip),
                    cutlass.Int32(b).ir_value(loc=loc, ip=ip),
                ],
                asm,
                "=r,r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    op.__name__ = name
    op.__qualname__ = name
    op.__doc__ = doc
    return op


_prmt_even = _binary_b32_op(
    "_prmt_even",
    "prmt.b32 $0, $1, $2, 0x6420;",
    "Select bytes [0,2,4,6] from a pair of b32 words.",
)

_prmt_odd = _binary_b32_op(
    "_prmt_odd",
    "prmt.b32 $0, $1, $2, 0x7531;",
    "Select bytes [1,3,5,7] from a pair of b32 words.",
)

_max_bf16x2 = _binary_b32_op(
    "_max_bf16x2",
    "max.bf16x2 $0, $1, $2;",
    "Elementwise max of two packed bf16x2 words (operands must be non-negative).",
)

_amax_bf16x2 = _binary_b32_op(
    "_amax_bf16x2",
    "max.xorsign.abs.bf16x2 $0, $1, $2;",
    "max(|a|,|b|) per bf16 lane; result sign bits are junk and must be masked.",
)


@dsl_user_op
def _bf16x2(hi, lo, *, loc=None, ip=None):
    """(hi, lo) f32 -> packed bf16x2 word, RNE. lo occupies bits [15:0]."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [F32(hi).ir_value(loc=loc, ip=ip), F32(lo).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sigmoid(x, *, loc=None, ip=None):
    """``1 / (1 + exp(-x))``, lowered exactly as the Triton reference does.

    Emitted as raw PTX because bitwise agreement with that kernel depends on the
    precise instruction choice, and both of these are approximations that a
    higher-level formulation would not reproduce::

        mul.f32        t, x, 0fBFB8AA3B    // -x * log2(e)
        ex2.approx.f32 t, t                // fast exponential
        add.f32        t, t, 0f3F800000
        div.full.f32   s, 1.0, t           // fast division, ~2 ULP

    Using an accurate exponential or a correctly rounded reciprocal here is both
    slower and *less* compatible: it disagrees with the reference on a few codes
    per million after the bf16 and E4M3 rounding.
    """
    return F32(
        llvm.inline_asm(
            T.f32(),
            [F32(x).ir_value(loc=loc, ip=ip)],
            "{ .reg .f32 t;\n"
            "mul.f32 t, $1, 0fBFB8AA3B;\n"
            "ex2.approx.f32 t, t;\n"
            "add.f32 t, t, 0f3F800000;\n"
            "div.full.f32 $0, 0f3F800000, t; }",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _e8m0_bits(u: I32) -> I32:
    """Biased E8M0 byte for a non-negative bf16 magnitude given as f32 bits."""
    return cutlass.max(((u + I32(0x1F0000)) >> 23) - I32(8), I32(0))


@cute.jit
def _rcp_b16(e: I32) -> I32:
    """2^(127-e) as a bf16 bit pattern (1.0 for e == 0)."""
    b = (I32(254) - e) << 7
    return I32(0x3F80) if e == I32(0) else b


@cute.kernel
def swiglu_mxfp8_kernel(
    mG: cute.Tensor,
    mGate: cute.Tensor,
    mUp: cute.Tensor,
    mRQ: cute.Tensor,
    mCQ: cute.Tensor,
    mRS: cute.Tensor,
    mCS: cute.Tensor,
    M: cutlass.Constexpr,
    K: cutlass.Constexpr,
    NJT: cutlass.Constexpr,
    NT: cutlass.Constexpr,
    IS_BWD: cutlass.Constexpr,
    ROWWISE: cutlass.Constexpr,
    COLWISE: cutlass.Constexpr,
):
    """TE-style SwiGLU + MXFP8 body specialized entirely at compile time.

    IS_BWD selects forward hidden output versus concatenated [dGate, dUp].
    ROWWISE enables row-major 1x32 data/scales.
    COLWISE enables column-major 32x1 data/scales.
    """
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()

    OUT_HALVES = 2 if cutlass.const_expr(IS_BWD) else 1
    NJ128 = K // 128
    CBrow = OUT_HALVES * K // 128
    CBcol = M // 128
    ONE = F32(1.0)

    if cutlass.const_expr(COLWISE):
        smem = cutlass.utils.SmemAllocator()
        sptr = smem.allocate(OUT_HALVES * SH * 4, byte_alignment=16)
        sW = cute.make_tensor(
            cute.recast_ptr(sptr, dtype=I32),
            cute.make_layout(
                (8, 8, TR, OUT_HALVES),
                stride=(8 * SP, SP, 1, SH),
            ),
        )
        sD = cute.make_tensor(
            cute.recast_ptr(sptr, dtype=I32),
            cute.make_layout(
                (4, 8, 64, OUT_HALVES),
                stride=(1, 4, SP, SH),
            ),
        )

    rt = bid // I32(NJT)
    jt = bid - rt * I32(NJT)

    NR = NT // 8
    NP = TR // NR
    chunk = tid & I32(7)
    rw = tid >> 3
    blk = chunk >> 1
    odd = (chunk & I32(1)) == I32(1)

    fg = cute.make_rmem_tensor(16, BF16)
    fu = cute.make_rmem_tensor(16, BF16)
    if cutlass.const_expr(IS_BWD):
        fh = cute.make_rmem_tensor(16, BF16)
    w0 = cute.make_rmem_tensor(8, I32)
    if cutlass.const_expr(IS_BWD):
        w1 = cute.make_rmem_tensor(8, I32)
    if cutlass.const_expr(ROWWISE):
        fq4r = cute.make_rmem_tensor(4, I32)

    # A runtime loop, not range_constexpr: fully unrolling this phase regressed
    # large-shape performance in the backward-bidirectional tuning runs.
    for p in cutlass.range(NP):
        rl = p * NR + rw
        row = rt * I32(TR) + rl
        cute.autovec_copy(
            mGate[None, chunk, jt, row],
            fg,
            l1c_evict_priority=EVICT_FIRST,
        )
        cute.autovec_copy(
            mUp[None, chunk, jt, row],
            fu,
            l1c_evict_priority=EVICT_FIRST,
        )
        if cutlass.const_expr(IS_BWD):
            cute.autovec_copy(
                mG[None, chunk, jt, row],
                fh,
                l1c_evict_priority=EVICT_FIRST,
            )

        for e in cutlass.range_constexpr(0, 16, 2):
            g0 = fg[e].to(F32)
            g1 = fg[e + 1].to(F32)
            u0 = fu[e].to(F32)
            u1 = fu[e + 1].to(F32)

            s0 = _sigmoid(g0)
            s1 = _sigmoid(g1)
            silu0, silu1 = cute.arch.mul_packed_f32x2((g0, g1), (s0, s1))

            if cutlass.const_expr(IS_BWD):
                h0 = fh[e].to(F32)
                h1 = fh[e + 1].to(F32)
                one_minus_s0, one_minus_s1 = cute.arch.sub_packed_f32x2(
                    (ONE, ONE), (s0, s1)
                )
                # fma, not a separate multiply and add: the Triton reference
                # writes `1.0 + gate * (1.0 - sigmoid)`, which its compiler
                # contracts into a single FMA. Splitting it rounds twice and
                # costs bitwise agreement on a few codes per million.
                deriv0, deriv1 = cute.arch.fma_packed_f32x2(
                    (g0, g1), (one_minus_s0, one_minus_s1), (ONE, ONE)
                )
                silu_grad0, silu_grad1 = cute.arch.mul_packed_f32x2(
                    (s0, s1), (deriv0, deriv1)
                )
                grad_up0, grad_up1 = cute.arch.mul_packed_f32x2((h0, h1), (u0, u1))
                dg0, dg1 = cute.arch.mul_packed_f32x2(
                    (grad_up0, grad_up1), (silu_grad0, silu_grad1)
                )
                du0, du1 = cute.arch.mul_packed_f32x2((h0, h1), (silu0, silu1))
                w0[e >> 1] = _bf16x2(dg1, dg0)
                w1[e >> 1] = _bf16x2(du1, du0)
            else:
                y0, y1 = cute.arch.mul_packed_f32x2((silu0, silu1), (u0, u1))
                w0[e >> 1] = _bf16x2(y1, y0)

        if cutlass.const_expr(COLWISE):
            cute.autovec_copy(w0, sW[None, chunk, rl, 0])
            if cutlass.const_expr(IS_BWD):
                cute.autovec_copy(w1, sW[None, chunk, rl, 1])

        if cutlass.const_expr(ROWWISE):
            am0 = I32(0)
            if cutlass.const_expr(IS_BWD):
                am1 = I32(0)
            for e in cutlass.range_constexpr(8):
                am0 = _amax_bf16x2(am0, w0[e])
                if cutlass.const_expr(IS_BWD):
                    am1 = _amax_bf16x2(am1, w1[e])

            am0 = am0 & I32(0x7FFF7FFF)
            am0 = _max_bf16x2(am0, cute.arch.shuffle_sync_bfly(am0, 1))
            am0 = _max_bf16x2(am0, am0 >> 16)
            e0 = _e8m0_bits((am0 & I32(0xFFFF)) << 16)
            r0 = _rcp_b16(e0) * I32(0x10001)
            for k in cutlass.range_constexpr(4):
                fq4r[k] = _qpack(w0[2 * k], w0[2 * k + 1], r0)
            cute.autovec_copy(fq4r, mRQ[None, chunk, jt, row])

            if cutlass.const_expr(IS_BWD):
                am1 = am1 & I32(0x7FFF7FFF)
                am1 = _max_bf16x2(am1, cute.arch.shuffle_sync_bfly(am1, 1))
                am1 = _max_bf16x2(am1, am1 >> 16)
                e1 = _e8m0_bits((am1 & I32(0xFFFF)) << 16)
                r1 = _rcp_b16(e1) * I32(0x10001)
                for k in cutlass.range_constexpr(4):
                    fq4r[k] = _qpack(w1[2 * k], w1[2 * k + 1], r1)
                cute.autovec_copy(
                    fq4r,
                    mRQ[None, chunk, I32(NJ128) + jt, row],
                )

            roff = (
                ((rt >> 2) * I32(CBrow) + jt) * 512
                + rw * 16
                + (rt & I32(3)) * 4
                + blk
                + p * (NR * 16)
            )
            if cutlass.const_expr(IS_BWD):
                if odd:
                    mRS[roff + I32(NJ128 * 512)] = e1.to(U8)
                else:
                    mRS[roff] = e0.to(U8)
            else:
                if not odd:
                    mRS[roff] = e0.to(U8)

    if cutlass.const_expr(COLWISE):
        cute.arch.sync_threads()

        if tid < I32(64 * OUT_HALVES):
            arr = tid >> 6
            c = tid & I32(63)
            jl = (c & I32(7)) * 16 + (c >> 3) * 2

            vs = [cute.make_rmem_tensor(4, I32) for _ in range(8)]
            for i in cutlass.range_constexpr(8):
                cute.autovec_copy(sD[None, i, c, arr], vs[i])

            ac = I32(0)
            for i in cutlass.range_constexpr(8):
                for t in cutlass.range_constexpr(4):
                    ac = _amax_bf16x2(ac, vs[i][t])
            ac = ac & I32(0x7FFF7FFF)

            ec0 = _e8m0_bits((ac & I32(0xFFFF)) << 16)
            ec1 = _e8m0_bits(ac & I32(-65536))
            s01 = _rcp_b16(ec0) | (_rcp_b16(ec1) << 16)

            fq8 = cute.make_rmem_tensor(8, I32)
            fq8b = cute.make_rmem_tensor(8, I32)
            cout = jt * I32(TJ) + arr * I32(K) + jl
            for k in cutlass.range_constexpr(8):
                a01 = _qpack(vs[k][0], vs[k][1], s01)
                a23 = _qpack(vs[k][2], vs[k][3], s01)
                fq8[k] = _prmt_even(a01, a23)
                fq8b[k] = _prmt_odd(a01, a23)
            cute.autovec_copy(fq8, mCQ[None, cout, rt])
            cute.autovec_copy(fq8b, mCQ[None, cout + 1, rt])

            coff = (
                ((jt + arr * I32(NJ128)) * I32(CBcol) + (rt >> 2)) * 512
                + (jl & I32(31)) * 16
                + ((jl >> 5) & I32(3)) * 4
                + (rt & I32(3))
            )
            mCS[coff] = ec0.to(U8)
            mCS[coff + 16] = ec1.to(U8)


@cute.jit
def launcher(
    ag: cutlass.Int64,
    agi: cutlass.Int64,
    arq: cutlass.Int64,
    ars: cutlass.Int64,
    acq: cutlass.Int64,
    acs: cutlass.Int64,
    stream,
    M: cutlass.Constexpr,
    K: cutlass.Constexpr,
    IS_BWD: cutlass.Constexpr,
    ROWWISE: cutlass.Constexpr,
    COLWISE: cutlass.Constexpr,
):
    OUT_HALVES = 2 if cutlass.const_expr(IS_BWD) else 1
    NJT = K // TJ
    NRT = M // TR

    pg = cute.make_ptr(BF16, ag, AddressSpace.gmem, assumed_align=32)
    pgate = cute.make_ptr(BF16, agi, AddressSpace.gmem, assumed_align=32)
    pup = pgate + K
    prq = cute.make_ptr(I32, arq, AddressSpace.gmem, assumed_align=16)
    pcq = cute.make_ptr(I32, acq, AddressSpace.gmem, assumed_align=32)
    prs = cute.make_ptr(U8, ars, AddressSpace.gmem, assumed_align=16)
    pcs = cute.make_ptr(U8, acs, AddressSpace.gmem, assumed_align=16)

    mG = cute.make_tensor(
        pg,
        cute.make_layout((16, 8, NJT, M), stride=(1, 16, TJ, K)),
    )
    mGate = cute.make_tensor(
        pgate,
        cute.make_layout((16, 8, NJT, M), stride=(1, 16, TJ, 2 * K)),
    )
    mUp = cute.make_tensor(
        pup,
        cute.make_layout((16, 8, NJT, M), stride=(1, 16, TJ, 2 * K)),
    )
    mRQ = cute.make_tensor(
        prq,
        cute.make_layout(
            (4, 8, OUT_HALVES * NJT, M),
            stride=(1, 4, 32, OUT_HALVES * K // 4),
        ),
    )
    mCQ = cute.make_tensor(
        pcq,
        cute.make_layout((8, OUT_HALVES * K, NRT), stride=(1, M // 4, 8)),
    )
    mRS = cute.make_tensor(
        prs,
        cute.make_layout(M * (OUT_HALVES * K // 32)),
    )
    mCS = cute.make_tensor(
        pcs,
        cute.make_layout(OUT_HALVES * K * (M // 32)),
    )

    NT = 256 if NRT * NJT < 1184 else 128
    swiglu_mxfp8_kernel(
        mG,
        mGate,
        mUp,
        mRQ,
        mCQ,
        mRS,
        mCS,
        M,
        K,
        NJT,
        NT,
        IS_BWD,
        ROWWISE,
        COLWISE,
    ).launch(
        grid=(NRT * NJT, 1, 1),
        block=(NT, 1, 1),
        stream=stream,
        min_blocks_per_mp=(4 if NT == 256 else 8),
    )


@functools.cache
def _compile_kernel(M, K, is_bwd, rowwise, colwise, device_index):
    """Compile and cache one kernel specialization.

    Tensor addresses are runtime arguments, so a single compilation per
    (shape, flags, device) serves every subsequent call. ``device_index`` is
    part of the cache key only; compilation targets the active device.
    """
    del device_index
    from cutlass.cute.runtime import make_fake_stream

    null = cutlass.Int64(0)
    return cute.compile(
        launcher,
        null,
        null,
        null,
        null,
        null,
        null,
        make_fake_stream(),
        M,
        K,
        is_bwd,
        rowwise,
        colwise,
    )


def _validate_inputs(gated_input, grad_out=None):
    if not gated_input.is_cuda:
        raise ValueError("gated_input must be a CUDA tensor")
    if gated_input.dtype != torch.bfloat16:
        raise TypeError("gated_input must have dtype torch.bfloat16")
    if gated_input.ndim != 2 or not gated_input.is_contiguous():
        raise ValueError("gated_input must be contiguous with shape [M, 2K]")
    M, two_k = gated_input.shape
    if two_k % 2:
        raise ValueError("gated_input.shape[1] must be even")
    K = two_k // 2
    # M % 128 keeps every CTA row-tile whole; K % 128 keeps K // 32 a multiple
    # of 4. Together they mean the blocked scale layout needs no padding, so
    # every element of the scale tensors is written by the kernel.
    if M % 128 or K % 128:
        raise ValueError("M and K must be divisible by 128")
    # Every gmem layout below is built from Python ints, so CuTe emits 32-bit
    # coordinate-to-index math. The largest element offset the input layout can
    # reach is 2*K*M - K - 1; beyond INT32_MAX it wraps to a negative offset and
    # silently corrupts memory, so reject those shapes outright.
    if 2 * K * M - K - 1 > _INT32_MAX:
        raise ValueError(
            f"M={M}, K={K} exceeds the kernel's 32-bit indexing limit "
            f"(needs 2*K*M <= {_INT32_MAX})"
        )
    if grad_out is not None:
        if (
            not grad_out.is_cuda
            or grad_out.dtype != torch.bfloat16
            or not grad_out.is_contiguous()
            or tuple(grad_out.shape) != (M, K)
        ):
            raise ValueError("grad_out must be contiguous BF16 CUDA [M, K]")
        if grad_out.device != gated_input.device:
            raise ValueError(
                f"grad_out is on {grad_out.device} but gated_input is on "
                f"{gated_input.device}; both must be on the same CUDA device"
            )
    return M, K


def _ptr(tensor):
    return 0 if tensor is None else tensor.data_ptr()


@torch.no_grad()
def _launch_swiglu_mxfp8(gated_input, grad_h, outputs, rowwise, colwise):
    """Validate, compile the matching specialization, and launch into ``outputs``.

    ``outputs`` is ``(output_rowwise, output_colwise, scales_rowwise,
    scales_colwise)``, allocated by the caller. Disabled directions are
    zero-sized and are not written.
    """
    M, K = _validate_inputs(gated_input, grad_h)
    output_rowwise, output_colwise, scales_rowwise, scales_colwise = outputs

    # Compile and launch under the input's device, not whatever device happens
    # to be current -- otherwise a caller holding cuda:0 current while passing a
    # cuda:1 tensor would launch with foreign pointers.
    with torch.cuda.device(gated_input.device):
        # Wrapping the handle is cheap; caching it would keep stale entries
        # alive after a stream is destroyed and could alias a recycled handle.
        stream = CUstream(torch.cuda.current_stream(gated_input.device).cuda_stream)
        fn = _compile_kernel(
            M,
            K,
            grad_h is not None,
            rowwise,
            colwise,
            gated_input.device.index,
        )
        fn(
            _ptr(grad_h),
            gated_input.data_ptr(),
            _ptr(output_rowwise) if rowwise else 0,
            _ptr(scales_rowwise) if rowwise else 0,
            _ptr(output_colwise) if colwise else 0,
            _ptr(scales_colwise) if colwise else 0,
            stream,
        )
