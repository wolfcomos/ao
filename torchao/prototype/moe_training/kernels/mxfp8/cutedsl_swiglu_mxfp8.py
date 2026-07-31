# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects._cute_nvgpu_enum_gen import AddressSpace
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

THREADS = 128
TR = 32       # rows per CTA tile
TJ = 128      # j-columns per CTA tile
SP = 36       # padded row-stride (words) of the transposed smem cache
SH = 64 * SP  # words per half


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


@dsl_user_op
def _prmt_even(a, b, *, loc=None, ip=None):
    """prmt.b32 with selector 0x6420: select bytes [0,2,4,6] from a pair."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(a).ir_value(loc=loc, ip=ip),
                cutlass.Int32(b).ir_value(loc=loc, ip=ip),
            ],
            "prmt.b32 $0, $1, $2, 0x6420;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _prmt_odd(a, b, *, loc=None, ip=None):
    """prmt.b32 with selector 0x7531: select bytes [1,3,5,7] from a pair."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(a).ir_value(loc=loc, ip=ip),
                cutlass.Int32(b).ir_value(loc=loc, ip=ip),
            ],
            "prmt.b32 $0, $1, $2, 0x7531;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
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
def _max_bf16x2(a, b, *, loc=None, ip=None):
    """Elementwise max of two packed bf16x2 words (operands must be non-negative)."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(a).ir_value(loc=loc, ip=ip),
                cutlass.Int32(b).ir_value(loc=loc, ip=ip),
            ],
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _amax_bf16x2(a, b, *, loc=None, ip=None):
    """max(|a|,|b|) per bf16 lane; result sign bits are junk and must be masked."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Int32(a).ir_value(loc=loc, ip=ip),
                cutlass.Int32(b).ir_value(loc=loc, ip=ip),
            ],
            "max.xorsign.abs.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _rcp2(d0: F32, d1: F32):
    """Correctly-rounded 1/d for d >= 1: rcp.approx + two Markstein FMA steps."""
    one = F32(1.0)
    x0 = cute.arch.rcp_approx(d0)
    x1 = cute.arch.rcp_approx(d1)
    n0, n1 = cute.arch.sub_packed_f32x2((F32(0.0), F32(0.0)), (d0, d1))
    e0, e1 = cute.arch.fma_packed_f32x2((n0, n1), (x0, x1), (one, one))
    x0, x1 = cute.arch.fma_packed_f32x2((x0, x1), (e0, e1), (x0, x1))
    e0, e1 = cute.arch.fma_packed_f32x2((n0, n1), (x0, x1), (one, one))
    x0, x1 = cute.arch.fma_packed_f32x2((x0, x1), (e0, e1), (x0, x1))
    return x0, x1


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

    for p in cutlass.range_constexpr(NP):
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

            d0 = ONE + cute.math.exp(-g0)
            d1 = ONE + cute.math.exp(-g1)
            s0, s1 = _rcp2(d0, d1)
            silu0, silu1 = cute.arch.mul_packed_f32x2((g0, g1), (s0, s1))

            if cutlass.const_expr(IS_BWD):
                h0 = fh[e].to(F32)
                h1 = fh[e + 1].to(F32)
                one_minus_s0, one_minus_s1 = cute.arch.sub_packed_f32x2(
                    (ONE, ONE), (s0, s1)
                )
                gx0, gx1 = cute.arch.mul_packed_f32x2(
                    (g0, g1), (one_minus_s0, one_minus_s1)
                )
                deriv0 = ONE + gx0
                deriv1 = ONE + gx1
                silu_grad0, silu_grad1 = cute.arch.mul_packed_f32x2(
                    (s0, s1), (deriv0, deriv1)
                )
                grad_up0, grad_up1 = cute.arch.mul_packed_f32x2(
                    (h0, h1), (u0, u1)
                )
                dg0, dg1 = cute.arch.mul_packed_f32x2(
                    (grad_up0, grad_up1), (silu_grad0, silu_grad1)
                )
                du0, du1 = cute.arch.mul_packed_f32x2(
                    (h0, h1), (silu0, silu1)
                )
                w0[e >> 1] = _bf16x2(dg1, dg0)
                w1[e >> 1] = _bf16x2(du1, du0)
            else:
                y0, y1 = cute.arch.mul_packed_f32x2(
                    (silu0, silu1), (u0, u1)
                )
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


_CACHE = {}
_STREAMS = {}


def _ceil_div(a, b):
    return (a + b - 1) // b


def _validate_inputs(gated_input, grad_h=None):
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
    if M % 128 or K % 128:
        raise ValueError("M and K must be divisible by 128")
    if grad_h is not None:
        if (
            not grad_h.is_cuda
            or grad_h.dtype != torch.bfloat16
            or not grad_h.is_contiguous()
            or tuple(grad_h.shape) != (M, K)
        ):
            raise ValueError("grad_h must be contiguous BF16 CUDA [M, K]")
    return M, K


def _allocate_outputs(M, K, out_halves, rowwise, colwise, device):
    out_k = out_halves * K
    row_qdata = row_scales = None
    col_qdata = col_scales = None

    if rowwise:
        row_qdata = torch.empty(
            (M, out_k),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        row_scale_rows = _ceil_div(M, 128) * 128
        row_scale_cols = _ceil_div(out_k // 32, 4) * 4
        row_scales = torch.empty(
            (row_scale_rows, row_scale_cols),
            device=device,
            dtype=torch.float8_e8m0fnu,
        )

    if colwise:
        col_qdata = torch.empty_strided(
            (M, out_k),
            (1, M),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        col_scale_rows = _ceil_div(out_k, 128) * 128
        col_scale_cols = _ceil_div(M // 32, 4) * 4
        col_scales = torch.empty(
            col_scale_rows * col_scale_cols,  # flat 1D, matching reference layout
            device=device,
            dtype=torch.float8_e8m0fnu,
        )

    return row_qdata, row_scales, col_qdata, col_scales


def _ptr(tensor):
    return 0 if tensor is None else tensor.data_ptr()


@torch.no_grad()
def _run(gated_input, grad_h, rowwise, colwise):
    if not rowwise and not colwise:
        raise ValueError("at least one of rowwise or colwise must be enabled")
    M, K = _validate_inputs(gated_input, grad_h)
    is_bwd = grad_h is not None
    out_halves = 2 if is_bwd else 1
    outputs = _allocate_outputs(
        M,
        K,
        out_halves,
        rowwise,
        colwise,
        gated_input.device,
    )
    row_qdata, row_scales, col_qdata, col_scales = outputs

    key = (is_bwd, rowwise, colwise, M, K)
    fn = _CACHE.get(key)
    stream_handle = torch.cuda.current_stream().cuda_stream
    stream = _STREAMS.get(stream_handle)
    if stream is None:
        stream = CUstream(stream_handle)
        _STREAMS[stream_handle] = stream

    args = (
        _ptr(grad_h),
        gated_input.data_ptr(),
        _ptr(row_qdata),
        _ptr(row_scales),
        _ptr(col_qdata),
        _ptr(col_scales),
    )
    if fn is None:
        fn = cute.compile(
            launcher,
            *(cutlass.Int64(arg) for arg in args),
            stream,
            M,
            K,
            is_bwd,
            rowwise,
            colwise,
        )
        _CACHE[key] = fn
    fn(*args, stream)
    return outputs


def swiglu_mxfp8_forward_row(gated_input):
    """Forward SwiGLU fused with MXFP8 rowwise (1x32) quantization."""
    row_qdata, row_scales, _, _ = _run(gated_input, None, True, False)
    return row_qdata, row_scales


def swiglu_mxfp8_forward_col(gated_input):
    """Forward SwiGLU fused with MXFP8 colwise (32x1) quantization."""
    _, _, col_qdata, col_scales = _run(gated_input, None, False, True)
    return col_qdata, col_scales


def swiglu_mxfp8_forward_bidirectional(gated_input):
    """Forward SwiGLU fused with both MXFP8 rowwise and colwise quantization."""
    return _run(gated_input, None, True, True)


def swiglu_mxfp8_backward_row(grad_h, gated_input):
    """Backward SwiGLU fused with MXFP8 rowwise (1x32) quantization.

    Returns [M, 2K] qdata with dGate in first K columns and dUp in last K columns.
    """
    row_qdata, row_scales, _, _ = _run(gated_input, grad_h, True, False)
    return row_qdata, row_scales


def swiglu_mxfp8_backward_col(grad_h, gated_input):
    """Backward SwiGLU fused with MXFP8 colwise (32x1) quantization."""
    _, _, col_qdata, col_scales = _run(gated_input, grad_h, False, True)
    return col_qdata, col_scales


def swiglu_mxfp8_backward_bidirectional(grad_h, gated_input):
    """Backward SwiGLU fused with both MXFP8 rowwise and colwise quantization."""
    return _run(gated_input, grad_h, True, True)
