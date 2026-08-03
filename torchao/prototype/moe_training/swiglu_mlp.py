# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Composite MXFP8 SwiGLU MLP for a fused w13 projection.

One autograd function covers the full dense MLP

    x -> MXFP8 w13 GEMM -> [gate | up] -> silu(gate) * up -> MXFP8 w2 GEMM

with both directions of every quantization done by the CuTeDSL kernels. The
``fuse_activation`` flag selects how the activation boundary is quantized:

* ``True``: the unified SwiGLU+MXFP8 kernel produces the rowwise and colwise
  MXFP8 copies directly; the BF16 activation ``h`` is never written to global
  memory.
* ``False``: ``h`` (forward) and ``[dGate | dUp]`` (backward) are materialized
  in BF16 and quantized by the standalone 1x32 / 32x1 CuTeDSL kernels.

Everything outside that boundary -- the w13/w2 GEMMs, their input, weight and
gradient casts -- is byte-for-byte identical between the two modes, so an A/B
comparison isolates the activation+quantization implementation.

The composite targets the normal quantized-wgrad MXFP8 recipe (RCEIL scales).
``wgrad_with_hp=True`` and shapes the CuTeDSL kernels cannot handle fall back
to the existing per-GEMM ``mx_mm`` path (or plain BF16 as a last resort)
instead of asserting.
"""

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    _mxfp8_cutedsl_kernels_available,
    mxfp8_quantize_2d_1x32_cutedsl,
    mxfp8_quantize_2d_32x1_cutedsl,
    swiglu_mxfp8_backward,
    swiglu_mxfp8_forward,
)
from torchao.prototype.moe_training.kernels.mxfp8 import (
    triton_mx_block_rearrange_2d_K_groups,
)
from torchao.prototype.moe_training.mxfp8_grouped_mm import (
    _compute_dgrad_sm100,
    _compute_fwd_sm100,
    _to_mxfp8_then_scaled_grouped_mm,
)
from torchao.prototype.moe_training.mxfp8_linear import _to_mxfp8_then_scaled_mm
from torchao.prototype.mx_formats.config import (
    MXFP8Dim1CastKernelChoice,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

__all__ = ["mxfp8_swiglu_mlp_w13", "mxfp8_swiglu_grouped_mlp_w13"]

_BLOCK_SIZE = 32
_ELEM_DTYPE = torch.float8_e4m3fn
_KERNEL_PREFERENCE = KernelPreference.AUTO
_SCALE_MODE = ScaleCalculationMode.RCEIL
_INT32_MAX = 2**31 - 1


def _wrap_rowwise(qdata, scales, orig_dtype):
    return MXTensor.from_qdata_and_scales(
        qdata,
        scales,
        orig_dtype,
        block_size=_BLOCK_SIZE,
        kernel_preference=_KERNEL_PREFERENCE,
        is_swizzled_scales=True,
    )


def _wrap_colwise(qdata, scales, orig_dtype):
    # Colwise kernel outputs are (M, N) with strides (1, M); wrapping the
    # transpose keeps qdata row-major, which torch.mm's MXFP8 dispatch
    # requires. The flat 1D blocked scales are unaffected by the transpose.
    return _wrap_rowwise(qdata.t(), scales, orig_dtype)


def _mx_rowwise(t):
    qdata, scales = mxfp8_quantize_2d_1x32_cutedsl(t, scaling_mode=_SCALE_MODE.value)
    return _wrap_rowwise(qdata, scales, t.dtype)


def _mx_colwise(t):
    # Returns the MXTensor for t.t() quantized along t's rows (32x1 blocks).
    return _to_mxfp8_dim1_kernel_wrapper(
        t,
        _BLOCK_SIZE,
        _ELEM_DTYPE,
        t.dtype,
        _KERNEL_PREFERENCE,
        MXFP8Dim1CastKernelChoice.CUTEDSL,
        _SCALE_MODE,
    )


def _pack_w13(w13):
    # (H, 2, D) with w13[:, 0] = gate and w13[:, 1] = up, packed to (2H, D)
    # with all gate rows first -- the layout whose GEMM output feeds the
    # SwiGLU kernel's [gate | up] contract.
    hidden, _, dim = w13.shape
    return w13.transpose(0, 1).reshape(2 * hidden, dim).contiguous()


def _swiglu_forward_casts(gated, fuse_activation):
    if fuse_activation:
        return swiglu_mxfp8_forward(gated, rowwise=True, colwise=True)
    k = gated.shape[1] // 2
    h = (F.silu(gated[:, :k].float()) * gated[:, k:].float()).to(gated.dtype)
    h_rw, hs_rw = mxfp8_quantize_2d_1x32_cutedsl(h, scaling_mode=_SCALE_MODE.value)
    h_cw, hs_cw = mxfp8_quantize_2d_32x1_cutedsl(h, scaling_mode=_SCALE_MODE.value)
    return h_rw, h_cw, hs_rw, hs_cw


def _swiglu_backward_casts(grad_h, gated, fuse_activation):
    if fuse_activation:
        return swiglu_mxfp8_backward(grad_h, gated, rowwise=True, colwise=True)
    k = gated.shape[1] // 2
    gate = gated[:, :k].float()
    up = gated[:, k:].float()
    grad_h_f = grad_h.float()
    # Same evaluation order as the unified kernel (which contracts `deriv`
    # into one FMA), so the two modes differ only in sigmoid lowering and
    # that contraction, not in association.
    sigmoid_gate = torch.sigmoid(gate)
    silu = gate * sigmoid_gate
    deriv = gate * (1.0 - sigmoid_gate) + 1.0
    d = torch.cat(
        [
            ((grad_h_f * up) * (sigmoid_gate * deriv)).to(gated.dtype),
            (grad_h_f * silu).to(gated.dtype),
        ],
        dim=1,
    )
    d_rw, ds_rw = mxfp8_quantize_2d_1x32_cutedsl(d, scaling_mode=_SCALE_MODE.value)
    d_cw, ds_cw = mxfp8_quantize_2d_32x1_cutedsl(d, scaling_mode=_SCALE_MODE.value)
    return d_rw, d_cw, ds_rw, ds_cw


@torch._dynamo.allow_in_graph
class _MXFP8SwiGLUMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w13, w2, fuse_activation):
        x2d = x.reshape(-1, x.shape[-1]).contiguous()
        w13_packed = _pack_w13(w13)
        gated = torch.mm(_mx_rowwise(x2d), _mx_rowwise(w13_packed).t())
        h_rw, h_cw, hs_rw, hs_cw = _swiglu_forward_casts(gated, fuse_activation)
        out = torch.mm(_wrap_rowwise(h_rw, hs_rw, x2d.dtype), _mx_rowwise(w2).t())
        ctx.save_for_backward(x2d, w13_packed, w2, gated, h_cw, hs_cw)
        ctx.fuse_activation = fuse_activation
        ctx.x_shape = x.shape
        return out.reshape(*x.shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_out):
        x2d, w13_packed, w2, gated, h_cw, hs_cw = ctx.saved_tensors
        hidden = w13_packed.shape[0] // 2
        go = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        grad_h = torch.mm(_mx_rowwise(go), _mx_colwise(w2).t())
        d_rw, d_cw, ds_rw, ds_cw = _swiglu_backward_casts(
            grad_h, gated, ctx.fuse_activation
        )
        grad_x = torch.mm(_wrap_rowwise(d_rw, ds_rw, go.dtype), _mx_colwise(w13_packed).t())
        grad_w13_packed = torch.mm(
            _wrap_colwise(d_cw, ds_cw, go.dtype), _mx_colwise(x2d).t()
        )
        grad_w2 = torch.mm(_mx_colwise(go), _wrap_colwise(h_cw, hs_cw, go.dtype).t())
        grad_w13 = grad_w13_packed.view(2, hidden, -1).transpose(0, 1).contiguous()
        return grad_x.reshape(ctx.x_shape), grad_w13, grad_w2, None


def _fused_path_ok(x, w13, w2):
    if not _mxfp8_cutedsl_kernels_available:
        return False
    if isinstance(x, DTensor) or isinstance(w13, DTensor) or isinstance(w2, DTensor):
        return False
    if not x.is_cuda:
        return False
    if x.dtype != torch.bfloat16 or w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return False
    if w13.ndim != 3 or w13.shape[1] != 2 or w2.ndim != 2:
        return False
    hidden, _, dim = w13.shape
    n = w2.shape[0]
    m = x.numel() // x.shape[-1]
    if x.shape[-1] != dim or w2.shape[1] != hidden:
        return False
    if m % 128 != 0 or hidden % 128 != 0 or dim % 128 != 0 or n % 128 != 0:
        return False
    # 32-bit index-math limit over BOTH A/B arms: the unified kernel's input
    # layout reaches element 2*hidden*m - hidden - 1, but the unfused arm's
    # standalone casts of the (m, 2*hidden) backward tensor reach
    # 2*hidden*m - 1, and those kernels do not validate. Gate on the max so
    # the two arms accept identical shapes.
    if 2 * hidden * m - 1 > _INT32_MAX:
        return False
    return True


def _mx_mm_path_ok(x, w13, w2):
    if isinstance(x, DTensor) or isinstance(w13, DTensor) or isinstance(w2, DTensor):
        return False
    if not x.is_cuda or x.dtype != torch.bfloat16:
        return False
    hidden, _, dim = w13.shape
    m = x.numel() // x.shape[-1]
    return (
        m % 32 == 0 and hidden % 32 == 0 and dim % 32 == 0 and w2.shape[0] % 32 == 0
    )


def _unfused_mlp(x, w13, w2, wgrad_with_hp):
    if _mx_mm_path_ok(x, w13, w2):
        gate_up = _to_mxfp8_then_scaled_mm(
            x, _pack_w13(w13), _KERNEL_PREFERENCE, _SCALE_MODE, wgrad_with_hp
        )
        gate, up = gate_up.chunk(2, dim=-1)
        h = F.silu(gate) * up
        return _to_mxfp8_then_scaled_mm(
            h, w2, _KERNEL_PREFERENCE, _SCALE_MODE, wgrad_with_hp
        )
    gate, up = torch.einsum("...d,hgd->...hg", x, w13).unbind(-1)
    return F.linear(F.silu(gate.float()).to(x.dtype) * up, w2)


def mxfp8_swiglu_mlp_w13(
    x, w13, down_weight, *, fuse_activation=True, wgrad_with_hp=False
):
    """Dense MXFP8 SwiGLU MLP with a fused (H, 2, D) w13 weight.

    Args:
        x: BF16 input of shape (..., D).
        w13: BF16 fused gate/up weight of shape (H, 2, D); w13[:, 0] is the
            gate (w1) and w13[:, 1] the up (w3) projection.
        down_weight: BF16 down-projection weight of shape (D_out, H).
        fuse_activation: quantize the SwiGLU boundary with the unified
            SwiGLU+MXFP8 kernel instead of standalone BF16 + cast kernels.
        wgrad_with_hp: the composite only implements the quantized-wgrad
            recipe; when True, fall back to the per-GEMM mx_mm path, which
            computes weight gradients in high precision.

    Returns:
        BF16 tensor of shape (..., D_out).
    """
    if wgrad_with_hp or not _fused_path_ok(x, w13, down_weight):
        return _unfused_mlp(x, w13, down_weight, wgrad_with_hp)
    return _MXFP8SwiGLUMLP.apply(x, w13, down_weight, fuse_activation)


def _pack_w13_grouped(w13):
    # (E, F, 2, D) with w13[:, :, 0] = gate and w13[:, :, 1] = up, packed to
    # (E, 2F, D) with all gate rows first per expert (the [gate | up] layout
    # the SwiGLU kernel consumes). The reshape of the transposed view copies.
    e, f, _, d = w13.shape
    return w13.transpose(1, 2).reshape(e, 2 * f, d)


def _reblock_scales_k_groups(scales, n_rows, m_total, offs):
    # The CuTeDSL kernels emit blocked scales in full-tensor row-block-major
    # tile order; a 2d-2d grouped GEMM contracting over tokens needs the tiles
    # regrouped per token group (row-block-major within each group). With every
    # group a multiple of 128 rows the two layouts hold identical (128, 4)
    # tiles, so this is a pure tile gather.
    rb = n_rows // 128
    cb = m_total // 128
    ends = (offs // 128).long()
    starts = torch.cat([ends.new_zeros(1), ends[:-1]])
    sizes = (ends - starts).clamp(min=1)
    t = torch.arange(rb * cb, device=scales.device)
    # The dispatcher may pad the token buffer globally past offs[-1]; tiles in
    # that tail belong to no group and are never read by the grouped GEMM, so
    # clamping them anywhere in bounds is enough to keep the gather valid.
    g = torch.searchsorted(ends * rb, t, right=True).clamp(max=ends.numel() - 1)
    local = t - starts[g] * rb
    src = ((local // sizes[g]) * cb + starts[g] + local % sizes[g]).clamp(
        max=rb * cb - 1
    )
    return scales.view(rb * cb, 512)[src].view(n_rows, -1)


def _wgrad_k_groups(a_qdata, a_scales, b, offs, out_dtype):
    # grad[e] = a[start:end].T @ b[start:end] for each token group. `a` arrives
    # colwise-quantized from the SwiGLU boundary ((M, Ka) with strides (1, M)
    # plus flat blocked scales); `b` gets the same GEMM-operand colwise cast the
    # existing grouped wgrad path uses.
    m, ka = a_qdata.shape
    b_t_mx = _to_mxfp8_dim1_kernel_wrapper(
        b,
        _BLOCK_SIZE,
        _ELEM_DTYPE,
        b.dtype,
        _KERNEL_PREFERENCE,
        MXFP8Dim1CastKernelChoice.CUDA,
        _SCALE_MODE,
    )
    b_scales = triton_mx_block_rearrange_2d_K_groups(b_t_mx.scale, offs // _BLOCK_SIZE)
    a_scales_2d = _reblock_scales_k_groups(a_scales, ka, m, offs)
    return torch._scaled_grouped_mm(
        a_qdata.t(),
        b_t_mx.qdata.transpose(-2, -1),
        a_scales_2d,
        b_scales,
        offs=offs,
        out_dtype=out_dtype,
    )


@torch._dynamo.allow_in_graph
class _MXFP8SwiGLUGroupedMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w13, w2_t, offs, fuse_activation):
        x = x.contiguous()
        w13_packed = _pack_w13_grouped(w13)
        gated = _compute_fwd_sm100(
            x, w13_packed.transpose(-2, -1), offs, _BLOCK_SIZE, x.dtype, _SCALE_MODE
        )
        h_rw, h_cw, hs_rw, hs_cw = _swiglu_forward_casts(gated, fuse_activation)
        out = _compute_fwd_sm100(
            _wrap_rowwise(h_rw, hs_rw, x.dtype),
            w2_t,
            offs,
            _BLOCK_SIZE,
            x.dtype,
            _SCALE_MODE,
        )
        ctx.save_for_backward(x, w13_packed, w2_t, offs, gated, h_cw, hs_cw)
        ctx.fuse_activation = fuse_activation
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w13_packed, w2_t, offs, gated, h_cw, hs_cw = ctx.saved_tensors
        e, two_f, d = w13_packed.shape
        go = grad_out.contiguous()
        grad_h = _compute_dgrad_sm100(
            go, w2_t, offs, _BLOCK_SIZE, go.dtype, _SCALE_MODE
        )
        d_rw, d_cw, ds_rw, ds_cw = _swiglu_backward_casts(
            grad_h, gated, ctx.fuse_activation
        )
        grad_x = _compute_dgrad_sm100(
            _wrap_rowwise(d_rw, ds_rw, go.dtype),
            w13_packed.transpose(-2, -1),
            offs,
            _BLOCK_SIZE,
            go.dtype,
            _SCALE_MODE,
        )
        grad_w13_packed = _wgrad_k_groups(d_cw, ds_cw, x, offs, go.dtype)
        grad_w2_t = _wgrad_k_groups(h_cw, hs_cw, go, offs, go.dtype)
        grad_w13 = grad_w13_packed.view(e, 2, two_f // 2, d).transpose(1, 2)
        return grad_x, grad_w13, grad_w2_t, None, None


def _grouped_path_ok(x, w13, w2_t, offs):
    if not _mxfp8_cutedsl_kernels_available:
        return False
    if any(isinstance(t, DTensor) for t in (x, w13, w2_t)):
        return False
    if not x.is_cuda or x.ndim != 2:
        return False
    if x.dtype != torch.bfloat16 or w13.dtype != torch.bfloat16 or w2_t.dtype != torch.bfloat16:
        return False
    if w13.ndim != 4 or w13.shape[2] != 2 or w2_t.ndim != 3:
        return False
    e, f, _, d = w13.shape
    m = x.shape[0]
    d_out = w2_t.shape[2]
    if x.shape[1] != d or w2_t.shape[:2] != (e, f) or offs.shape != (e,):
        return False
    if f % 128 != 0 or d % 128 != 0 or d_out % 128 != 0:
        return False
    # Group boundaries must additionally be 128-row aligned (the token
    # dispatcher's pad_multiple guarantees it); checking offs here would sync.
    # M is routing-dependent under compile (an unbacked SymInt, which type
    # tests cannot tell apart from int inside traced code), so the M
    # conditions use identity tests: literal bools keep the trace-time
    # fallback, symbolic ones become deferred runtime asserts. The m >= 128
    # and m % 32 forms are redundant with m % 128 (plus non-emptiness) but
    # must be recorded separately: downstream cast-kernel wrappers and GEMM
    # metas check exactly those forms, and the symbolic engine resolves them
    # by expression match / value range, not by deriving them from mod-128.
    # The last condition is the 32-bit index-math limit over BOTH A/B arms
    # (the unfused arm's standalone casts of the (m, 2f) backward tensor
    # reach element 2*f*m - 1, slightly past the unified kernel's own input
    # bound, and those kernels do not validate).
    for cond in (
        m >= 128,
        m % 128 == 0,
        m % 32 == 0,
        2 * f * m - 1 <= _INT32_MAX,
    ):
        if cond is False:
            return False
        if cond is not True:
            torch._check(cond)
    return True


def _unfused_grouped_mlp(x, w13, w2_t, offs, wgrad_with_hp):
    # The per-GEMM grouped mx path's cast kernels need every dim to be a
    # multiple of 128, so it only serves the wgrad_with_hp (and non-shape
    # guard) fallbacks; unsupported shapes drop to BF16 grouped GEMMs.
    e, f, _, d = w13.shape
    m = x.shape[0]
    mx_ok = (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and f % 128 == 0
        and d % 128 == 0
        and w2_t.shape[2] % 128 == 0
    )
    if mx_ok:
        # Same identity-test treatment of the routing-dependent M as
        # _grouped_path_ok, for when this fallback is itself compiled.
        m_ok = m % 128 == 0
        if m_ok is False:
            mx_ok = False
        elif m_ok is not True:
            torch._check(m_ok)
    if mx_ok:
        gated = _to_mxfp8_then_scaled_grouped_mm(
            x.contiguous(),
            _pack_w13_grouped(w13).transpose(-2, -1),
            offs,
            kernel_preference=_KERNEL_PREFERENCE,
            wgrad_with_hp=wgrad_with_hp,
            scale_calculation_mode=_SCALE_MODE,
        )
        h = (F.silu(gated[:, :f].float()) * gated[:, f:].float()).to(gated.dtype)
        return _to_mxfp8_then_scaled_grouped_mm(
            h,
            w2_t,
            offs,
            kernel_preference=_KERNEL_PREFERENCE,
            wgrad_with_hp=wgrad_with_hp,
            scale_calculation_mode=_SCALE_MODE,
        )
    gated = torch._grouped_mm(x, _pack_w13_grouped(w13).transpose(-2, -1), offs=offs)
    h = (F.silu(gated[:, :f].float()) * gated[:, f:].float()).to(gated.dtype)
    return torch._grouped_mm(h, w2_t, offs=offs)


def mxfp8_swiglu_grouped_mlp_w13(
    x, w13, down_weight_t, offs, *, fuse_activation=True, wgrad_with_hp=False
):
    """Grouped-expert MXFP8 SwiGLU MLP with a fused (E, F, 2, D) w13 weight.

    Args:
        x: BF16 token rows of shape (M, D) in expert-major order, with every
            expert's group padded to a multiple of 128 rows (padded rows must
            be zero) so 32x1 scale blocks never cross group boundaries.
        w13: BF16 fused gate/up weight of shape (E, F, 2, D); w13[:, :, 0] is
            the gate and w13[:, :, 1] the up projection.
        down_weight_t: BF16 down-projection weight of shape (E, F, D_out) in
            per-expert column-major layout (a transposed view of (E, D_out, F)).
        offs: int32 group end offsets of shape (E,), each a multiple of 128.
        fuse_activation: quantize the SwiGLU boundary with the unified
            SwiGLU+MXFP8 kernel instead of standalone BF16 + cast kernels.
        wgrad_with_hp: the composite only implements the quantized-wgrad
            recipe; when True, fall back to the per-GEMM grouped path, which
            computes weight gradients in high precision.

    Returns:
        BF16 tensor of shape (M, D_out).
    """
    if wgrad_with_hp or not _grouped_path_ok(x, w13, down_weight_t, offs):
        return _unfused_grouped_mlp(x, w13, down_weight_t, offs, wgrad_with_hp)
    return _MXFP8SwiGLUGroupedMLP.apply(x, w13, down_weight_t, offs, fuse_activation)
