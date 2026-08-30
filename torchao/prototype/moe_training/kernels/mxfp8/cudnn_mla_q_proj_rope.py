# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 MLA Q-projection + RoPE + dual-quantize op over the cuDNN-frontend kernel.

One custom op, one launch of the ``cudnn.gemm_proj_rope_mxfp8_wrapper_sm100``
kernel from the standalone cudnn-frontend python package (>= 1.27, Blackwell
SM100+; no TransformerEngine dependency), ported from the fusion TE PR #3303
wires for DeepSeek-V3 MLA MXFP8 training; the matching public wrapper lives at
the bottom of this module:

* :func:`mxfp8_mla_q_proj_rope_cudnn` -- Q-projection GEMM (BF16, or MXFP8 on
  prequantized operands) + per-head YARN RoPE on the trailing 64 features of
  each 192-feature head + rowwise 1x32 AND columnwise 32x1 MXFP8 RCEIL
  quantization of the projected Q, with the GEMM accumulator staged through
  BF16 before RoPE (matching the unfused BF16-output GEMM -> RoPE chain).

CONTRACT: ``tokens`` (the flattened token count) must be a positive multiple
of **128** -- the kernel tiles one head per CTA with ``TILE_M = 128`` and has
no tail handling. The head geometry is FIXED at ``HEAD_DIM = 192``
(``128`` pass-through features + ``64`` rotary features); the projected
feature count must equal ``num_heads * 192``. On the MXFP8 input path
``in_features`` must be a multiple of **128** (the kernel derives its
compact-scale words as ``in_features // 128``) and ``num_heads`` must be
EVEN -- the kernel's SFB scale-relay pairs heads, and an odd count reads past
the weight-scale buffer SILENTLY. ``num_heads`` must also be at least **8**:
the kernel's own ``check_support`` accepts 2/4/6 heads, but the outputs are
numerically corrupt there (measured 1.7-10.8 dB with 12-31% code mismatches
on at least one dtype path), so :func:`is_supported` refuses them.

RoPE convention (Megatron YARN): the rotary tail is read INTERLEAVED
(``x1 = q_pe[0::2]``, ``x2 = q_pe[1::2]``) and written HALF-CONCATENATED
``[x1*cos - x2*sin ; x2*cos + x1*sin]``. ``cos``/``sin`` are ``[tokens, 64]``
BF16 tables whose halves DUPLICATE the 32 frequencies
(``cos[:, :32] == cos[:, 32:]``): output half ``j`` uses table column ``j``
for BOTH product terms and output half ``32 + j`` uses column ``32 + j``, so
the rotation identity holds ONLY for duplicated tables -- a non-duplicated
table produces a silent non-rotation. YARN lives in the table values (and any
mscale in the caller's softmax scale); the op applies no scaling of its own.

All tensors must be CUDA, same device, C-contiguous. Scale tensors are uint8
E8M0, unswizzled: rowwise ``[tokens, num_heads, 6]`` (block-32 along
HEAD_DIM), columnwise ``[tokens // 32, num_heads, 192]`` (block-32 along
tokens, codes NOT transposed).

Importing this module registers the ``torchao::`` custom op; the ``cudnn``
package itself is imported lazily inside the op body at first real launch.
:func:`is_supported` is the static shape predicate to call before selecting
this op.
"""

from typing import Optional, Tuple

import torch

__all__ = [
    "DIM_ALIGNMENT",
    "HEAD_DIM",
    "MIN_NUM_HEADS",
    "QK_NOPE_HEAD_DIM",
    "QK_ROPE_HEAD_DIM",
    "SCALE_BLOCK_SIZE",
    "TOKEN_ALIGNMENT",
    "is_supported",
    "mxfp8_mla_q_proj_rope_cudnn",
]

# MXFP8 scaling block: 32 values share one E8M0 scale.
SCALE_BLOCK_SIZE = 32
# Fixed head geometry baked into the kernel epilogue.
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
# Flattened-token granularity (the kernel's TILE_M; no tail handling).
TOKEN_ALIGNMENT = 128
# in_features granularity on the MXFP8 input path (compact-scale words).
DIM_ALIGNMENT = 128
# Smallest head count with correct kernel output: 2/4/6 pass the kernel's own
# check_support but produce corrupt values (see the module docstring).
MIN_NUM_HEADS = 8

_E4M3 = torch.float8_e4m3fn
_BLOCK = SCALE_BLOCK_SIZE


def is_supported(
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    in_features: int,
    num_heads: int,
) -> bool:
    """True when the head geometry matches the kernel's fixed 128+64 epilogue,
    ``in_features`` is a positive multiple of 128, and ``num_heads`` is an
    EVEN count of at least :data:`MIN_NUM_HEADS` (the MXFP8 path's SFB
    scale-relay pairs heads -- an odd count reads out of bounds silently --
    and head counts below 8 produce corrupt output despite passing the
    kernel's own ``check_support``). Integration code must ALSO guarantee the
    runtime contract (flattened tokens a positive multiple of 128,
    C-contiguous CUDA tensors): token counts are runtime values and are not
    checkable here."""
    return (
        qk_nope_head_dim == QK_NOPE_HEAD_DIM
        and qk_rope_head_dim == QK_ROPE_HEAD_DIM
        and in_features > 0
        and in_features % DIM_ALIGNMENT == 0
        and num_heads >= MIN_NUM_HEADS
        and num_heads % 2 == 0
    )


def _check_normalized(
    tensor: torch.Tensor, *, name: str, shape: tuple, dtype: torch.dtype
) -> torch.Tensor:
    """Guard against wrapper-output metadata drifting from the fake spec."""
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
        raise RuntimeError(
            f"cudnn wrapper output {name} has shape {tuple(tensor.shape)} dtype "
            f"{tensor.dtype}; expected {tuple(shape)} {dtype}. The installed "
            "cudnn-frontend's output contract changed; the registered fake no "
            "longer matches eager."
        )
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


def _stream(device: torch.device):
    from cuda.bindings import driver as cuda_driver

    return cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)


def _output_specs(tokens: int, num_heads: int):
    return (
        ("q_row_q", (tokens, num_heads, HEAD_DIM), _E4M3),
        ("q_row_sf", (tokens, num_heads, HEAD_DIM // _BLOCK), torch.uint8),
        ("q_col_q", (tokens, num_heads, HEAD_DIM), _E4M3),
        ("q_col_sf", (tokens // _BLOCK, num_heads, HEAD_DIM), torch.uint8),
    )


def _allocate_from_specs(specs, device) -> Tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(shape, dtype=dtype, device=device) for _, shape, dtype in specs
    )


def _proj_dims(x, w):
    tokens = x.shape[0]
    num_heads = w.shape[0] // HEAD_DIM
    return tokens, num_heads


@torch.library.custom_op("torchao::mxfp8_mla_q_proj_rope_cudnn", mutates_args=())
def _mxfp8_mla_q_proj_rope_cudnn(
    x: torch.Tensor,
    w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q-projection GEMM + per-head YARN RoPE + dual MXFP8 RCEIL quantization
    (one cuDNN launch).

    Inputs (all CUDA, one device, C-contiguous; ``tokens`` a positive multiple
    of 128 -- caller invariant):
      x        ``[tokens, in_features]``. BF16 selects the BF16-GEMM path;
               E4M3 (with both scales present) selects the MXFP8-GEMM path.
      w        projection weight ``[num_heads * 192, in_features]`` ([out, in],
               the ``nn.Linear`` native layout), SAME dtype as ``x``.
      cos/sin  BF16 ``[tokens, 64]`` duplicated-freq halves tables
               (``cos[:, :32] == cos[:, 32:]`` -- see the module docstring for
               the per-output-half semantics; a non-duplicated table is a
               silent non-rotation).
      x_scale  MXFP8 path only: uint8 E8M0 rowwise 1x32 scales
               ``[tokens, in_features // 32]``; ``in_features`` %128
               (caller invariant).
      w_scale  MXFP8 path only: uint8 E8M0 ``[num_heads * 192,
               in_features // 32]``; ``num_heads`` must be EVEN (odd counts
               read past this buffer silently -- caller invariant).

    Returns ``(q_row_q, q_row_sf, q_col_q, q_col_sf)``:
      q_row_q  E4M3 ``[tokens, num_heads, 192]`` contiguous, rowwise 1x32
               quantized; q_row_sf uint8 ``[tokens, num_heads, 6]``.
      q_col_q  E4M3 ``[tokens, num_heads, 192]`` contiguous columnwise 32x1
               quantized bytes (un-transposed kernel layout); q_col_sf uint8
               ``[tokens // 32, num_heads, 192]``. Scales unswizzled.

    The GEMM accumulator is staged through BF16 before the fp32 RoPE and the
    quantize, so numerics match the unfused BF16-output GEMM -> RoPE ->
    quantize chain, not an fp32 end-to-end fusion.
    """
    tokens, num_heads = _proj_dims(x, w)
    specs = _output_specs(tokens, num_heads)

    import cudnn

    out = cudnn.gemm_proj_rope_mxfp8_wrapper_sm100(
        x,
        w,
        cos,
        sin,
        x_scale=x_scale,
        w_scale=w_scale,
        w_out_in=True,
        stream=_stream(x.device),
    )
    results = (
        out["out_fp8_row"],
        out["out_scales_row"],
        out["out_fp8_col"],
        out["out_scales_col"],
    )
    return tuple(
        _check_normalized(t, name=spec[0], shape=spec[1], dtype=spec[2])
        for t, spec in zip(results, specs)
    )


@_mxfp8_mla_q_proj_rope_cudnn.register_fake
def _(x, w, cos, sin, x_scale=None, w_scale=None):
    tokens, num_heads = _proj_dims(x, w)
    return _allocate_from_specs(_output_specs(tokens, num_heads), x.device)


def mxfp8_mla_q_proj_rope_cudnn(x, w, cos, sin, x_scale=None, w_scale=None):
    """Q-projection GEMM + per-head YARN RoPE + dual MXFP8 quantization.

    See ``torchao::mxfp8_mla_q_proj_rope_cudnn`` for the full ABI. ``w`` is
    ``[num_heads * 192, in_features]`` in the SAME dtype as ``x`` (BF16, or
    E4M3 with both uint8 E8M0 scales); ``cos``/``sin`` are BF16
    ``[tokens, 64]`` duplicated-freq halves tables; ``tokens`` must be a
    positive multiple of 128. Returns
    ``(q_row_q [tokens, nh, 192], q_row_sf [tokens, nh, 6],
    q_col_q [tokens, nh, 192], q_col_sf [tokens // 32, nh, 192])``.
    """
    return torch.ops.torchao.mxfp8_mla_q_proj_rope_cudnn(
        x, w, cos, sin, x_scale, w_scale
    )
