# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the MXFP8 MLA Q-projection + RoPE + dual-quantize custom op.

The op wraps the cudnn-frontend package's CuTe DSL kernel
(``cudnn.gemm_proj_rope_mxfp8_wrapper_sm100``).

Both dtype paths (BF16 GEMM and MXFP8 GEMM on prequantized operands) are
validated against a standalone PyTorch reference (``_ref_mla_q_proj_rope``:
dequantization, an FP64 ``torch`` matmul, the BF16 staging round, the
halves-out RoPE, and ``to_mx`` requantization only) under fixed gates; the
gate rationale and measured margins live next to the gate constants below.

The reference requantizes with the same ``to_mx``/RCEIL step the kernel
applies, so the comparison is matched-requant: both sides carry the identical
output quantization and agreement is GEMM/RoPE exactness, not a quantization
band.
"""

import math

import pytest
import torch

from torchao.utils import is_sm_version

# Exactly SM 10.0, matching the op module's contract: the wrapped cudnn
# kernel is *_sm100-specific.
if not (torch.cuda.is_available() and is_sm_version(10, 0)):
    pytest.skip(
        "MXFP8 MLA Q-projection + RoPE requires CUDA SM100",
        allow_module_level=True,
    )

try:
    # Importing the module registers the torchao:: custom op.
    import torchao.prototype.moe_training.kernels.mxfp8.cudnn_mla_q_proj_rope  # noqa: F401
except ImportError:
    pytest.skip(
        "installed torchao does not provide the cudnn_mla_q_proj_rope module",
        allow_module_level=True,
    )

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.kernels.mxfp8.cudnn_mla_q_proj_rope import (
    HEAD_DIM,
    QK_NOPE_HEAD_DIM,
    SCALE_BLOCK_SIZE,
)
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import get_fp_scale, to_mx

_E4M3 = torch.float8_e4m3fn
_E8M0 = torch.float8_e8m0fnu
_BLOCK = SCALE_BLOCK_SIZE
_RCEIL = ScaleCalculationMode.RCEIL

_OPS = torch.ops.torchao


# ---------------------------------------------------------------------------
# Pure-torch quantization / dequantization helpers.
# ---------------------------------------------------------------------------


def _quant_rowwise(x: torch.Tensor):
    """[M, K] -> (qdata [M, K] e4m3, uint8 e8m0 scales [M, K/32], unswizzled)."""
    s, q = to_mx(x, _E4M3, _BLOCK, scaling_mode=_RCEIL)
    return q, s.view(_E8M0).view(torch.uint8)


def _dequant_input(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """[M, K] e4m3 + [M, K/32] rowwise scales -> FP64 (feeds the FP64 GEMM)."""
    s = get_fp_scale(sf.view(_E8M0)).double()
    return q.to(torch.float64) * s.repeat_interleave(_BLOCK, dim=1)


def _dequant_rowwise(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """[T, nh, 192] codes + [T, nh, 6] scales (block-32 along HEAD_DIM) -> f32."""
    s = get_fp_scale(sf.view(_E8M0)).double()
    return (q.to(torch.float64) * s.repeat_interleave(_BLOCK, dim=-1)).float()


def _dequant_colwise(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """[T, nh, 192] codes + [T/32, nh, 192] scales (block-32 along tokens) -> f32."""
    s = get_fp_scale(sf.view(_E8M0)).double()
    return (q.to(torch.float64) * s.repeat_interleave(_BLOCK, dim=0)).float()


def _quant_out_rowwise(y: torch.Tensor):
    """FP32 [T, nh, 192] -> rowwise 1x32 codes + scales in the op's layouts."""
    tokens, nh, _ = y.shape
    s, q = to_mx(y.reshape(tokens * nh, HEAD_DIM), _E4M3, _BLOCK, scaling_mode=_RCEIL)
    return (
        q.reshape(tokens, nh, HEAD_DIM),
        s.view(_E8M0).view(torch.uint8).reshape(tokens, nh, HEAD_DIM // _BLOCK),
    )


def _quant_out_colwise(y: torch.Tensor):
    """FP32 [T, nh, 192] -> columnwise 32x1 codes (un-transposed) + scales."""
    tokens, nh, _ = y.shape
    s_t, q_t = to_mx(
        y.reshape(tokens, nh * HEAD_DIM).t().contiguous(),
        _E4M3,
        _BLOCK,
        scaling_mode=_RCEIL,
    )
    codes = q_t.t().contiguous().reshape(tokens, nh, HEAD_DIM)
    scales = (
        s_t.view(_E8M0)
        .view(torch.uint8)
        .t()
        .contiguous()
        .reshape(tokens // _BLOCK, nh, HEAD_DIM)
    )
    return codes, scales


def _rope_tables(tokens: int, device):
    """Random-angle BF16 tables in the kernel's duplicated-freq halves layout.

    The kernel applies table column ``j`` to output half ``j`` and column
    ``32 + j`` to half ``32 + j``; the rotation identity requires
    ``cos[:, :32] == cos[:, 32:]``, so the tables MUST be built duplicated.
    """
    theta = torch.rand(tokens, 32, device=device) * (2 * math.pi)
    cos32, sin32 = theta.cos(), theta.sin()
    return (
        torch.cat([cos32, cos32], dim=-1).bfloat16().contiguous(),
        torch.cat([sin32, sin32], dim=-1).bfloat16().contiguous(),
    )


# ---------------------------------------------------------------------------
# Case construction: build inputs once per case, both dtype paths.
# ---------------------------------------------------------------------------

_CASES = {
    # name: (in_features K, num_heads, tokens)
    #
    # num_heads floor: the kernel is numerically WRONG below 8 heads (probed
    # 2026-08-30: nh in {2, 4, 6} lands at 2-11 dB with ~20% code mismatches
    # on at least one dtype path; nh >= 8 is clean on everything tested), so
    # the smallest case pins nh=8, the smallest verified-correct head count.
    "minimal": (256, 8, 128),  # one TILE_M tile, smallest verified head count
    "671b": (1536, 128, 256),  # DeepSeek-V3 671B wq_b dims (the kernel's design point)
    "16b": (2048, 16, 256),  # DeepSeek-V3 16B wq dims
    "16b_big": (2048, 16, 4096),  # multi-tile scale relay at a realistic token count
}


def _build_case(K, num_heads, tokens, device="cuda", seed=0):
    torch.manual_seed(seed)
    N = num_heads * HEAD_DIM
    c = {"tokens": tokens, "num_heads": num_heads}
    c["x"] = torch.randn(tokens, K, dtype=torch.bfloat16, device=device)
    c["w"] = torch.randn(N, K, dtype=torch.bfloat16, device=device) / (K**0.5)
    c["cos"], c["sin"] = _rope_tables(tokens, device)
    c["x_q"], c["x_sf"] = _quant_rowwise(c["x"])
    c["w_q"], c["w_sf"] = _quant_rowwise(c["w"])
    return c


# ---------------------------------------------------------------------------
# PyTorch reference: dequantization, FP64 matmul, BF16 staging, halves RoPE,
# to_mx requantization -- nothing else.
# ---------------------------------------------------------------------------

# Matched-requant agreement (both sides carry the identical to_mx/RCEIL output
# quantization): measured 67-125 dB across the cases below and three seeds;
# the residual is the kernel's FP32 accumulation-order noise surviving the
# BF16 staging round. Convention/layout bugs (swapped cos/sin,
# interleaved-write RoPE, missing staging, transposed colwise scales) land at
# 2-40 dB.
_MIN_SQNR_MATCHED_REQUANT = 50.0
# Fraction of output code BYTES allowed to differ from the reference
# (single-ulp accumulation-order flips at near-zero sites): measured
# <= 7.5e-5 (bf16 path) / <= 6.4e-6 (mxfp8 path) across cases and seeds.
_MAX_CODE_MISMATCH_FRACTION = 5e-4
# Scale bytes match the to_mx/RCEIL reference almost exactly, but not
# unconditionally: one RCEIL boundary tie flipped under accumulation-order
# noise in one of 24 measured case/seed combos (1 byte in 786k), so equality
# is asserted as a tight budget rather than torch.equal.
_MAX_SCALE_MISMATCH_FRACTION = 1e-5


def _ref_mla_q_proj_rope(x, w, cos, sin, x_scale=None, w_scale=None):
    """PyTorch reference for mxfp8_mla_q_proj_rope_cudnn: dequantize the
    operands (MXFP8 path), FP64 projection GEMM (the container defaults FP32
    matmul to TF32; FP64 is immune), the kernel's BF16 smem staging round,
    interleaved-read/halves-write RoPE on the trailing 64 features in FP32,
    then rowwise and columnwise to_mx/RCEIL requantization."""
    tokens, num_heads = x.shape[0], w.shape[0] // HEAD_DIM
    if x_scale is None:
        x_f64, w_f64 = x.double(), w.double()
    else:
        x_f64, w_f64 = _dequant_input(x, x_scale), _dequant_input(w, w_scale)
    y = (x_f64 @ w_f64.t()).float().bfloat16().float()
    y = y.view(tokens, num_heads, HEAD_DIM)

    c = cos.float().unsqueeze(1)  # [T, 1, 64], halves duplicated
    s = sin.float().unsqueeze(1)
    pe = y[..., QK_NOPE_HEAD_DIM:]
    x1, x2 = pe[..., 0::2], pe[..., 1::2]
    rot = torch.cat(
        [x1 * c[..., :32] - x2 * s[..., :32], x2 * c[..., 32:] + x1 * s[..., 32:]],
        dim=-1,
    )
    y = torch.cat([y[..., :QK_NOPE_HEAD_DIM], rot], dim=-1)
    return (*_quant_out_rowwise(y), *_quant_out_colwise(y))


def _mismatch_fraction(actual: torch.Tensor, ref: torch.Tensor) -> float:
    return (actual.view(torch.uint8) != ref.view(torch.uint8)).float().mean().item()


def _assert_outputs_match(outs, refs, label):
    q_row, row_sf, q_col, col_sf = outs
    r_row, r_row_sf, r_col, r_col_sf = refs
    for name, actual, ref, budget in (
        ("row scales", row_sf, r_row_sf, _MAX_SCALE_MISMATCH_FRACTION),
        ("col scales", col_sf, r_col_sf, _MAX_SCALE_MISMATCH_FRACTION),
        ("row codes", q_row, r_row, _MAX_CODE_MISMATCH_FRACTION),
        ("col codes", q_col, r_col, _MAX_CODE_MISMATCH_FRACTION),
    ):
        frac = _mismatch_fraction(actual, ref)
        assert frac <= budget, (
            f"{label} {name} byte mismatch fraction {frac} exceeds {budget}"
        )
    for name, actual, ref in (
        ("row", _dequant_rowwise(q_row, row_sf), _dequant_rowwise(r_row, r_row_sf)),
        ("col", _dequant_colwise(q_col, col_sf), _dequant_colwise(r_col, r_col_sf)),
    ):
        sqnr = compute_error(ref, actual).item()
        assert sqnr >= _MIN_SQNR_MATCHED_REQUANT, (
            f"{label} {name} sqnr {sqnr} is too low, must be >= "
            f"{_MIN_SQNR_MATCHED_REQUANT}"
        )


@pytest.mark.parametrize("case", list(_CASES))
def test_mxfp8_mla_q_proj_rope_cudnn_bf16_matches_reference(case):
    c = _build_case(*_CASES[case])
    args = (c["x"], c["w"], c["cos"], c["sin"])
    outs = _OPS.mxfp8_mla_q_proj_rope_cudnn(*args)
    refs = _ref_mla_q_proj_rope(*args)
    _assert_outputs_match(outs, refs, f"{case} bf16")


@pytest.mark.parametrize("case", list(_CASES))
def test_mxfp8_mla_q_proj_rope_cudnn_mxfp8_matches_reference(case):
    c = _build_case(*_CASES[case])
    args = (c["x_q"], c["w_q"], c["cos"], c["sin"], c["x_sf"], c["w_sf"])
    outs = _OPS.mxfp8_mla_q_proj_rope_cudnn(*args)
    refs = _ref_mla_q_proj_rope(*args)
    _assert_outputs_match(outs, refs, f"{case} mxfp8")
