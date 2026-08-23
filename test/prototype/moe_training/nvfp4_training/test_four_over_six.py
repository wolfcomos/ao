# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.nvfp4_training.four_over_six import (
    NVFP4FourOverSixLinear,
    four_over_six_global_encode_scale,
    four_over_six_linear,
    four_over_six_quantize,
)
from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32, unpack_uint4
from torchao.utils import is_sm_at_least_100, torch_version_at_least

_skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)
_skip_no_sm100 = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and is_sm_at_least_100()
        and torch_version_at_least("2.10.0")
    ),
    reason="requires SM100+ and PyTorch 2.10+ (FP4 scaled_mm)",
)


def _dequantize(codes, scales, global_amax, e4m3_scale_bound):
    """Reconstruct FP32 values from packed codes, block scales, and global amax."""
    rows = codes.shape[0]
    values = f4_unpacked_to_f32(unpack_uint4(codes)).view(rows, -1, 16)
    s_dec = 1.0 / four_over_six_global_encode_scale(global_amax, e4m3_scale_bound)
    if s_dec.dim() == 1:
        s_dec = s_dec.view(rows, 1, 1)
    return (values * scales.to(torch.float32).unsqueeze(-1) * s_dec).view(rows, -1)


def _map6_reference(x, global_amax, e4m3_scale_bound):
    """Standard (map-to-6 only) encoding with the four-over-six scale chain."""
    from torchao.prototype.moe_training.nvfp4_training.four_over_six import (
        _FP32_MAX,
        FP4_E2M1_MAX,
        FP8_E4M3_MAX,
        _fp4_rtne,
    )

    rows, cols = x.shape
    xf = x.float().view(rows, cols // 16, 16)
    s_enc = four_over_six_global_encode_scale(global_amax, e4m3_scale_bound)
    fp4_max = torch.full((), FP4_E2M1_MAX, dtype=torch.float32, device=x.device)
    base = (xf.abs().amax(dim=-1) / fp4_max) * s_enc
    scale6 = base.clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    inv6 = (1.0 / (scale6.to(torch.float32) * (1.0 / s_enc))).clamp(max=_FP32_MAX)
    _, values6 = _fp4_rtne(xf * inv6.unsqueeze(-1))
    s_dec = (1.0 / s_enc).view(-1, 1, 1) if s_enc.dim() == 1 else 1.0 / s_enc
    dequant6 = values6 * scale6.to(torch.float32).unsqueeze(-1) * s_dec
    return dequant6.view(rows, cols)


@_skip_no_cuda
@pytest.mark.parametrize("err_mode", ["mae", "mse"])
@pytest.mark.parametrize("e4m3_scale_bound", [256, 448])
@pytest.mark.parametrize("block", ["1x16", "16x16"])
def test_scales_are_candidate_scales(err_mode, e4m3_scale_bound, block):
    """Every stored block scale is one of the two candidate scales."""
    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    amax = x.abs().amax().to(torch.float32)
    _, scales = four_over_six_quantize(
        x, amax, block=block, err_mode=err_mode, e4m3_scale_bound=e4m3_scale_bound
    )

    xf = x.float().view(128, 16, 16)
    if block == "16x16":
        tiles = x.float().abs().view(8, 16, 16, 16)
        block_amax = tiles.amax(dim=(1, 3)).repeat_interleave(16, dim=0)
    else:
        block_amax = xf.abs().amax(dim=-1)
    s_enc = four_over_six_global_encode_scale(amax, e4m3_scale_bound)
    fp4_max = torch.full((), 6.0, dtype=torch.float32, device="cuda")
    base = (block_amax / fp4_max) * s_enc
    scale6 = base.clamp(max=448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    scale4 = (base * 1.5).clamp(max=448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    got = scales.view(torch.uint8)
    assert ((got == scale6) | (got == scale4)).all()


@_skip_no_cuda
@pytest.mark.parametrize("e4m3_scale_bound", [256, 448])
def test_selection_not_worse_than_map6(e4m3_scale_bound):
    """Per-block MAE of the stored encoding <= the map-to-6-only encoding."""
    torch.manual_seed(0)
    x = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
    amax = x.abs().amax().to(torch.float32)
    codes, scales = four_over_six_quantize(
        x, amax, block="1x16", err_mode="mae", e4m3_scale_bound=e4m3_scale_bound
    )
    dq = _dequantize(codes, scales, amax, e4m3_scale_bound)
    dq6 = _map6_reference(x, amax, e4m3_scale_bound)
    xf = x.float()
    err = (dq - xf).abs().view(256, -1, 16).sum(dim=-1).double()
    err6 = (dq6 - xf).abs().view(256, -1, 16).sum(dim=-1).double()
    # Selection minimizes the FP32 sequential-sum error; allow FP32-vs-FP64
    # summation slack on ties.
    assert (err <= err6 + 1e-4).all()
    # And the recipe must actually engage: some blocks pick map-to-4.
    assert (err < err6 - 1e-4).any()


@_skip_no_cuda
def test_row_scaled_matches_per_row_quantization():
    """Row-scaled output == each row quantized alone with its own scalar amax."""
    torch.manual_seed(0)
    x = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
    row_amax = x.abs().amax(dim=1).to(torch.float32)
    codes, scales = four_over_six_quantize(x, row_amax, block="1x16")
    for r in range(0, 64, 17):
        codes_r, scales_r = four_over_six_quantize(x[r : r + 1], row_amax[r].view(()))
        torch.testing.assert_close(codes[r : r + 1], codes_r, atol=0, rtol=0)
        torch.testing.assert_close(
            scales[r : r + 1].view(torch.uint8),
            scales_r.view(torch.uint8),
            atol=0,
            rtol=0,
        )


@_skip_no_cuda
def test_row_scaled_rejects_16x16():
    x = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
    row_amax = x.abs().amax(dim=1).to(torch.float32)
    with pytest.raises(ValueError, match="1x16 blocks only"):
        four_over_six_quantize(x, row_amax, block="16x16")


@_skip_no_cuda
@pytest.mark.parametrize("block", ["1x16", "16x16"])
def test_dequant_sqnr(block):
    torch.manual_seed(0)
    x = torch.randn(128, 512, dtype=torch.bfloat16, device="cuda")
    amax = x.abs().amax().to(torch.float32)
    codes, scales = four_over_six_quantize(x, amax, block=block)
    dq = _dequantize(codes, scales, amax, 256)
    assert compute_error(x.float(), dq).item() > 14.0


@_skip_no_sm100
@pytest.mark.parametrize("row_scaled_activation", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_linear_forward_backward(row_scaled_activation, bias):
    torch.manual_seed(0)
    M, K, N = 256, 512, 384
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    w = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1).requires_grad_(
        True
    )
    b = (
        torch.randn(N, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        if bias
        else None
    )
    y = four_over_six_linear(x, w, b, "mae", 256, row_scaled_activation)
    assert y.shape == (M, N)
    dy = torch.randn_like(y)
    y.backward(dy)

    y_ref = x.detach().float() @ w.detach().float().t()
    if bias:
        y_ref = y_ref + b.detach().float()
    dx_ref = dy.float() @ w.detach().float()
    dw_ref = dy.float().t() @ x.detach().float()
    assert compute_error(y_ref, y.float()).item() > 14.0
    assert compute_error(dx_ref, x.grad.float()).item() > 14.0
    assert compute_error(dw_ref, w.grad.float()).item() > 14.0
    if bias:
        # grad_bias is reduced in bf16, matching nvfp4_linear.
        torch.testing.assert_close(b.grad, dy.sum(dim=0))


@_skip_no_sm100
def test_linear_module():
    torch.manual_seed(0)
    lin = NVFP4FourOverSixLinear(512, 384, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(128, 512, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    y = lin(x)
    y.sum().backward()
    assert y.shape == (128, 384)
    assert lin.weight.grad is not None


@_skip_no_sm100
def test_linear_rejects_unaligned_dims():
    x = torch.randn(100, 512, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(384, 512, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="divisible by 128"):
        four_over_six_linear(x, w, None, "mae", 256, False)


@_skip_no_cuda
@pytest.mark.parametrize("err_mode", ["mae", "mse"])
@pytest.mark.parametrize("e4m3_scale_bound", [256, 448])
@pytest.mark.parametrize("block", ["1x16", "16x16"])
@pytest.mark.parametrize("row_scaled", [False, True])
def test_bitwise_parity_with_transformer_engine(
    err_mode, e4m3_scale_bound, block, row_scaled
):
    """Bitwise codes/scales vs TransformerEngine's 4over6 kernels, if available."""
    te = pytest.importorskip("transformer_engine.pytorch")
    if row_scaled and block == "16x16":
        pytest.skip("row-scaled is 1x16 only")
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    if not te.is_nvfp4_available():
        pytest.skip("NVFP4 not available in this TransformerEngine build")

    torch.manual_seed(0)
    M, N = 256, 512
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=(block == "16x16"),
        row_scaled_nvfp4=row_scaled,
        nvfp4_use_4over6=True,
        nvfp4_e4m3_max=e4m3_scale_bound,
        nvfp4_4over6_err_mode=err_mode.upper(),
    )
    t = quantizer(x)
    if row_scaled:
        amax = x.abs().amax(dim=1).to(torch.float32)
    else:
        amax = x.abs().amax().to(torch.float32)
    codes, scales = four_over_six_quantize(
        x, amax, block=block, err_mode=err_mode, e4m3_scale_bound=e4m3_scale_bound
    )
    te_codes = t._rowwise_data.view(torch.uint8)[:, : N // 2]
    te_scales = t._rowwise_scale_inv[:M, : N // 16].view(torch.uint8)
    torch.testing.assert_close(te_codes, codes, atol=0, rtol=0)
    torch.testing.assert_close(te_scales, scales.view(torch.uint8), atol=0, rtol=0)
