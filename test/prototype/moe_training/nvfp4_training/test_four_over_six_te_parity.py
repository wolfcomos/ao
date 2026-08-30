# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Local-only TransformerEngine parity harness, kept out of upstream PRs.

These tests pin four_over_six_quantize/four_over_six_dequantize bitwise
against TransformerEngine's own 4over6 NVFP4 kernels, so they require
transformer_engine >= 2.17 (the first release whose ``NVFP4Quantizer``
exposes ``nvfp4_use_4over6``); older builds and TE-less environments skip.
The upstream test suite covers the same arithmetic TE-lessly via the
pure-PyTorch oracle in ``test_four_over_six.py``.
"""

import inspect

import pytest
import torch

from torchao.prototype.moe_training.nvfp4_training.four_over_six import (
    four_over_six_dequantize,
    four_over_six_quantize,
)
from torchao.prototype.moe_training.nvfp4_training.four_over_six_cutedsl import (
    _cutedsl_quantize_available,
)

_skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)
_cutedsl_available = torch.cuda.is_available() and _cutedsl_quantize_available()


def _import_te_quantizer_or_skip():
    """Import TE's NVFP4Quantizer, skipping unless it has the 4over6 knobs.

    TE < 2.17 has an NVFP4Quantizer without ``nvfp4_use_4over6`` (and the
    sibling knobs these tests pass), which would TypeError instead of skip.
    """
    te = pytest.importorskip("transformer_engine.pytorch")
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    if "nvfp4_use_4over6" not in inspect.signature(NVFP4Quantizer).parameters:
        pytest.skip(
            "requires transformer_engine >= 2.17 "
            "(NVFP4Quantizer has no nvfp4_use_4over6)"
        )
    if not te.is_nvfp4_available():
        pytest.skip("NVFP4 not available in this TransformerEngine build")
    return NVFP4Quantizer


@_skip_no_cuda
@pytest.mark.parametrize("err_mode", ["mae", "mse"])
@pytest.mark.parametrize("e4m3_scale_bound", [256, 448])
@pytest.mark.parametrize("block", ["1x16", "16x16"])
@pytest.mark.parametrize("row_scaled", [False, True])
def test_bitwise_parity_with_transformer_engine(
    err_mode, e4m3_scale_bound, block, row_scaled
):
    """Bitwise codes/scales vs TransformerEngine's 4over6 kernels, if available."""
    if row_scaled and block == "16x16":
        pytest.skip("row-scaled is 1x16 only")
    NVFP4Quantizer = _import_te_quantizer_or_skip()

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
    # Also run the CuTe DSL kernel explicitly against TE (the call above
    # already dispatches to it when eligible; this pins the op itself).
    if _cutedsl_available:
        dsl_codes, dsl_scales = torch.ops.torchao.four_over_six_quantize_cutedsl(
            x, amax, block, err_mode, e4m3_scale_bound
        )
        torch.testing.assert_close(te_codes, dsl_codes, atol=0, rtol=0)
        torch.testing.assert_close(
            te_scales, dsl_scales.view(torch.uint8), atol=0, rtol=0
        )


@_skip_no_cuda
@pytest.mark.parametrize("e4m3_scale_bound", [256, 448])
@pytest.mark.parametrize("row_scaled", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_dequantize_bitwise_parity_with_transformer_engine(
    e4m3_scale_bound, row_scaled, out_dtype
):
    """four_over_six_dequantize matches TE's NVFP4 dequantize kernel bitwise."""
    NVFP4Quantizer = _import_te_quantizer_or_skip()

    torch.manual_seed(0)
    M, N = 256, 512
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        row_scaled_nvfp4=row_scaled,
        nvfp4_use_4over6=True,
        nvfp4_e4m3_max=e4m3_scale_bound,
    )
    t = quantizer(x)
    te_dq = t.dequantize(dtype=out_dtype)
    amax = (x.abs().amax(dim=1) if row_scaled else x.abs().amax()).to(torch.float32)
    codes, scales = four_over_six_quantize(x, amax, e4m3_scale_bound=e4m3_scale_bound)
    dq = four_over_six_dequantize(
        codes,
        scales,
        amax,
        e4m3_scale_bound=e4m3_scale_bound,
        out_dtype=out_dtype,
    )
    torch.testing.assert_close(dq, te_dq, atol=0, rtol=0)
