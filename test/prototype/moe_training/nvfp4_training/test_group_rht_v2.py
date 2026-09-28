# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""V2 dynamic RHT-128 activation kernels against independent PyTorch math."""

import pytest
import torch

from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_cutedsl import (
    cutedsl_group_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_cutedsl import (
    cutedsl_group_rht_quantize_row_col,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.prototype.mx_formats.utils import from_blocked

from .nvfp4_v2_reference import (
    from_blocked_grouped,
    reference_group_row_cast_col_rht_amax,
    reference_group_row_cast_col_rht_quantize,
)

pytestmark = pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(), reason="requires SM100+ and CuTeDSL"
)


def _case(sizes, hidden, tail):
    torch.manual_seed(225)
    x = torch.randn(sum(sizes) + tail, hidden, device="cuda", dtype=torch.bfloat16)
    x[sum(sizes) :] = torch.nan
    offs = torch.tensor(sizes, device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    signs = torch.randint(0, 2, (128,), device="cuda", dtype=torch.int8) * 2 - 1
    return x, offs, signs


@pytest.mark.parametrize(
    "sizes,hidden,tail",
    [
        ([256], 512, 0),
        ([128, 256, 128], 384, 0),
        ([128, 0, 256], 512, 0),
        ([128, 256], 512, 256),
        ([128] * 64, 128, 0),
        ([128, 128], 7168, 0),
    ],
)
@torch.no_grad()
def test_dynamic_amax_and_quantize_bitwise(sizes, hidden, tail):
    x, offs, signs = _case(sizes, hidden, tail)
    args = (x, [], offs, len(sizes), x.shape[0], hidden, 1)
    kw = dict(sign_tensor=signs, dynamic_rht=True, logical_packed_length=offs[-1:])
    for _ in range(2):
        col_amax, row_amax = cutedsl_group_rht_amax(*args, **kw)
        ref_amax = reference_group_row_cast_col_rht_amax(x, signs, offs, len(sizes))
        for got, want in zip((col_amax, row_amax), ref_amax):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        got = cutedsl_group_rht_quantize_row_col(
            *args, row_amax, col_amax, None, False, use_fast_math=False, **kw
        )
        want = reference_group_row_cast_col_rht_quantize(
            x, row_amax, col_amax, signs, offs, len(sizes)
        )
        valid = sum(sizes)
        g = (
            got[0][:valid],
            from_blocked(got[1], x.shape[0], hidden // 16)[:valid],
            got[2][:, : valid // 2],
            from_blocked_grouped(got[3], hidden, sizes),
        )
        r = (
            want[0][:valid],
            from_blocked(want[1], x.shape[0], hidden // 16)[:valid],
            want[2][:, : valid // 2],
            from_blocked_grouped(want[3], hidden, sizes),
        )
        for actual, expected in zip(g, r):
            assert torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        signs.copy_(
            torch.randint(0, 2, (128,), device="cuda", dtype=torch.int8) * 2 - 1
        )


def test_dynamic_rht_rejects_stochastic_rounding():
    x, offs, signs = _case([128, 128], 256, 0)
    args = (x, [], offs, 2, 256, 256, 1)
    kw = dict(sign_tensor=signs, dynamic_rht=True)
    ac, ar = cutedsl_group_rht_amax(*args, **kw)
    with pytest.raises(
        ValueError, match="stochastic rounding is not supported with dynamic_rht"
    ):
        cutedsl_group_rht_quantize_row_col(
            *args, ar, ac, torch.tensor([1, 2, 3, 4], device="cuda"), True, **kw
        )


def test_dynamic_rht_requires_128_signs():
    x, offs, _ = _case([256], 256, 0)
    with pytest.raises(ValueError, match=r"sign_tensor must be a \(128,\) tensor"):
        cutedsl_group_rht_amax(
            x,
            [],
            offs,
            1,
            256,
            256,
            1,
            sign_tensor=torch.ones(16, device="cuda", dtype=torch.int8),
            dynamic_rht=True,
        )


@pytest.mark.parametrize(
    "sizes,hidden", [([256], 512), ([128, 0, 256], 384), ([128, 256], 7168)]
)
@torch.no_grad()
def test_dynamic_fast_math_reconstruction_sqnr(sizes, hidden):
    """Use the V1 fast-math test's 25 dB fast-vs-exact reconstruction floor.

    Skipping the bf16 accumulator rounding moves values across occasional FP4
    midpoints (about 30-32 dB here), even though its input perturbation is small.
    This is a numerical bound; exact-mode byte checks remain separate.
    """
    from torchao.quantization.utils import compute_error

    from .nvfp4_v2_reference import reference_dequantize_rowwise

    x, offs, signs = _case(sizes, hidden, 0)
    args = (x, [], offs, len(sizes), x.shape[0], hidden, 1)
    kw = dict(sign_tensor=signs, dynamic_rht=True, logical_packed_length=offs[-1:])
    col_amax, row_amax = reference_group_row_cast_col_rht_amax(
        x, signs, offs, len(sizes)
    )
    got = cutedsl_group_rht_quantize_row_col(
        *args, row_amax, col_amax, None, False, use_fast_math=True, **kw
    )
    ref = reference_group_row_cast_col_rht_quantize(
        x, row_amax, col_amax, signs, offs, len(sizes)
    )
    plain = [
        (
            from_blocked(out[1], x.shape[0], hidden // 16),
            from_blocked_grouped(out[3], hidden, sizes),
        )
        for out in (got, ref)
    ]
    start = 0
    for group, size in enumerate(sizes):
        if size == 0:
            continue
        end = start + size
        for axis, amax in ((0, row_amax[group]), (1, col_amax[group])):
            decoded = []
            for out, sf in zip((got, ref), plain):
                codes = (
                    out[0][start:end] if axis == 0 else out[2][:, start // 2 : end // 2]
                )
                scales = (
                    sf[0][start:end] if axis == 0 else sf[1][:, start // 16 : end // 16]
                )
                decoded.append(
                    reference_dequantize_rowwise(codes, scales, amax, is_swizzled=False)
                )
            assert torch.isfinite(decoded[0]).all()
            sqnr = compute_error(decoded[1], decoded[0])
            assert sqnr >= 25.0, (
                f"expert={group}, axis={axis}: fast-vs-PyTorch SQNR {sqnr:.2f} dB < 25 dB"
            )
        start = end


@pytest.mark.parametrize(
    "kind", ["zero_negative_signs", "nonfinite_blocks", "padded_rows"]
)
@torch.no_grad()
def test_dynamic_quantize_edge_cases_bitwise(kind):
    sizes, hidden = [256, 256], 512
    x, offs, signs = _case(sizes, hidden, 0)
    if kind == "zero_negative_signs":
        x[:256] = 0
        signs.fill_(-1)
    elif kind == "nonfinite_blocks":
        x[0, 0] = torch.nan
        x[1, -1] = torch.nan
        x[2, 16:32] = torch.nan
        x[3, 40] = torch.inf
    else:
        x[127:256] = 0
        x[-17:] = 0
    finite = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    ac, ar = reference_group_row_cast_col_rht_amax(finite, signs, offs, 2)
    got = cutedsl_group_rht_quantize_row_col(
        x,
        [],
        offs,
        2,
        512,
        hidden,
        0,
        ar,
        ac,
        None,
        False,
        sign_tensor=signs,
        dynamic_rht=True,
        use_fast_math=False,
    )
    expected = reference_group_row_cast_col_rht_quantize(x, ar, ac, signs, offs, 2)
    for actual, reference in zip(got, expected):
        assert torch.equal(
            actual.contiguous().view(torch.uint8),
            reference.contiguous().view(torch.uint8),
        )
