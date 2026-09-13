# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped rotated columnwise weight requantization. Design doc §11.4 and §11.5.

§11.6/§11.7 with an RHT-128 rotation, for V2's dgrad. The rotation is what lets the
dgrad GEMM cancel: the ``dy`` operand carries ``R_n`` too. One ``dgrad_rht`` is shared
across every expert, which is the grouped-plus-RHT case the implementation must
support.

Both ops select their backend through the ``kernel`` parametrization (``_KERNELS``);
the requantize tests take the same backend's amax, which is bitwise across backends.
"""

import pytest
import torch

from torchao.prototype.moe_training.nvfp4_training.group_col_rht_requantize_cutedsl import (
    cutedsl_group_col_rht_requant_amax,
    cutedsl_group_col_rht_requantize,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)

from ._assertions import assert_codes_bitwise, assert_scales_bitwise
from ._v2_marks import TRITON_AVAILABLE, kernel_gate, maybe_sm100
from .nvfp4_reference import (
    reference_col_rht_requant_amax,
    reference_group_col_rht_requant_amax,
    reference_group_col_rht_requantize,
)

# Flip to True once both @triton.jit bodies in group_col_rht_requantize_triton.py land.
_KERNEL_IMPLEMENTED = True
_needs_kernel = kernel_gate(_KERNEL_IMPLEMENTED, "group_col_rht_requantize_triton.py")
_skip_no_cutedsl = pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(),
    reason="requires SM100 (Blackwell) + CuteDSL runtime (cuda-python, nvidia-cutlass-dsl)",
)
_KERNELS = [
    pytest.param("triton", id="triton"),
    pytest.param("cutedsl", marks=_skip_no_cutedsl, id="cutedsl"),
]

if TRITON_AVAILABLE:
    from torchao.prototype.moe_training.nvfp4_training.group_col_rht_requantize_triton import (
        triton_group_col_rht_requant_amax,
        triton_group_col_rht_requantize,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_row_cast_quantize_triton import (
        triton_group_row_cast_quantize,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_weight_amax_triton import (
        triton_group_weight_amax,
    )


def _signs(device="cuda", seed=0):
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (128,), generator=generator, dtype=torch.int8)
    return (bits * 2 - 1).to(device)


def _packed_weights(E, M, N, *, seed=0, scale=0.05):
    torch.manual_seed(seed)
    W = (torch.randn(E, M, N, device="cuda") * scale).bfloat16()
    amax = triton_group_weight_amax(W, E)
    codes, scales = triton_group_row_cast_quantize(W, amax, E)
    return W, codes, scales, amax


def _amax(kernel, codes, scales, amax, d, E):
    op = (
        triton_group_col_rht_requant_amax
        if kernel == "triton"
        else cutedsl_group_col_rht_requant_amax
    )
    return op(codes, scales, amax, d, E)


def _requant(kernel, codes, scales, amax, amax_t, d, E):
    op = (
        triton_group_col_rht_requantize
        if kernel == "triton"
        else cutedsl_group_col_rht_requantize
    )
    return op(codes, scales, amax, amax_t, d, E)


# --- §11.4 ------------------------------------------------------------------


@_needs_kernel
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_amax_single_expert_matches_reference(kernel):
    _, codes, scales, amax = _packed_weights(1, 256, 512)
    d = _signs()
    got = _amax(kernel, codes, scales, amax, d, 1)
    want = reference_col_rht_requant_amax(codes[0], scales[0], amax[0], d)
    torch.testing.assert_close(got[0], want, rtol=1e-3, atol=1e-3)


@_needs_kernel
@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("E", [2, 4])
@torch.no_grad()
def test_amax_multi_expert_matches_reference(kernel, E):
    _, codes, scales, amax = _packed_weights(E, 256, 512)
    d = _signs()
    got = _amax(kernel, codes, scales, amax, d, E)
    want = reference_group_col_rht_requant_amax(codes, scales, amax, d)
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)


@_needs_kernel
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_amax_changes_with_the_sign_vector(kernel):
    """A different rotation gives a different bound; the signs really reach the kernel."""
    _, codes, scales, amax = _packed_weights(1, 256, 512)
    a = _amax(kernel, codes, scales, amax, _signs(seed=0), 1)
    b = _amax(kernel, codes, scales, amax, _signs(seed=1), 1)
    assert a[0].item() != b[0].item()


@_needs_kernel
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_a_strided_sign_view_matches_the_contiguous_call(kernel):
    """A stride-2 int8 view of the sign vector gives the contiguous call's outputs."""
    _, codes, scales, amax = _packed_weights(1, 256, 512)
    d = _signs()
    strided = torch.stack([d, d], 1)[:, 0]
    amax_t = _amax(kernel, codes, scales, amax, d, 1)
    assert torch.equal(_amax(kernel, codes, scales, amax, strided, 1), amax_t)
    want = _requant(kernel, codes, scales, amax, amax_t, d, 1)
    got = _requant(kernel, codes, scales, amax, amax_t, strided, 1)
    assert_codes_bitwise(got[0], want[0], "codes")
    assert torch.equal(got[1].view(torch.uint8), want[1].view(torch.uint8))


@_needs_kernel
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_amax_is_computed_from_the_quantized_weight(kernel):
    """Discriminating test that the lazy path consumes ``row_fp4_w``, not the BF16 W.

    Computing the same quantity from the original weight gives a different answer
    once the rotation is involved, because ``W`` and ``W_qdq`` differ per element even
    though their maxima coincide (see the note in the unrotated twin's test file).
    """
    from .nvfp4_reference import reference_dynamic_rht

    W, codes, scales, amax = _packed_weights(1, 256, 512)
    d = _signs()
    got = _amax(kernel, codes, scales, amax, d, 1)
    from_bf16 = reference_dynamic_rht(W[0], d, transpose=True).float().abs().max()
    assert abs(got[0].item() - from_bf16.item()) > 1e-6, (
        "an amax taken over the bf16 weight would not match; the op must measure W_qdq"
    )


@_needs_kernel
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_amax_per_expert_isolation_and_shared_sign_vector(kernel):
    """One ``B`` operand serves every expert, and no expert's amax leaks into another."""
    E = 4
    _, codes, scales, amax = _packed_weights(E, 256, 512)
    d = _signs()
    baseline = _amax(kernel, codes, scales, amax, d, E)
    hot = amax.clone()
    hot[2] *= 1000.0
    got = _amax(kernel, codes, scales, hot, d, E)
    for e in range(E):
        if e == 2:
            continue
        assert got[e].item() == baseline[e].item(), f"expert {e} leaked"


# --- §11.5 ------------------------------------------------------------------


@_needs_kernel
@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("E", [1, 2, 4])
@torch.no_grad()
def test_requantize_matches_reference(kernel, E):
    _, codes, scales, amax = _packed_weights(E, 256, 512)
    d = _signs()
    amax_t = _amax(kernel, codes, scales, amax, d, E)
    got_codes, got_scales = _requant(kernel, codes, scales, amax, amax_t, d, E)
    refs = reference_group_col_rht_requantize(codes, scales, amax, amax_t, d)
    for e in range(E):
        assert_codes_bitwise(got_codes[e], refs[e].codes, f"codes[{e}]")
        assert_scales_bitwise(got_scales[e], refs[e].scales, f"scales[{e}]")


@_needs_kernel
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_a_halved_amax_saturates_only_that_expert(kernel):
    E = 4
    _, codes, scales, amax = _packed_weights(E, 256, 512)
    d = _signs()
    amax_t = _amax(kernel, codes, scales, amax, d, E)
    baseline = _requant(kernel, codes, scales, amax, amax_t, d, E)
    halved = amax_t.clone()
    halved[1] *= 0.5
    got = _requant(kernel, codes, scales, amax, halved, d, E)
    assert not torch.equal(got[0][1], baseline[0][1])
    for e in (0, 2, 3):
        assert_codes_bitwise(got[0][e], baseline[0][e], f"codes[{e}] leaked")


@_needs_kernel
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_decode_numerator_is_2688_not_1536(kernel):
    """The weight is a *cast* operand even in V2; only MS-EDEN operands use 1536.

    Checked through the reference, which is parameterized on the FP8 ceiling: decoding
    with the MS-EDEN numerator must disagree, so the test is sensitive to the mistake
    the design doc calls out as "backward off by roughly 40%".
    """
    from .nvfp4_reference import EDEN_BLOCK_SCALE_MAX, reference_dequantize_rowwise

    _, codes, scales, amax = _packed_weights(1, 256, 512)
    d = _signs()
    amax_t = _amax(kernel, codes, scales, amax, d, 1)
    col_codes, col_scales = _requant(kernel, codes, scales, amax, amax_t, d, 1)
    right = reference_dequantize_rowwise(col_codes[0], col_scales[0], amax_t[0])
    wrong = reference_dequantize_rowwise(
        col_codes[0], col_scales[0], amax_t[0], fp8_max=EDEN_BLOCK_SCALE_MAX
    )
    assert not torch.allclose(right, wrong, rtol=1e-3, atol=1e-6)


# --- wrapper layer, runs today ----------------------------------------------


@maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_register_fake_shapes(kernel):
    from torch._subclasses.fake_tensor import FakeTensorMode

    E, M, N = 2, 256, 512
    with FakeTensorMode():
        codes = torch.empty(E, M, N // 2, dtype=torch.uint8, device="cuda")
        scales = torch.empty(
            E, M // 128, N // 64, 32, 16, dtype=torch.float8_e4m3fn, device="cuda"
        )
        amax = torch.empty(E, dtype=torch.float32, device="cuda")
        d = torch.empty(128, dtype=torch.int8, device="cuda")
        got = _amax(kernel, codes, scales, amax, d, E)
        assert got.shape == (E,) and got.dtype == torch.float32
        col_codes, col_scales = _requant(kernel, codes, scales, amax, got, d, E)
    assert col_codes.shape == (E, N, M // 2)
    assert col_scales.shape == (E, N // 128, M // 64, 32, 16)


@maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("bad_len", [16, 64, 256])
@torch.no_grad()
def test_rejects_a_non_128_sign_vector(kernel, bad_len):
    """V2 is RHT-128 only. A 16-element vector is V1's and must not be accepted here."""
    E, M, N = 2, 256, 512
    codes = torch.zeros(E, M, N // 2, dtype=torch.uint8, device="cuda")
    scales = torch.zeros(
        E, M // 128, N // 64, 32, 16, dtype=torch.float8_e4m3fn, device="cuda"
    )
    amax = torch.ones(E, dtype=torch.float32, device="cuda")
    bad = torch.ones(bad_len, dtype=torch.int8, device="cuda")
    with pytest.raises(ValueError, match=r"dgrad_rht must be a \(128,\) tensor"):
        _amax(kernel, codes, scales, amax, bad, E)


# --- CuteDSL vs Triton, bitwise ---------------------------------------------


_BITWISE_CASES = [
    pytest.param(1, 128, 128, id="single-tile"),
    pytest.param(2, 256, 512, id="file-default"),
    pytest.param(4, 384, 1408, id="three-m-tiles"),
    pytest.param(8, 1408, 2048, id="eight-experts"),
    pytest.param(4, 1408, 2048, id="deepseek-16B-gate-up"),
    pytest.param(4, 2048, 1408, id="deepseek-16B-down"),
    pytest.param(512, 128, 128, id="many-experts-multi-expert-cta"),
]


@_needs_kernel
@_skip_no_cutedsl
@pytest.mark.parametrize("E,M,N", _BITWISE_CASES)
@torch.no_grad()
def test_cutedsl_amax_matches_triton(E, M, N):
    """Bitwise by construction: both backends issue the same tcgen05 (128,128,16) bf16
    UMMAs over K = 128 in the same order on the same product values (the sign vector
    moves from ``R_n`` onto the dequantized operand, an exact flip), then round once to
    bf16. A mismatch is a bug or a Triton lowering change, never tolerance."""
    _, codes, scales, amax = _packed_weights(E, M, N, seed=225)
    d = _signs(seed=0)
    t_amax = _amax("triton", codes, scales, amax, d, E)
    c_amax = _amax("cutedsl", codes, scales, amax, d, E)
    assert torch.equal(c_amax, t_amax)


@_needs_kernel
@_skip_no_cutedsl
@pytest.mark.parametrize("E,M,N", _BITWISE_CASES)
@torch.no_grad()
def test_cutedsl_requantize_matches_triton(E, M, N):
    """Same chain, then the RTNE 1x16 quantize instruction for instruction: codes and
    scale bytes are bitwise, each backend fed by its own (bitwise) amax."""
    _, codes, scales, amax = _packed_weights(E, M, N, seed=225)
    d = _signs(seed=0)
    t_amax = _amax("triton", codes, scales, amax, d, E)
    c_amax = _amax("cutedsl", codes, scales, amax, d, E)
    assert torch.equal(c_amax, t_amax)
    t_codes, t_scales = _requant("triton", codes, scales, amax, t_amax, d, E)
    c_codes, c_scales = _requant("cutedsl", codes, scales, amax, c_amax, d, E)
    assert_codes_bitwise(c_codes, t_codes, "codes")
    assert torch.equal(c_scales.view(torch.uint8), t_scales.view(torch.uint8))


@_needs_kernel
@_skip_no_cutedsl
@pytest.mark.parametrize("signs", ["random", "all-minus-one"])
@torch.no_grad()
def test_cutedsl_degenerate_experts_match_triton(signs):
    """An all-zero expert and NaN / inf ``global_amax`` experts reconstruct to zero rows
    in both backends; with every sign negative each zero product is ``-0`` and only an
    accumulator that starts from ``+0``, as Triton's does, keeps the code nibble 0x0."""
    E = 4
    torch.manual_seed(0)
    W = (torch.randn(E, 256, 512, device="cuda") * 0.05).bfloat16()
    W[3] = 0
    amax = triton_group_weight_amax(W, E)
    codes, scales = triton_group_row_cast_quantize(W, amax, E)
    amax[1] = float("nan")
    amax[2] = float("inf")
    d = (
        _signs(seed=0)
        if signs == "random"
        else -torch.ones(128, dtype=torch.int8, device="cuda")
    )
    t_amax = _amax("triton", codes, scales, amax, d, E)
    c_amax = _amax("cutedsl", codes, scales, amax, d, E)
    assert torch.equal(c_amax, t_amax)
    t_codes, t_scales = _requant("triton", codes, scales, amax, t_amax, d, E)
    c_codes, c_scales = _requant("cutedsl", codes, scales, amax, c_amax, d, E)
    assert_codes_bitwise(c_codes, t_codes, "codes")
    assert torch.equal(c_scales.view(torch.uint8), t_scales.view(torch.uint8))
