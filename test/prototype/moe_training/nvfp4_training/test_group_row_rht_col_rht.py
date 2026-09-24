# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped RHT-128 on both axes: amax and MS-EDEN quantize. Design doc §11.2, §11.3.

V2's backward gradient path. What distinguishes it from §11.8/§11.9 is that **both**
axes are transformed with **independent** sign vectors. A crossed pair produces no
error, only a wrong gradient, so the tests that separate the two vectors are the
important ones here.

Both ops select their backend through the ``kernel`` parametrization (``_KERNELS``);
the MS-EDEN tests take the same backend's amax, which is bitwise across backends.
"""

import math

import pytest
import torch

from torchao.prototype.moe_training.nvfp4_training.group_row_rht_col_rht_amax_cutedsl import (
    cutedsl_group_row_rht_col_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.group_row_rht_col_rht_quantize_ms_eden_cutedsl import (
    cutedsl_group_row_rht_col_rht_quantize_ms_eden,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)

from ._assertions import assert_codes_bitwise, assert_scales_adjacent
from ._v2_marks import TRITON_AVAILABLE, kernel_gate, maybe_sm100
from .nvfp4_reference import (
    decode_fp4_codes,
    reference_group_row_rht_col_rht_amax,
    reference_row_rht_col_rht_amax,
)

_AMAX_IMPLEMENTED = True
_MS_EDEN_IMPLEMENTED = True
_needs_amax = kernel_gate(_AMAX_IMPLEMENTED, "group_row_rht_col_rht_amax_triton.py")
_needs_ms_eden = kernel_gate(
    _AMAX_IMPLEMENTED and _MS_EDEN_IMPLEMENTED,
    "group_row_rht_col_rht_quantize_ms_eden_triton.py",
)
_skip_no_cutedsl = pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(),
    reason="requires SM100 (Blackwell) + CuteDSL runtime (cuda-python, nvidia-cutlass-dsl)",
)
_KERNELS = [
    pytest.param("triton", id="triton"),
    pytest.param("cutedsl", marks=_skip_no_cutedsl, id="cutedsl"),
]

if TRITON_AVAILABLE:
    from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
        VARYING_FIRST_DIM,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_row_rht_col_rht_amax_triton import (
        triton_group_row_rht_col_rht_amax,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_row_rht_col_rht_quantize_ms_eden_triton import (
        triton_group_row_rht_col_rht_quantize_ms_eden,
    )


def _signs(device="cuda", seed=0):
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (128,), generator=generator, dtype=torch.int8)
    return (bits * 2 - 1).to(device)


def _packed(group_sizes, hidden, *, seed=0):
    torch.manual_seed(seed)
    dy = torch.randn(sum(group_sizes), hidden, device="cuda", dtype=torch.bfloat16)
    offs = torch.cumsum(
        torch.tensor(group_sizes, dtype=torch.int32, device="cuda"),
        0,
        dtype=torch.int32,
    )
    return dy, offs


def _amax(kernel, dy, d, w, offs, E):
    op = (
        triton_group_row_rht_col_rht_amax
        if kernel == "triton"
        else cutedsl_group_row_rht_col_rht_amax
    )
    return op(dy, d, w, offs, E, dy.shape[0], dy.shape[1], VARYING_FIRST_DIM, offs[-1:])


def _ms_eden(kernel, dy, ar, ac, d, w, offs, E, rng):
    op = (
        triton_group_row_rht_col_rht_quantize_ms_eden
        if kernel == "triton"
        else cutedsl_group_row_rht_col_rht_quantize_ms_eden
    )
    return op(
        dy,
        ar,
        ac,
        d,
        w,
        offs,
        E,
        dy.shape[0],
        dy.shape[1],
        VARYING_FIRST_DIM,
        rng,
        offs[-1:],
    )


def _ms_eden_fast(dy, ar, ac, d, w, offs, E, rng):
    """The CuteDSL op's ``FAST_PATH`` (hardware stochastic rounding, one Philox draw per
    16 scales): a different random stream from the Triton op's, so never compared bitwise
    to it -- only its codes are."""
    return cutedsl_group_row_rht_col_rht_quantize_ms_eden(
        dy,
        ar,
        ac,
        d,
        w,
        offs,
        E,
        dy.shape[0],
        dy.shape[1],
        VARYING_FIRST_DIM,
        rng,
        offs[-1:],
        fast_path=True,
    )


def _e4m3_neighbours(corrected):
    """The two E4M3 grid values a stochastic rounding of ``corrected`` chooses between and
    the 20-bit fractional position between them -- the software path's own 2^-120 grid
    trick (E4M3 subnormals included), so the neighbours are exactly what both roundings
    pick from."""
    bits = (corrected.float() * 2.0**-120).view(torch.int32)
    frac20 = bits & 0xFFFFF
    lo = (bits - frac20).view(torch.float32) * 2.0**120
    hi = (bits - frac20 + (1 << 20)).view(torch.float32) * 2.0**120
    return lo, hi, frac20.float() / 2.0**20


def _fast_cross_dot_overflows(rht, codes):
    """The 1x16 blocks whose ``FAST_PATH`` cross dot is not finite: ``_dot16_e_q_bf16``
    forms ``<e, q>`` on the unscaled bf16 values in f32 down two 8-deep chains (even and odd
    elements) that the kernel adds ahead of the encode multiply, so it overflows once the
    block's sum of |e| x |q| reaches 2^128 (from |e| ~ 2^128 / 96, about 3.5e36, for sixteen
    saturated codes) where Triton's ``<v, q>`` of the scaled values stays finite. Every
    product is exact (bf16 x E2M1), so the same f32 additions reproduce the chains bit for
    bit."""
    rows, cols = rht.shape
    e = rht.float().reshape(rows, cols // 16, 8, 2)
    q = decode_fp4_codes(codes).reshape(rows, cols // 16, 8, 2)
    acc = torch.zeros_like(e[:, :, 0])
    for k in range(8):
        acc = acc + e[:, :, k] * q[:, :, k]
    return ~torch.isfinite(acc[..., 0] + acc[..., 1])


# --- §11.2 ------------------------------------------------------------------


@_needs_amax
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_amax_single_group_matches_the_linear_reference(kernel):
    dy, offs = _packed([256], 512)
    d, w = _signs(seed=0), _signs(seed=1)
    got_row, got_col = _amax(kernel, dy, d, w, offs, 1)
    ref_row, ref_col = reference_row_rht_col_rht_amax(dy, d, w)
    torch.testing.assert_close(got_row[0], ref_row, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(got_col[0], ref_col, rtol=1e-3, atol=1e-3)


@_needs_amax
@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("group_sizes", [[128, 128], [256, 128, 384, 128]])
@torch.no_grad()
def test_amax_multi_group_matches_the_reference(kernel, group_sizes):
    dy, offs = _packed(group_sizes, 512)
    d, w = _signs(seed=0), _signs(seed=1)
    E = len(group_sizes)
    got_row, got_col = _amax(kernel, dy, d, w, offs, E)
    ref_row, ref_col = reference_group_row_rht_col_rht_amax(dy, d, w, offs, E)
    torch.testing.assert_close(got_row, ref_row, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(got_col, ref_col, rtol=1e-3, atol=1e-3)


@_needs_amax
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_each_sign_vector_drives_exactly_one_output(kernel):
    """Changing ``dgrad_rht`` may move only the rowwise amax, and vice versa.

    The cleanest statement that the two are not crossed: if the kernel wired them the
    other way round, each assertion below would fail in the opposite direction.
    """
    dy, offs = _packed([256], 512)
    d0, w0 = _signs(seed=0), _signs(seed=1)
    base_row, base_col = _amax(kernel, dy, d0, w0, offs, 1)

    row_only, col_unchanged = _amax(kernel, dy, _signs(seed=2), w0, offs, 1)
    assert row_only[0].item() != base_row[0].item(), "dgrad_rht must move amax_rht_dy"
    assert col_unchanged[0].item() == base_col[0].item(), (
        "dgrad_rht must not touch amax_rht_dy_t"
    )

    row_unchanged, col_only = _amax(kernel, dy, d0, _signs(seed=3), offs, 1)
    assert col_only[0].item() != base_col[0].item(), "wgrad_rht must move amax_rht_dy_t"
    assert row_unchanged[0].item() == base_row[0].item(), (
        "wgrad_rht must not touch amax_rht_dy"
    )


@_needs_amax
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_swapping_the_two_sign_vectors_changes_both_outputs(kernel):
    """The discriminating test for a crossed-argument bug."""
    dy, offs = _packed([256], 512)
    d, w = _signs(seed=0), _signs(seed=1)
    a_row, a_col = _amax(kernel, dy, d, w, offs, 1)
    b_row, b_col = _amax(kernel, dy, w, d, offs, 1)
    assert a_row[0].item() != b_row[0].item()
    assert a_col[0].item() != b_col[0].item()


@_needs_amax
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_identical_sign_vectors_are_not_assumed_equal(kernel):
    """``dgrad_rht == wgrad_rht`` must still compute each axis on its own data."""
    dy, offs = _packed([256], 384)
    d = _signs(seed=0)
    got_row, got_col = _amax(kernel, dy, d, d, offs, 1)
    ref_row, ref_col = reference_row_rht_col_rht_amax(dy, d, d)
    torch.testing.assert_close(got_row[0], ref_row, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(got_col[0], ref_col, rtol=1e-3, atol=1e-3)


@_needs_amax
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_amax_per_group_isolation(kernel):
    group_sizes = [128, 128, 128, 128]
    dy, offs = _packed(group_sizes, 512)
    d, w = _signs(seed=0), _signs(seed=1)
    base = _amax(kernel, dy, d, w, offs, 4)
    dy2 = dy.clone()
    dy2[256:384] *= 1000.0
    hot = _amax(kernel, dy2, d, w, offs, 4)
    for g in (0, 1, 3):
        assert hot[0][g].item() == base[0][g].item(), f"group {g} row leaked"
        assert hot[1][g].item() == base[1][g].item(), f"group {g} col leaked"


@_needs_amax
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_zero_gradient_gives_zero_amaxes_without_nan(kernel):
    dy, offs = _packed([128, 128], 512)
    got = _amax(kernel, torch.zeros_like(dy), _signs(seed=0), _signs(seed=1), offs, 2)
    assert torch.equal(torch.stack(got), torch.zeros(2, 2, device="cuda"))


@_needs_amax
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_resampled_signs_change_the_output_without_retracing(kernel):
    """V2 mutates its sign buffers in place, so they must stay runtime inputs.

    ``resample_nvfp4_rht_signs`` copies a fresh draw into the live buffer every
    accumulation microbatch. If the op lets the sign values into the traced graph,
    one of two things happens and neither raises: the graph is retraced on every
    draw, so each microbatch pays a compile, or the first draw is baked in, the
    transform stops cancelling, and the gradient is quietly wrong. Counting graphs
    is the only way to tell those apart from a working kernel, which is why this
    asserts on the graph count as well as on the outputs moving.
    """
    graphs = []

    def counting_backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    op = (
        triton_group_row_rht_col_rht_amax
        if kernel == "triton"
        else cutedsl_group_row_rht_col_rht_amax
    )

    @torch.compile(backend=counting_backend, fullgraph=True)
    def amax(dy, dgrad_rht, wgrad_rht, offs):
        return op(
            dy,
            dgrad_rht,
            wgrad_rht,
            offs,
            1,
            dy.shape[0],
            dy.shape[1],
            VARYING_FIRST_DIM,
            offs[-1:],
        )

    dy, offs = _packed([256], 512)
    d, w = _signs(seed=0), _signs(seed=1)
    first_row, first_col = amax(dy, d, w, offs)
    d.copy_(_signs(seed=2))
    w.copy_(_signs(seed=3))
    second_row, second_col = amax(dy, d, w, offs)

    assert len(graphs) == 1, f"resampling must not retrace; traced {len(graphs)} times"
    assert not torch.equal(first_row, second_row), "rowwise amax must follow dgrad_rht"
    assert not torch.equal(first_col, second_col), "colwise amax must follow wgrad_rht"


_BITWISE_CASES = [
    pytest.param([128, 256, 384, 128], 1024, id="ragged"),
    pytest.param([128, 384], 256, id="jagged-two-tiles"),
    pytest.param([256] * 8, 1408, id="eight-groups"),
    pytest.param([1408] * 4, 1408, id="deepseek-16B-gate-up"),
    pytest.param([128, 0, 256], 512, id="empty-middle-group"),
    pytest.param([128] * 64, 128, id="max-groups"),
]


@_needs_amax
@_skip_no_cutedsl
@pytest.mark.parametrize("group_sizes,hidden", _BITWISE_CASES)
@torch.no_grad()
def test_cutedsl_amax_matches_triton(group_sizes, hidden):
    """Bitwise by construction: both backends issue the same tcgen05 (128,128,16) bf16
    UMMAs over K = 128 in the same order on the same operand bytes, then round once to
    bf16. A mismatch is a bug or a Triton lowering change, never tolerance."""
    dy, offs = _packed(group_sizes, hidden, seed=225)
    d, w = _signs(seed=0), _signs(seed=1)
    E = len(group_sizes)
    t_row, t_col = _amax("triton", dy, d, w, offs, E)
    c_row, c_col = _amax("cutedsl", dy, d, w, offs, E)
    assert torch.equal(c_row, t_row) and torch.equal(c_col, t_col)


@_needs_amax
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_amax_excludes_rows_past_logical_packed_length(kernel):
    dy, offs = _packed([128, 128], 512)
    capacity = torch.full((512, 512), float("inf"), device="cuda", dtype=torch.bfloat16)
    capacity[:256] = dy
    d, w = _signs(seed=0), _signs(seed=1)
    op = (
        triton_group_row_rht_col_rht_amax
        if kernel == "triton"
        else cutedsl_group_row_rht_col_rht_amax
    )
    got = op(capacity, d, w, offs, 2, 512, 512, VARYING_FIRST_DIM, offs[-1:])
    full = _amax(kernel, dy, d, w, offs, 2)
    assert torch.equal(torch.stack(got), torch.stack(full))


@maybe_sm100
@_skip_no_cutedsl
@torch.no_grad()
def test_cutedsl_amax_rejects_too_many_groups():
    dy, offs = _packed([128] * 65, 128)
    with pytest.raises(ValueError, match="num_tensors must be <= 64"):
        _amax("cutedsl", dy, _signs(seed=0), _signs(seed=1), offs, 65)


# --- §11.3 ------------------------------------------------------------------


@_needs_ms_eden
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_return_order_is_rowwise_first(kernel):
    """§11.3 returns the dgrad pair first, like every sibling quantize op.

    Pinned by shape on a non-square input, where a swapped unpack is a shape error.
    On a square layer it would corrupt silently, which is why this is asserted rather
    than left to the type checker -- and why this op was moved off the design doc's
    columnwise-first spelling onto the directory-wide one.
    """
    dy, offs = _packed([256], 512)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax(kernel, dy, d, w, offs, 1)
    rng = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    row_codes, row_sf, col_codes, col_sf = _ms_eden(
        kernel, dy, ar, ac, d, w, offs, 1, rng
    )
    assert row_codes.shape == (256, 256), "rowwise operand must come first"
    assert col_codes.shape == (512, 128), "columnwise operand must come second"


@_needs_ms_eden
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_block_scales_never_exceed_the_eden_ceiling(kernel):
    """MS-EDEN caps block scales at 256, not 448 -- the reason its numerator is 1536."""
    dy, offs = _packed([256], 512)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax(kernel, dy, d, w, offs, 1)
    rng = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    _, row_sf, _, col_sf = _ms_eden(kernel, dy, ar, ac, d, w, offs, 1, rng)
    assert col_sf.float().max() <= 256.0
    assert row_sf.float().max() <= 256.0


@_needs_ms_eden
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_fixed_rng_state_reproduces_bitwise(kernel):
    dy, offs = _packed([256], 512)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax(kernel, dy, d, w, offs, 1)
    rng = torch.tensor([5, 6, 7, 8], dtype=torch.int64, device="cuda")
    a = _ms_eden(kernel, dy, ar, ac, d, w, offs, 1, rng)
    b = _ms_eden(kernel, dy, ar, ac, d, w, offs, 1, rng)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@_needs_ms_eden
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_ms_eden_is_unbiased(kernel):
    """``E[sampled scale] == corrected scale`` and ``E[dequant] == ideal_dequant``.

    Deliberately *not* ``E[dequant] == dy @ R_n``. The FP4 codes are RTNE, so the only
    unbiased step is the E4M3 rounding of the corrected block scale; the expectation
    converges on the unrounded Eden-corrected reconstruction, which the correction
    leaves a full FP4 quantization error away from the input. Asserting against the
    input would be asserting a property MS-EDEN does not have.

    The bound is five standard errors of the sample mean, so it tightens as ``draws``
    grows. A fixed fraction of the tensor's magnitude would instead be satisfied by
    any bias smaller than the residual FP4 error, which is most of them.
    """
    from torchao.prototype.mx_formats.utils import from_blocked

    from .nvfp4_reference import (
        EDEN_BLOCK_SCALE_MAX,
        reference_dequantize_rowwise,
        reference_dynamic_rht,
        reference_ms_eden,
    )

    M, N = 128, 256
    dy, offs = _packed([M], N)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax(kernel, dy, d, w, offs, 1)
    row_ref = reference_ms_eden(reference_dynamic_rht(dy, d, transpose=False), ar[0])
    col_ref = reference_ms_eden(reference_dynamic_rht(dy, w, transpose=True), ac[0])

    draws = 64
    row_scales, col_scales, dequants = [], [], []
    for i in range(draws):
        rng = torch.tensor([1, i, 2, i + 1000], dtype=torch.int64, device="cuda")
        row_codes, row_sf, _, col_sf = _ms_eden(kernel, dy, ar, ac, d, w, offs, 1, rng)
        assert torch.equal(row_codes, row_ref.codes), (
            "MS-EDEN codes are RTNE and must not move with the seed"
        )
        row_scales.append(from_blocked(row_sf, M, N // 16).float())
        col_scales.append(from_blocked(col_sf, N, M // 16).float())
        dequants.append(
            reference_dequantize_rowwise(
                row_codes, row_sf, ar[0], fp8_max=EDEN_BLOCK_SCALE_MAX
            )
        )

    for samples, target, label in (
        (torch.stack(row_scales), row_ref.corrected_scale, "rowwise block scale"),
        (torch.stack(col_scales), col_ref.corrected_scale, "colwise block scale"),
        (torch.stack(dequants), row_ref.ideal_dequant, "reconstruction"),
    ):
        mean = samples.mean(dim=0)
        se = samples.std(dim=0, unbiased=True) / math.sqrt(draws)
        assert (mean - target).norm() <= 5.0 * se.norm() + 1e-5 * target.norm(), (
            f"{label} mean is biased away from the unrounded Eden target"
        )


@_needs_ms_eden
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_codes_are_rtne_from_the_pre_correction_scale(kernel):
    """Codes come from RTNE against the *original* block scale, not from FP4 SR.

    MS-EDEN's randomness lives entirely in the block scale, so the codes have to be
    reproducible bitwise from the pre-correction, pre-SR scale at every seed. This is
    the assertion that separates MS-EDEN from ordinary NVFP4 stochastic rounding --
    a kernel that reached for ``_pack_fp4(..., STOCHASTIC_ROUNDING=True)`` passes
    every other test in this file. It also pins the whole deterministic chain at once:
    RHT-128, the 256 ceiling, the block amax, the TE scale chain, RTNE and the packing.
    """
    from .nvfp4_reference import reference_dynamic_rht, reference_ms_eden

    M, N = 256, 512
    dy, offs = _packed([M], N, seed=2)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax(kernel, dy, d, w, offs, 1)
    row_ref = reference_ms_eden(reference_dynamic_rht(dy, d, transpose=False), ar[0])
    col_ref = reference_ms_eden(reference_dynamic_rht(dy, w, transpose=True), ac[0])

    for seed in (0, 1, 17, 29):
        rng = torch.tensor([seed, 0, seed + 7, 0], dtype=torch.int64, device="cuda")
        row_codes, _, col_codes, _ = _ms_eden(kernel, dy, ar, ac, d, w, offs, 1, rng)
        assert_codes_bitwise(row_codes, row_ref.codes, f"rowwise codes @ seed {seed}")
        assert_codes_bitwise(col_codes, col_ref.codes, f"colwise codes @ seed {seed}")


@_needs_ms_eden
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_each_rng_slice_drives_exactly_one_output(kernel):
    """``rng_state`` is ``[col_seed, col_offset, row_seed, row_offset]``.

    Both axes draw from one tensor, so a kernel that hands an axis the wrong slice
    raises nothing -- it correlates the two operands' scales, which no shape or dtype
    check catches. Same failure mode as the crossed sign vectors in §11.2, and the
    same reason to test it explicitly.
    """
    M, N = 256, 512
    dy, offs = _packed([M], N)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax(kernel, dy, d, w, offs, 1)

    def run(rng):
        return _ms_eden(
            kernel,
            dy,
            ar,
            ac,
            d,
            w,
            offs,
            1,
            torch.tensor(rng, dtype=torch.int64, device="cuda"),
        )

    base = run([1, 2, 3, 4])
    for slot, name, drives_col in (
        (0, "col_seed", True),
        (1, "col_offset", True),
        (2, "row_seed", False),
        (3, "row_offset", False),
    ):
        rng = [1, 2, 3, 4]
        rng[slot] += 100
        row_codes, row_sf, col_codes, col_sf = run(rng)
        assert torch.equal(row_codes, base[0]), "codes are RTNE and never move"
        assert torch.equal(col_codes, base[2]), "codes are RTNE and never move"
        moved, held = (col_sf, row_sf) if drives_col else (row_sf, col_sf)
        moved_base, held_base = (base[3], base[1]) if drives_col else (base[1], base[3])
        assert not torch.equal(moved, moved_base), f"{name} must move its own scales"
        assert torch.equal(held, held_base), f"{name} must leave the other axis alone"


@_needs_ms_eden
@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("group_sizes", [[256], [128, 128], [256, 128, 384]])
@torch.no_grad()
def test_multi_group_matches_the_reference(kernel, group_sizes):
    """The only multi-group numerics test for §11.3, and the only scale-layout one.

    Codes are RTNE, so they are pinned bitwise. The scale bytes are one stochastic
    draw away and cannot be, but stochastic rounding always lands on one of the two
    E4M3 neighbours of the value it rounds, so every byte must be within one ULP of
    the reference's corrected scale -- positive E4M3 bytes are magnitude-monotonic,
    which makes a byte delta a ULP delta. That bound is loose on value and *tight* on
    position: a scale written to the wrong offset lands nowhere near its neighbour
    pair. It is what guards the columnwise swizzle, whose tiling restarts at every
    group boundary, and which no other test in this file reads.
    """
    from torchao.prototype.mx_formats.utils import from_blocked

    from .nvfp4_reference import (
        from_blocked_grouped,
        reference_group_row_rht_col_rht_quantize_ms_eden,
    )

    N = 512
    E = len(group_sizes)
    M = sum(group_sizes)
    dy, offs = _packed(group_sizes, N, seed=3)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax(kernel, dy, d, w, offs, E)
    rng = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    row_codes, row_sf, col_codes, col_sf = _ms_eden(
        kernel, dy, ar, ac, d, w, offs, E, rng
    )
    ref_row_codes, ref_row_scale, ref_col_codes, ref_col_scale = (
        reference_group_row_rht_col_rht_quantize_ms_eden(dy, ar, ac, d, w, offs, E)
    )

    assert_codes_bitwise(row_codes, ref_row_codes, "row codes")
    assert_codes_bitwise(col_codes, ref_col_codes, "col codes")
    assert_scales_adjacent(
        from_blocked(row_sf, M, N // 16),
        ref_row_scale.to(torch.float8_e4m3fn),
        "row scales",
    )
    assert_scales_adjacent(
        from_blocked_grouped(col_sf, N, group_sizes),
        ref_col_scale.to(torch.float8_e4m3fn),
        "col scales",
    )


@_needs_ms_eden
@_skip_no_cutedsl
@pytest.mark.parametrize("group_sizes,hidden", _BITWISE_CASES)
@torch.no_grad()
def test_cutedsl_ms_eden_matches_triton(group_sizes, hidden):
    """Codes are bitwise by construction (same UMMAs, one bf16 rounding, RTNE); the scale
    bytes are the same Philox word at the same counter through the same fp32 chain, so
    they are bitwise too. A mismatch is a bug or a Triton lowering change, never
    tolerance. The second rng_state has a negative seed (both key words 0xFFFFFFFx), an
    offset whose high word must be ignored, a max seed and an offset with the top bit of
    its low word set."""
    dy, offs = _packed(group_sizes, hidden, seed=225)
    d, w = _signs(seed=0), _signs(seed=1)
    E = len(group_sizes)
    ar, ac = _amax("cutedsl", dy, d, w, offs, E)
    for words in ([1, 2, 3, 4], [-7, 2**32 + 5, 2**63 - 1, 2**31]):
        rng = torch.tensor(words, dtype=torch.int64, device="cuda")
        t = _ms_eden("triton", dy, ar, ac, d, w, offs, E, rng)
        c = _ms_eden("cutedsl", dy, ar, ac, d, w, offs, E, rng)
        for got, ref, label in zip(
            c, t, ("row codes", "row scales", "col codes", "col scales")
        ):
            assert torch.equal(got, ref), f"{label} differ for rng_state {words}"


@_needs_ms_eden
@_skip_no_cutedsl
@pytest.mark.parametrize("group_sizes,hidden", _BITWISE_CASES)
@torch.no_grad()
def test_cutedsl_ms_eden_matches_triton_on_nan_blocks(group_sizes, hidden):
    """The RHT-128 on both axes spreads a NaN in ``dy`` over whole 1x16 blocks, so no
    block mixes a NaN with finite values and the backends agree independently of the
    maxNum block amax the raw-row kernels rely on. The amaxes are taken over the NaN-free
    tensor, since the NaN-propagating group amax would hide the block."""
    dy, offs = _packed(group_sizes, hidden, seed=225)
    dy[0, 0] = float("nan")
    dy[1, hidden - 1] = float("nan")
    dy[sum(group_sizes) - 1, 0:2] = float("nan")
    d, w = _signs(seed=0), _signs(seed=1)
    E = len(group_sizes)
    ar, ac = _amax("cutedsl", torch.nan_to_num(dy, 0.0, 0.0, 0.0), d, w, offs, E)
    rng = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    t = _ms_eden("triton", dy, ar, ac, d, w, offs, E, rng)
    c = _ms_eden("cutedsl", dy, ar, ac, d, w, offs, E, rng)
    for got, ref, label in zip(
        c, t, ("row codes", "row scales", "col codes", "col scales")
    ):
        assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8)), (
            f"{label} differ"
        )


@_needs_ms_eden
@_skip_no_cutedsl
@torch.no_grad()
def test_fast_path_fixed_rng_state_reproduces_bitwise():
    """The fast path is a separate compiled variant and, like the default, a pure function
    of ``(rng_state, position)``: same state, same bytes, launch after launch."""
    from torchao.prototype.moe_training.nvfp4_training._cutedsl_group_kernels_impl import (
        _compile_group_row_rht_col_rht_quantize_ms_eden_kernel,
    )

    dy, offs = _packed([256], 512)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax("cutedsl", dy, d, w, offs, 1)
    rng = torch.tensor([5, 6, 7, 8], dtype=torch.int64, device="cuda")
    a = _ms_eden_fast(dy, ar, ac, d, w, offs, 1, rng)
    b = _ms_eden_fast(dy, ar, ac, d, w, offs, 1, rng)
    for x, y in zip(a, b):
        assert torch.equal(x, y)
    idx = dy.device.index
    assert _compile_group_row_rht_col_rht_quantize_ms_eden_kernel(
        idx, True
    ) is not _compile_group_row_rht_col_rht_quantize_ms_eden_kernel(idx, False)


@_needs_ms_eden
@_skip_no_cutedsl
@pytest.mark.parametrize(
    "group_sizes", [[256], [128, 128], [256, 128, 384], [128, 256]]
)
@torch.no_grad()
def test_fast_path_scales_are_neighbours_of_the_default_path(group_sizes):
    """Both paths round the same fp32 corrected scale onto its two E4M3 neighbours (the
    hardware ``cvt.rs`` never leaves the pair, measured on sm_100a), so a fast byte is
    the default byte or one E4M3 step from it on every scale, on every group layout --
    and the codes are the default path's RTNE codes. The reference's corrected scale is
    held to the same one-step band as in ``test_multi_group_matches_the_reference``."""
    from torchao.prototype.mx_formats.utils import from_blocked

    from .nvfp4_reference import (
        from_blocked_grouped,
        reference_group_row_rht_col_rht_quantize_ms_eden,
    )

    N = 512
    E = len(group_sizes)
    M = sum(group_sizes)
    dy, offs = _packed(group_sizes, N, seed=3)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax("cutedsl", dy, d, w, offs, E)
    rng = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    row_codes, row_sf, col_codes, col_sf = _ms_eden_fast(dy, ar, ac, d, w, offs, E, rng)
    default = _ms_eden("cutedsl", dy, ar, ac, d, w, offs, E, rng)
    ref_row_codes, ref_row_scale, ref_col_codes, ref_col_scale = (
        reference_group_row_rht_col_rht_quantize_ms_eden(dy, ar, ac, d, w, offs, E)
    )

    assert_codes_bitwise(row_codes, ref_row_codes, "row codes")
    assert_codes_bitwise(col_codes, ref_col_codes, "col codes")
    assert torch.equal(row_codes, default[0]) and torch.equal(col_codes, default[2])
    assert_scales_adjacent(row_sf, default[1], "row scales vs the default path")
    assert_scales_adjacent(col_sf, default[3], "col scales vs the default path")
    assert_scales_adjacent(
        from_blocked(row_sf, M, N // 16),
        ref_row_scale.to(torch.float8_e4m3fn),
        "row scales",
    )
    assert_scales_adjacent(
        from_blocked_grouped(col_sf, N, group_sizes),
        ref_col_scale.to(torch.float8_e4m3fn),
        "col scales",
    )


@_needs_ms_eden
@_skip_no_cutedsl
@torch.no_grad()
def test_fast_path_keeps_the_uncorrected_scale_where_its_cross_dot_overflows():
    """A block whose sum of |e| x |q| reaches 2^128 (rotated values from ~3.5e36) overflows
    the fast cross dot, formed on the unscaled bf16 values (``_fast_cross_dot_overflows``),
    where Triton's ``<v, q>`` stays finite; ``rcp.approx.ftz`` of that inf is 0, so the ratio
    arrives as a finite 0. A select that rejected only inf and NaN kept it and stored the
    block's scale byte as 0x00 -- the whole block dequantized to zero, codes intact. The fast
    select rejects the zero too, so such a block keeps its uncorrected E4M3 scale (a
    correction of 1.0, Triton's fallback for an undefined ratio; on the grid, so the hardware
    rounding cannot move it) while every other scale stays one E4M3 step from the default
    path's. +-1e37 planted along one row's 128-column segment and one column's 128-row
    segment overflow blocks in both epilogues with both group amaxes finite; the default path
    stays bitwise with the Triton op here."""
    from torchao.prototype.mx_formats.utils import from_blocked

    from .nvfp4_reference import reference_dynamic_rht, reference_ms_eden

    M, N = 256, 512
    dy, offs = _packed([M], N, seed=3)
    dy[5, :128] = _signs(seed=4).to(torch.bfloat16) * 1e37
    dy[128:, 300] = _signs(seed=5).to(torch.bfloat16) * 1e37
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax("cutedsl", dy, d, w, offs, 1)
    assert torch.isfinite(ar).all() and torch.isfinite(ac).all()
    rng = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    fast = _ms_eden_fast(dy, ar, ac, d, w, offs, 1, rng)
    default = _ms_eden("cutedsl", dy, ar, ac, d, w, offs, 1, rng)
    for got, ref in zip(default, _ms_eden("triton", dy, ar, ac, d, w, offs, 1, rng)):
        assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))
    row_rht = reference_dynamic_rht(dy, d, transpose=False)
    col_rht = reference_dynamic_rht(dy, w, transpose=True)
    row_ref = reference_ms_eden(row_rht, ar[0])
    col_ref = reference_ms_eden(col_rht, ac[0])
    assert_codes_bitwise(fast[0], row_ref.codes, "row codes")
    assert_codes_bitwise(fast[2], col_ref.codes, "col codes")
    assert torch.equal(fast[0], default[0]) and torch.equal(fast[2], default[2])

    for got, base, rht, ref, (rows, cols), label in (
        (fast[1], default[1], row_rht, row_ref, (M, N // 16), "rowwise"),
        (fast[3], default[3], col_rht, col_ref, (N, M // 16), "colwise"),
    ):
        got = from_blocked(got, rows, cols).view(torch.uint8)
        base = from_blocked(base, rows, cols).view(torch.uint8)
        zeroed = (got == 0) & (base != 0)
        assert not zeroed.any(), (
            f"{label}: {int(zeroed.sum())} fast scale bytes are 0x00 where the default "
            "path's are not; (block, fast, default) = "
            f"{[(r, c, hex(got[r, c]), hex(base[r, c])) for r, c in zeroed.nonzero()[:4].tolist()]}"
        )
        overflow = _fast_cross_dot_overflows(rht, ref.codes)
        assert overflow.any(), f"{label}: no block overflows the fast cross dot"
        uncorrected = ref.block_scale.to(torch.float8_e4m3fn).view(torch.uint8)
        assert torch.equal(got[overflow], uncorrected[overflow]), (
            f"{label}: an overflowed block's fast scale is not its uncorrected E4M3 scale"
        )
        assert_scales_adjacent(
            got[~overflow], base[~overflow], f"{label} scales vs the default path"
        )


@_needs_ms_eden
@_skip_no_cutedsl
@torch.no_grad()
def test_fast_path_rounds_up_with_the_fractional_position():
    """The acceptance test of the stochastic rounding, over 2^20 scales per axis.

    Both paths round the reference's fp32 corrected scale: every scale clear of the E4M3
    grid lands on one of its two neighbours, and the round-up rate follows the fractional
    position between them within 0.01 in each of 16 bins and within five standard errors
    overall (the fast path truncates the position to 16 bits, a bias of at most 15 / 2^20
    of a step, far below resolution). A correction fed the clamped scaled values misses
    the pair on 0.3 % of the scales and rounds up 0.035 less often than the position says
    (z about -90). The two paths are then held to each other: the mean signed E4M3-step
    difference fast - default is zero within five standard errors overall and per bin, and
    they differ on about a third of the scales (independent decisions at uniform
    positions: ``E[2 p (1 - p)] = 1/3``).
    """
    from torchao.prototype.mx_formats.utils import from_blocked

    from .nvfp4_reference import reference_dynamic_rht, reference_ms_eden

    M = N = 4096
    dy, offs = _packed([M], N, seed=7)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax("cutedsl", dy, d, w, offs, 1)
    rng = torch.tensor([11, 12, 13, 14], dtype=torch.int64, device="cuda")
    fast = _ms_eden_fast(dy, ar, ac, d, w, offs, 1, rng)
    default = _ms_eden("cutedsl", dy, ar, ac, d, w, offs, 1, rng)
    row_ref = reference_ms_eden(reference_dynamic_rht(dy, d, transpose=False), ar[0])
    col_ref = reference_ms_eden(reference_dynamic_rht(dy, w, transpose=True), ac[0])
    assert_codes_bitwise(fast[0], row_ref.codes, "row codes")
    assert_codes_bitwise(fast[2], col_ref.codes, "col codes")

    for got, base, ref, (rows, cols), label in (
        (fast[1], default[1], row_ref.corrected_scale, (M, N // 16), "rowwise"),
        (fast[3], default[3], col_ref.corrected_scale, (N, M // 16), "colwise"),
    ):
        got = from_blocked(got, *(rows, cols))
        base = from_blocked(base, *(rows, cols))
        lo, hi, frac = _e4m3_neighbours(ref)
        keep = frac > 0  # scales exactly on the grid have nothing to round
        n = int(keep.sum())
        assert n >= 2**20 - 2**12, f"{label}: {n} scales"
        # The reference sums the dot products in another order than the kernels, so only
        # scales clear of a grid point by more than that noise are held to the pair.
        clear = (frac > 2**-10) & (frac < 1 - 2**-10)
        bins = (frac[keep] * 16).long().clamp_(max=15)
        var = (frac[keep] * (1 - frac[keep])).sum()
        ups = {}
        for path, x in (("fast", got), ("default", base)):
            x = x.float()
            outside = ~((x == lo) | (x == hi)) & clear
            assert not outside.any(), (
                f"{label}: {int(outside.sum())} {path} scales are not a neighbour of the "
                "reference's corrected scale"
            )
            up = (x[keep] == hi[keep]).float()
            z = (up.sum() - frac[keep].sum()) / var.sqrt()
            assert abs(z.item()) < 5.0, (
                f"{label}: {path} round-up rate vs position, z = {z:.2f}"
            )
            ups[path] = up
        # Signed byte delta = signed E4M3-step delta (positive bytes are monotonic).
        steps = (
            got.view(torch.uint8).to(torch.int16)
            - base.view(torch.uint8).to(torch.int16)
        ).float()
        assert (steps.abs() <= 1).all(), f"{label}: fast is not a neighbour of default"
        steps, frac = steps[keep], frac[keep]
        z = steps.mean() * math.sqrt(n) / steps.std()
        assert abs(z.item()) < 5.0, (
            f"{label}: fast vs default round-up rate, z = {z:.2f}"
        )
        differ = (steps != 0).float().mean().item()
        assert 0.28 <= differ <= 0.38, (
            f"{label}: paths differ on {differ:.3f}, not ~1/3"
        )
        for b in range(16):
            sel = bins == b
            nb = int(sel.sum())
            position = frac[sel].mean().item()
            for path, up in ups.items():
                assert abs(up[sel].mean().item() - position) < 0.01, (
                    f"{label} bin {b}: {path} P(up) {up[sel].mean():.4f} vs position "
                    f"{position:.4f} over {nb} scales"
                )
            z_bin = steps[sel].mean() * math.sqrt(nb) / steps[sel].std()
            # mean(steps) = P(up | fast) - P(up | default) in E4M3 steps.
            assert abs(z_bin.item()) < 5.0 and abs(steps[sel].mean().item()) < 0.01, (
                f"{label} bin {b}: fast vs default round-up rate, z = {z_bin:.2f}, "
                f"delta = {steps[sel].mean():.4f} over {nb} scales"
            )


@_needs_ms_eden
@_skip_no_cutedsl
@torch.no_grad()
def test_fast_path_is_unbiased():
    """``test_ms_eden_is_unbiased`` for the fast path: ``E[sampled scale] == corrected
    scale`` and ``E[dequant] == ideal_dequant`` within five standard errors over 64
    draws (the 16-bit truncation bias, at most 15 / 2^20 of an E4M3 step, is far below
    the bound)."""
    from torchao.prototype.mx_formats.utils import from_blocked

    from .nvfp4_reference import (
        EDEN_BLOCK_SCALE_MAX,
        reference_dequantize_rowwise,
        reference_dynamic_rht,
        reference_ms_eden,
    )

    M, N = 128, 256
    dy, offs = _packed([M], N)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax("cutedsl", dy, d, w, offs, 1)
    row_ref = reference_ms_eden(reference_dynamic_rht(dy, d, transpose=False), ar[0])
    col_ref = reference_ms_eden(reference_dynamic_rht(dy, w, transpose=True), ac[0])

    draws = 64
    row_scales, col_scales, dequants = [], [], []
    for i in range(draws):
        rng = torch.tensor([1, i, 2, i + 1000], dtype=torch.int64, device="cuda")
        row_codes, row_sf, _, col_sf = _ms_eden_fast(dy, ar, ac, d, w, offs, 1, rng)
        assert torch.equal(row_codes, row_ref.codes), (
            "MS-EDEN codes are RTNE and must not move with the seed"
        )
        row_scales.append(from_blocked(row_sf, M, N // 16).float())
        col_scales.append(from_blocked(col_sf, N, M // 16).float())
        dequants.append(
            reference_dequantize_rowwise(
                row_codes, row_sf, ar[0], fp8_max=EDEN_BLOCK_SCALE_MAX
            )
        )

    for samples, target, label in (
        (torch.stack(row_scales), row_ref.corrected_scale, "rowwise block scale"),
        (torch.stack(col_scales), col_ref.corrected_scale, "colwise block scale"),
        (torch.stack(dequants), row_ref.ideal_dequant, "reconstruction"),
    ):
        mean = samples.mean(dim=0)
        se = samples.std(dim=0, unbiased=True) / math.sqrt(draws)
        assert (mean - target).norm() <= 5.0 * se.norm() + 1e-5 * target.norm(), (
            f"{label} mean is biased away from the unrounded Eden target"
        )


@_needs_ms_eden
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_ms_eden_keeps_rn_error_and_sr_unbiasedness(kernel):
    """RN, per-element SR, MS-EDEN default and MS-EDEN fast on one RHT-128 matrix.

    TransformerEngine's ``check_quantization_nvfp4_versus_reference`` quantizes once with
    RN and ``n_iters`` times with SR, averages the SR dequants and asserts the average is
    closer to the input than RN is. This is that comparison with MS-EDEN in it, all four
    on ``x = dy @ R_128`` at the EDEN ceiling so the two-level scale grid is the same:

    * RN: RTNE codes against the RTNE E4M3 block scale (``nvfp4_reference_quantize``).
      Deterministic, so its mean of draws is itself.
    * per-element SR: the same block scale, every element rounded onto its two E2M1
      neighbours with probability given by its fractional position
      (``pack_fp4_stochastic``; no RHT-128 kernel in the tree does this). Unbiased for
      what it rounds, which is the *clamped* scaled value: the block amax saturates at 6
      whenever its E4M3 block scale rounded down, so its mean of draws converges on the
      clamped input and its inner product with ``x`` falls short of ``<x, x>`` by the
      clamp's share (-0.35 % here). RN inherits the same shrink and adds RTNE's own --
      on a density that falls with ``|x|`` more elements round down than up (-0.41 %).
    * MS-EDEN default and fast (CuteDSL only): the RN codes and the Eden-corrected block
      scale, stochastically rounded to E4M3 by software Philox or by hardware ``cvt.rs``.

    ``E[MS-EDEN dequant]`` is ``ideal_dequant``, not ``x``: the codes are RTNE, and the
    correction ``<v, v> / <v, q>`` rescales each block so that its inner product with the
    input is preserved, which leaves the reconstruction a full FP4 quantization error away
    from ``x`` (``MSEdenReferenceOutput``). So the RMSE of the MS-EDEN mean of draws
    against ``x`` stays at RN's level (0.95 RN, held above 0.5 RN) while per-element SR's
    falls with the draw count (0.21 RN after 64). What MS-EDEN makes unbiased is the inner
    product ``<x, dequant>``, the quantity the GEMM consumes: the relative bias
    ``(<x, mean dq> - <x, x>) / <x, x>`` is zero within five standard errors for both
    MS-EDEN paths and, for SR, equal to the clamp's within five.

    Single-draw error: SR pays for its unbiasedness with variance -- uniform positions
    give ``E[p (1 - p)] = 1/6`` of a step squared against RTNE's ``1/12`` -- so its
    single-draw RMSE is about ``sqrt 2`` times RN's. MS-EDEN only moves the block scale by one
    E4M3 step, so its single-draw MSE is the ideal reconstruction's plus the scale draw's
    variance, ``p (1 - p) (step / S_enc)^2 q^2`` per element -- an identity held within
    five standard errors in place of a tolerance on RN's RMSE. With the block scales in
    E4M3's normal range, as here, one step is at most 1/8 of the scale, well below the
    E2M1 element step, so that variance stays small: on this Gaussian input MS-EDEN
    lands 2 % above RN and at ``1 / sqrt 2`` of SR (a subnormal block scale can move by
    half of itself and would put MS-EDEN above SR). The table and the reference line
    under it are what ``pytest -s`` prints.
    """
    from .nvfp4_reference import (
        EDEN_BLOCK_SCALE_MAX,
        FP4_E2M1_MAX,
        decode_fp4_codes,
        global_encode_scale,
        nvfp4_reference_quantize,
        pack_fp4_stochastic,
        reference_dequantize_rowwise,
        reference_dynamic_rht,
        reference_ms_eden,
    )

    M, N = 256, 512
    dy, offs = _packed([M], N, seed=11)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax(kernel, dy, d, w, offs, 1)
    x = reference_dynamic_rht(dy, d, transpose=False).float()
    rn = nvfp4_reference_quantize(x, ar[0], fp8_max=EDEN_BLOCK_SCALE_MAX)
    ref = reference_ms_eden(x, ar[0])
    s_enc = global_encode_scale(ar[0], EDEN_BLOCK_SCALE_MAX)
    generator = torch.Generator(device="cuda").manual_seed(0)

    def dequant(codes, sf, is_swizzled):
        return reference_dequantize_rowwise(
            codes, sf, ar[0], is_swizzled=is_swizzled, fp8_max=EDEN_BLOCK_SCALE_MAX
        )

    def ms_eden(i, fast):
        rng = torch.tensor([1, i, 2, i + 1000], dtype=torch.int64, device="cuda")
        args = (dy, ar, ac, d, w, offs, 1, rng)
        row_codes, row_sf, _, _ = (
            _ms_eden_fast(*args) if fast else _ms_eden(kernel, *args)
        )
        assert torch.equal(row_codes, rn.codes), "MS-EDEN codes are the RN codes"
        return dequant(row_codes, row_sf, is_swizzled=True)

    quantizers = {
        "RN": lambda i: dequant(rn.codes, rn.scales, is_swizzled=False),
        "per-element SR": lambda i: dequant(
            pack_fp4_stochastic(rn.scaled, generator), rn.scales, is_swizzled=False
        ),
        "MS-EDEN default": lambda i: ms_eden(i, fast=False),
    }
    if kernel == "cutedsl":
        quantizers["MS-EDEN fast"] = lambda i: ms_eden(i, fast=True)

    draws = 64
    # fp64 sums: the relative biases below are ~1e-5 of <x, x>.
    xx = x.double().pow(2).sum()
    stats = {}
    print(f"\n[{kernel}] {draws} draws, {M}x{N}")
    print(
        f"{'scheme':17}{'rmse_single':>12}{'rmse_mean':>11}{'ip_bias_rel':>13}"
        f"{'+-se':>9}{'rmse_mean_vs_ideal':>20}"
    )
    for name, quantize in quantizers.items():
        dq = torch.stack([quantize(i) for i in range(draws)])
        mse = (dq - x).pow(2).mean(dim=(1, 2))
        bias = ((dq.double() * x.double()).sum(dim=(1, 2)) - xx) / xx
        mean = dq.mean(dim=0)
        s = stats[name] = dict(
            rmse_single=mse.sqrt().mean().item(),
            mse=mse.mean().item(),
            mse_se=mse.std().item() / math.sqrt(draws),
            rmse_mean=(mean - x).pow(2).mean().sqrt().item(),
            ip_bias=bias.mean().item(),
            ip_se=bias.std().item() / math.sqrt(draws),
            rmse_mean_vs_ideal=(mean - ref.ideal_dequant).pow(2).mean().sqrt().item(),
            mean=mean,
            se=dq.std(dim=0) / math.sqrt(draws),
        )
        print(
            f"{name:17}{s['rmse_single']:12.4e}{s['rmse_mean']:11.4e}"
            f"{s['ip_bias']:+13.2e}{s['ip_se']:9.1e}{s['rmse_mean_vs_ideal']:20.4e}"
        )

    rn_, sr = stats["RN"], stats["per-element SR"]
    # E[SR dequant] is the clamped scaled value; its inner-product bias is the clamp's.
    enc = rn.encode_scale.repeat_interleave(16, dim=1)
    clamped = rn.scaled.clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX) / enc
    clamp_bias = (((clamped.double() * x.double()).sum() - xx) / xx).item()
    # E[MS-EDEN single-draw MSE] = the ideal reconstruction's + the scale draw's variance.
    lo, hi, frac = _e4m3_neighbours(ref.corrected_scale)
    scale_var = frac * (1 - frac) * ((hi - lo) / s_enc).pow(2)
    ideal_err = (ref.ideal_dequant - x).pow(2)
    q = decode_fp4_codes(rn.codes)
    predicted = (ideal_err + scale_var.repeat_interleave(16, dim=1) * q.pow(2)).mean()
    print(
        f"RMSE(ideal, x) {ideal_err.mean().sqrt():.4e}; MS-EDEN predicted rmse_single "
        f"{predicted.sqrt():.4e}; clamp ip_bias_rel {clamp_bias:+.2e}"
    )

    # TransformerEngine's assertion: the averaged SR dequant is closer to x than RN is.
    assert sr["rmse_mean"] < rn_["rmse_single"], "mean of SR draws must beat RN"
    bound = 5 * sr["se"].norm() + 1e-5 * clamped.norm()
    assert (sr["mean"] - clamped).norm() <= bound, (
        "mean of SR draws is biased away from the clamped input"
    )
    assert abs(sr["ip_bias"] - clamp_bias) <= 5 * sr["ip_se"], (
        f"SR <x, dq> bias {sr['ip_bias']:+.2e} +- {sr['ip_se']:.1e} "
        f"vs clamp {clamp_bias:+.2e}"
    )
    for name, s in stats.items():
        if not name.startswith("MS-EDEN"):
            continue
        bound = 5 * s["se"].norm() + 1e-5 * ref.ideal_dequant.norm()
        assert (s["mean"] - ref.ideal_dequant).norm() <= bound, (
            f"{name}: mean of draws is biased away from the unrounded Eden target"
        )
        assert abs(s["ip_bias"]) <= 5 * s["ip_se"], (
            f"{name}: <x, dq> bias {s['ip_bias']:+.2e} +- {s['ip_se']:.1e}"
        )
        assert abs(s["mse"] - predicted) <= 5 * s["mse_se"], (
            f"{name}: single-draw MSE {s['mse']:.4e} +- {s['mse_se']:.1e} vs ideal + "
            f"scale variance {predicted:.4e}"
        )
        assert s["rmse_single"] < sr["rmse_single"], f"{name}: not below per-element SR"
        # E[dq] is the ideal reconstruction, a per-block rescaling of RN's that cannot
        # remove a rounding error uncorrelated with the block (SR's rmse_mean_vs_ideal is
        # that distance), so no number of draws brings the mean near x. Half of RN's RMSE
        # separates it from SR's mean, already at RN / 5 after 64 draws.
        assert s["rmse_mean"] >= 0.5 * rn_["rmse_single"], (
            f"{name}: mean of draws must not converge on x"
        )
    if "MS-EDEN fast" in stats:
        fast, default = stats["MS-EDEN fast"], stats["MS-EDEN default"]
        se = (fast["se"].pow(2) + default["se"].pow(2)).sqrt()
        assert (fast["mean"] - default["mean"]).norm() <= 5 * se.norm(), (
            "fast and default means of draws disagree"
        )
        for key, se_key in (("ip_bias", "ip_se"), ("mse", "mse_se")):
            assert abs(fast[key] - default[key]) <= 5 * math.hypot(
                fast[se_key], default[se_key]
            ), f"fast and default {key} disagree"


@_needs_ms_eden
@_skip_no_cutedsl
@pytest.mark.parametrize("group_sizes,hidden", [([1408] * 4, 1408), ([2048] * 4, 2048)])
@torch.no_grad()
def test_fast_path_sqnr_matches_the_default_path(group_sizes, hidden):
    """Two unbiased roundings of the same corrected scales reconstruct the rotated input
    equally well: dequantized SQNR within 0.1 dB of the default path on both axes at the
    recipe shapes, with no NaN scale byte and zero input quantized to zero."""
    from torchao.float8.float8_utils import compute_error
    from torchao.prototype.mx_formats.utils import from_blocked

    from ._assertions import assert_zero_quantized
    from .nvfp4_reference import (
        EDEN_BLOCK_SCALE_MAX,
        from_blocked_grouped,
        reference_dequantize_rowwise,
        reference_dynamic_rht,
    )

    E = len(group_sizes)
    M = sum(group_sizes)
    dy, offs = _packed(group_sizes, hidden, seed=5)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax("cutedsl", dy, d, w, offs, E)
    rng = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    outs = {
        "default": _ms_eden("cutedsl", dy, ar, ac, d, w, offs, E, rng),
        "fast": _ms_eden_fast(dy, ar, ac, d, w, offs, E, rng),
    }
    sqnr = {}
    for path, (row_codes, row_sf, col_codes, col_sf) in outs.items():
        for sf in (row_sf, col_sf):
            assert not (sf.view(torch.uint8) & 0x7F == 0x7F).any(), f"{path}: NaN scale"
        row_scale = from_blocked(row_sf, M, hidden // 16)
        col_scale = from_blocked_grouped(col_sf, hidden, group_sizes)
        start = 0
        for g, size in enumerate(group_sizes):
            end = start + size
            rht = reference_dynamic_rht(dy[start:end], d, transpose=False)
            dq = reference_dequantize_rowwise(
                row_codes[start:end],
                row_scale[start:end],
                ar[g],
                is_swizzled=False,
                fp8_max=EDEN_BLOCK_SCALE_MAX,
            )
            sqnr[path, "row", g] = compute_error(rht.float(), dq).item()
            rht_t = reference_dynamic_rht(dy[start:end], w, transpose=True)
            dq_t = reference_dequantize_rowwise(
                col_codes[:, start // 2 : end // 2],
                col_scale[:, start // 16 : end // 16],
                ac[g],
                is_swizzled=False,
                fp8_max=EDEN_BLOCK_SCALE_MAX,
            )
            sqnr[path, "col", g] = compute_error(rht_t.float(), dq_t).item()
            start = end
    for (path, axis, g), value in sqnr.items():
        if path == "fast":
            assert abs(value - sqnr["default", axis, g]) <= 0.1, (
                f"{axis} group {g}: fast {value:.3f} dB vs default "
                f"{sqnr['default', axis, g]:.3f} dB"
            )

    zero = torch.zeros_like(dy)
    zr, zc = _amax("cutedsl", zero, d, w, offs, E)
    row_codes, row_sf, col_codes, col_sf = _ms_eden_fast(
        zero, zr, zc, d, w, offs, E, rng
    )
    assert_zero_quantized(row_codes, row_sf)
    assert_zero_quantized(col_codes, col_sf)


@_needs_ms_eden
@_skip_no_cutedsl
@torch.no_grad()
def test_fast_path_each_rng_slice_drives_exactly_one_output():
    """``test_each_rng_slice_drives_exactly_one_output`` for the fast path, plus the
    statement that two seeds are two independent streams: their round-up decisions agree
    on ~2/3 of the scales (uniform fractional positions: ``E[p^2 + (1 - p)^2] = 2/3``),
    not on all of them and not on the software path's fraction."""
    M, N = 256, 512
    dy, offs = _packed([M], N)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax("cutedsl", dy, d, w, offs, 1)

    def run(rng):
        return _ms_eden_fast(
            dy,
            ar,
            ac,
            d,
            w,
            offs,
            1,
            torch.tensor(rng, dtype=torch.int64, device="cuda"),
        )

    base = run([1, 2, 3, 4])
    for slot, name, drives_col in (
        (0, "col_seed", True),
        (1, "col_offset", True),
        (2, "row_seed", False),
        (3, "row_offset", False),
    ):
        rng = [1, 2, 3, 4]
        rng[slot] += 100
        row_codes, row_sf, col_codes, col_sf = run(rng)
        assert torch.equal(row_codes, base[0]), "codes are RTNE and never move"
        assert torch.equal(col_codes, base[2]), "codes are RTNE and never move"
        moved, held = (col_sf, row_sf) if drives_col else (row_sf, col_sf)
        moved_base, held_base = (base[3], base[1]) if drives_col else (base[1], base[3])
        assert not torch.equal(moved, moved_base), f"{name} must move its own scales"
        assert torch.equal(held, held_base), f"{name} must leave the other axis alone"
        agree = (moved.view(torch.uint8) == moved_base.view(torch.uint8)).float().mean()
        assert 0.60 <= agree.item() <= 0.74, (
            f"{name}: two seeds agree on {agree:.3f} of the scales, not ~2/3"
        )


@_needs_ms_eden
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_ms_eden_excludes_rows_past_logical_packed_length(kernel):
    """Capacity rows beyond offsets[-1] are never read; the tails are left as allocated in
    both backends. Codes and rowwise scales do not depend on the capacity, so their valid
    prefixes equal the full-extent call's. The columnwise stochastic draw is counted over
    the capacity (Triton's ``INNER = M``), so the full-extent call is a different draw and
    its scales are compared between two capacity buffers that differ only past
    ``offsets[-1]``: equal valid prefixes show the valid outputs do not depend on the
    tail's contents. The CuteDSL capacity draw is also compared with Triton's."""
    dy, offs = _packed([128, 128], 512)
    capacity = torch.full((512, 512), float("inf"), device="cuda", dtype=torch.bfloat16)
    capacity[:256] = dy
    capacity_zero = torch.zeros((512, 512), device="cuda", dtype=torch.bfloat16)
    capacity_zero[:256] = dy
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax(kernel, dy, d, w, offs, 2)
    rng = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    op = (
        triton_group_row_rht_col_rht_quantize_ms_eden
        if kernel == "triton"
        else cutedsl_group_row_rht_col_rht_quantize_ms_eden
    )
    got = op(
        capacity, ar, ac, d, w, offs, 2, 512, 512, VARYING_FIRST_DIM, rng, offs[-1:]
    )
    other = op(
        capacity_zero,
        ar,
        ac,
        d,
        w,
        offs,
        2,
        512,
        512,
        VARYING_FIRST_DIM,
        rng,
        offs[-1:],
    )
    full = _ms_eden(kernel, dy, ar, ac, d, w, offs, 2, rng)
    # Row codes / row scales: the valid rows.
    assert torch.equal(got[0][:256], full[0]) and torch.equal(got[1][:256], full[1])
    # Col codes: the valid token columns.
    assert torch.equal(got[2][:, :128], full[2])
    # Col scales: the valid per-group prefix, between the two capacity buffers.
    valid = 512 * 256 // 16
    assert torch.equal(got[3].flatten()[:valid], other[3].flatten()[:valid])
    # The columnwise draw is counted over the capacity (Triton's INNER = M), so the
    # capacity call's valid col scales are a different draw from the full-extent call's
    # -- both backends.
    assert not torch.equal(got[3].flatten()[:valid], full[3].flatten())
    if kernel == "cutedsl":
        # The only spare-capacity case: the CuteDSL columnwise draw must be pitched by
        # the capacity like Triton's, not by offsets[-1].
        tri = triton_group_row_rht_col_rht_quantize_ms_eden(
            capacity, ar, ac, d, w, offs, 2, 512, 512, VARYING_FIRST_DIM, rng, offs[-1:]
        )
        assert torch.equal(got[3].flatten()[:valid], tri[3].flatten()[:valid])


@maybe_sm100
@_skip_no_cutedsl
@torch.no_grad()
def test_cutedsl_ms_eden_rejects_too_many_groups():
    dy, offs = _packed([128] * 65, 128)
    amax = torch.ones(65, dtype=torch.float32, device="cuda")
    rng = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError, match="num_tensors must be <= 64"):
        _ms_eden(
            "cutedsl", dy, amax, amax, _signs(seed=0), _signs(seed=1), offs, 65, rng
        )


# --- wrapper layer, runs today ----------------------------------------------


@maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_register_fake_shapes_and_return_order(kernel):
    from torch._subclasses.fake_tensor import FakeTensorMode

    op = (
        triton_group_row_rht_col_rht_amax
        if kernel == "triton"
        else cutedsl_group_row_rht_col_rht_amax
    )
    ms_eden_op = (
        triton_group_row_rht_col_rht_quantize_ms_eden
        if kernel == "triton"
        else cutedsl_group_row_rht_col_rht_quantize_ms_eden
    )
    M, N, E = 512, 256, 2
    with FakeTensorMode():
        dy = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        sv = torch.empty(128, dtype=torch.int8, device="cuda")
        offs = torch.empty(E, dtype=torch.int32, device="cuda")
        amax = torch.empty(E, dtype=torch.float32, device="cuda")
        rng = torch.empty(4, dtype=torch.int64, device="cuda")
        row, col = op(dy, sv, sv, offs, E, M, N, VARYING_FIRST_DIM, None)
        assert row.shape == (E,) and col.shape == (E,)
        out = ms_eden_op(
            dy, amax, amax, sv, sv, offs, E, M, N, VARYING_FIRST_DIM, rng, None
        )
    # Rowwise pair first.
    assert [tuple(t.shape) for t in out] == [
        (M, N // 2),
        (M, N // 16),
        (N, M // 2),
        (N, M // 16),
    ]


@maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("which", ["dgrad_rht", "wgrad_rht"])
@torch.no_grad()
def test_rejects_a_non_128_sign_vector(kernel, which):
    M, N, E = 512, 256, 2
    dy = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    offs = torch.tensor([256, 512], dtype=torch.int32, device="cuda")
    good = torch.ones(128, dtype=torch.int8, device="cuda")
    bad = torch.ones(16, dtype=torch.int8, device="cuda")
    d, w = (bad, good) if which == "dgrad_rht" else (good, bad)
    with pytest.raises(ValueError, match=rf"{which} must be a \(128,\) tensor"):
        _amax(kernel, dy, d, w, offs, E)


@maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_ms_eden_always_requires_an_rng_state(kernel):
    """Unlike the cast quantizers there is no deterministic mode to fall back to."""
    M, N, E = 512, 256, 2
    dy = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    offs = torch.tensor([256, 512], dtype=torch.int32, device="cuda")
    sv = torch.ones(128, dtype=torch.int8, device="cuda")
    amax = torch.ones(E, dtype=torch.float32, device="cuda")
    op = (
        triton_group_row_rht_col_rht_quantize_ms_eden
        if kernel == "triton"
        else cutedsl_group_row_rht_col_rht_quantize_ms_eden
    )
    with pytest.raises(TypeError, match="rng_state must be a torch.Tensor"):
        op(dy, amax, amax, sv, sv, offs, E, M, N, VARYING_FIRST_DIM, None, offs[-1:])
