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

Both backends of the amax op are selected by the ``kernel`` parametrization
(``_KERNELS``); the MS-EDEN tests consume the Triton amax only.
"""

import math

import pytest
import torch

from torchao.prototype.moe_training.nvfp4_training.group_row_rht_col_rht_amax_cutedsl import (
    cutedsl_group_row_rht_col_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)

from ._assertions import assert_codes_bitwise, assert_scales_adjacent
from ._v2_marks import TRITON_AVAILABLE, kernel_gate, maybe_sm100
from .nvfp4_reference import (
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
@torch.no_grad()
def test_return_order_is_rowwise_first():
    """§11.3 returns the dgrad pair first, like every sibling quantize op.

    Pinned by shape on a non-square input, where a swapped unpack is a shape error.
    On a square layer it would corrupt silently, which is why this is asserted rather
    than left to the type checker -- and why this op was moved off the design doc's
    columnwise-first spelling onto the directory-wide one.
    """
    dy, offs = _packed([256], 512)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax("triton", dy, d, w, offs, 1)
    row_codes, row_sf, col_codes, col_sf = (
        triton_group_row_rht_col_rht_quantize_ms_eden(
            dy,
            ar,
            ac,
            d,
            w,
            offs,
            1,
            256,
            512,
            VARYING_FIRST_DIM,
            torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda"),
            offs[-1:],
        )
    )
    assert row_codes.shape == (256, 256), "rowwise operand must come first"
    assert col_codes.shape == (512, 128), "columnwise operand must come second"


@_needs_ms_eden
@torch.no_grad()
def test_block_scales_never_exceed_the_eden_ceiling():
    """MS-EDEN caps block scales at 256, not 448 -- the reason its numerator is 1536."""
    dy, offs = _packed([256], 512)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax("triton", dy, d, w, offs, 1)
    _, row_sf, _, col_sf = triton_group_row_rht_col_rht_quantize_ms_eden(
        dy,
        ar,
        ac,
        d,
        w,
        offs,
        1,
        256,
        512,
        VARYING_FIRST_DIM,
        torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda"),
        offs[-1:],
    )
    assert col_sf.float().max() <= 256.0
    assert row_sf.float().max() <= 256.0


@_needs_ms_eden
@torch.no_grad()
def test_fixed_rng_state_reproduces_bitwise():
    dy, offs = _packed([256], 512)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax("triton", dy, d, w, offs, 1)
    rng = torch.tensor([5, 6, 7, 8], dtype=torch.int64, device="cuda")
    args = (dy, ar, ac, d, w, offs, 1, 256, 512, VARYING_FIRST_DIM, rng, offs[-1:])
    a = triton_group_row_rht_col_rht_quantize_ms_eden(*args)
    b = triton_group_row_rht_col_rht_quantize_ms_eden(*args)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@_needs_ms_eden
@torch.no_grad()
def test_ms_eden_is_unbiased():
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
    ar, ac = _amax("triton", dy, d, w, offs, 1)
    row_ref = reference_ms_eden(reference_dynamic_rht(dy, d, transpose=False), ar[0])
    col_ref = reference_ms_eden(reference_dynamic_rht(dy, w, transpose=True), ac[0])

    draws = 64
    row_scales, col_scales, dequants = [], [], []
    for i in range(draws):
        rng = torch.tensor([1, i, 2, i + 1000], dtype=torch.int64, device="cuda")
        row_codes, row_sf, _, col_sf = triton_group_row_rht_col_rht_quantize_ms_eden(
            dy, ar, ac, d, w, offs, 1, M, N, VARYING_FIRST_DIM, rng, offs[-1:]
        )
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
@torch.no_grad()
def test_codes_are_rtne_from_the_pre_correction_scale():
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
    ar, ac = _amax("triton", dy, d, w, offs, 1)
    row_ref = reference_ms_eden(reference_dynamic_rht(dy, d, transpose=False), ar[0])
    col_ref = reference_ms_eden(reference_dynamic_rht(dy, w, transpose=True), ac[0])

    for seed in (0, 1, 17, 29):
        rng = torch.tensor([seed, 0, seed + 7, 0], dtype=torch.int64, device="cuda")
        row_codes, _, col_codes, _ = triton_group_row_rht_col_rht_quantize_ms_eden(
            dy, ar, ac, d, w, offs, 1, M, N, VARYING_FIRST_DIM, rng, offs[-1:]
        )
        assert_codes_bitwise(row_codes, row_ref.codes, f"rowwise codes @ seed {seed}")
        assert_codes_bitwise(col_codes, col_ref.codes, f"colwise codes @ seed {seed}")


@_needs_ms_eden
@torch.no_grad()
def test_each_rng_slice_drives_exactly_one_output():
    """``rng_state`` is ``[col_seed, col_offset, row_seed, row_offset]``.

    Both axes draw from one tensor, so a kernel that hands an axis the wrong slice
    raises nothing -- it correlates the two operands' scales, which no shape or dtype
    check catches. Same failure mode as the crossed sign vectors in §11.2, and the
    same reason to test it explicitly.
    """
    M, N = 256, 512
    dy, offs = _packed([M], N)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax("triton", dy, d, w, offs, 1)

    def run(rng):
        return triton_group_row_rht_col_rht_quantize_ms_eden(
            dy,
            ar,
            ac,
            d,
            w,
            offs,
            1,
            M,
            N,
            VARYING_FIRST_DIM,
            torch.tensor(rng, dtype=torch.int64, device="cuda"),
            offs[-1:],
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
@pytest.mark.parametrize("group_sizes", [[256], [128, 128], [256, 128, 384]])
@torch.no_grad()
def test_multi_group_matches_the_reference(group_sizes):
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
    ar, ac = _amax("triton", dy, d, w, offs, E)
    rng = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    row_codes, row_sf, col_codes, col_sf = (
        triton_group_row_rht_col_rht_quantize_ms_eden(
            dy, ar, ac, d, w, offs, E, M, N, VARYING_FIRST_DIM, rng, offs[-1:]
        )
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
    M, N, E = 512, 256, 2
    with FakeTensorMode():
        dy = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        sv = torch.empty(128, dtype=torch.int8, device="cuda")
        offs = torch.empty(E, dtype=torch.int32, device="cuda")
        amax = torch.empty(E, dtype=torch.float32, device="cuda")
        rng = torch.empty(4, dtype=torch.int64, device="cuda")
        row, col = op(dy, sv, sv, offs, E, M, N, VARYING_FIRST_DIM, None)
        assert row.shape == (E,) and col.shape == (E,)
        out = triton_group_row_rht_col_rht_quantize_ms_eden(
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
@torch.no_grad()
def test_ms_eden_always_requires_an_rng_state():
    """Unlike the cast quantizers there is no deterministic mode to fall back to."""
    M, N, E = 512, 256, 2
    dy = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    offs = torch.tensor([256, 512], dtype=torch.int32, device="cuda")
    sv = torch.ones(128, dtype=torch.int8, device="cuda")
    amax = torch.ones(E, dtype=torch.float32, device="cuda")
    with pytest.raises(TypeError, match="rng_state must be a torch.Tensor"):
        triton_group_row_rht_col_rht_quantize_ms_eden(
            dy, amax, amax, sv, sv, offs, E, M, N, VARYING_FIRST_DIM, None, offs[-1:]
        )
