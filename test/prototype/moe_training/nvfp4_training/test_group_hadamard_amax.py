"""Tests for the grouped RHT amax kernels (SM100+), triton and cutedsl.

These tests compare the grouped kernels against per-expert calls to the non-grouped
triton_rht_amax and verify the grouped custom ops' fake output shapes and shared
input validation.

Semantics match the non-grouped triton_rht_amax (single sign vector):
  col_amax[g] = max|RHT(A_g.T)|, row_amax[g] = max|A_g|.

Both backends round the RHT output to bfloat16 before reducing -- triton the tl.dot
accumulator, cutedsl the tcgen05 one -- so both match the bf16-rounded oracle, and each
other, bitwise. The plain rowwise amax is exact for both.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.utils._triton import has_triton

from benchmarks.prototype.nvfp4_training.deepseek_v3_shapes import (
    get_deepseek_v3_weight_shapes,
)
from test.prototype.moe_training.nvfp4_training.nvfp4_reference import (
    reference_group_rht_amax,
    reference_group_row_cast_col_rht_amax,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    DEFAULT_SIGN_VECTOR,
)
from torchao.utils import is_sm_at_least_100, torch_version_at_least

if has_triton() and is_sm_at_least_100() and torch_version_at_least("2.10.0"):
    from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_triton import (
        triton_group_rht_amax,
    )
    from torchao.prototype.moe_training.nvfp4_training.hadamard_amax_triton import (
        triton_rht_amax,
    )
if cutedsl_nvfp4_kernels_available():
    from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_cutedsl import (
        cutedsl_group_rht_amax,
    )

_HARDCODED_SIGN_VECTOR = DEFAULT_SIGN_VECTOR

requires_sm100 = [
    pytest.mark.skipif(not has_triton(), reason="unsupported without triton"),
    pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+"),
    pytest.mark.skipif(
        not torch_version_at_least("2.10.0"),
        reason="requires PyTorch 2.10+",
    ),
]

_skip_no_cutedsl = pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(),
    reason="requires SM100 (Blackwell) + CuteDSL runtime (cuda-python, nvidia-cutlass-dsl)",
)

_KERNELS = [
    pytest.param("triton", id="triton"),
    pytest.param("cutedsl", marks=_skip_no_cutedsl, id="cutedsl"),
]


def _maybe_sm100(fn):
    for mark in requires_sm100:
        fn = mark(fn)
    return fn


def _group_rht_amax(
    kernel, A, sign_vector, offsets, num_tensors, psl, hidden, rep, **kwargs
):
    """Dispatch to a backend's grouped RHT amax op; returns ``(col_amax, row_amax)``."""
    op = triton_group_rht_amax if kernel == "triton" else cutedsl_group_rht_amax
    return op(A, list(sign_vector), offsets, num_tensors, psl, hidden, rep, **kwargs)


def _skip_if_unsupported_groups(kernel: str, num_tensors: int) -> None:
    """The cutedsl group lookup is a fixed-depth binary search capped at 64 groups."""
    if kernel == "cutedsl" and num_tensors > 64:
        pytest.skip("cutedsl grouped kernel supports at most 64 groups")


def _build_packed(groups, hidden_size, device, seed):
    """Return (A, offsets, group_tensors) for row-concatenated expert groups."""
    torch.manual_seed(seed)
    group_tensors = [
        torch.randn((m, hidden_size), dtype=torch.bfloat16, device=device)
        for m in groups
    ]
    A = torch.cat(group_tensors, dim=0)
    offsets = torch.cumsum(
        torch.tensor(groups, dtype=torch.int32, device=device),
        dim=0,
        dtype=torch.int32,
    )
    return A, offsets, group_tensors


def _group_rht_amax_reference(A, offsets, num_tensors, sign_vector):
    """Grouped oracle: per-expert single-GPU triton_rht_amax over packed rows."""
    col = A.new_empty((num_tensors,), dtype=torch.float32)
    row = A.new_empty((num_tensors,), dtype=torch.float32)
    for g in range(num_tensors):
        row_start = 0 if g == 0 else offsets[g - 1]
        row_end = offsets[g]
        Ag = A[row_start:row_end]
        c, r = triton_rht_amax(Ag, list(sign_vector))
        col[g] = c
        row[g] = r
    return col, row


@_maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_group_rht_amax_matches_per_group_kernel(kernel):
    """Grouped outputs match per-expert triton_rht_amax outputs."""
    device = torch.device("cuda", 0)
    groups = (128, 256)
    hidden_size = 256
    A, offsets, group_tensors = _build_packed(groups, hidden_size, device, seed=223)

    expected_col, expected_row = _group_rht_amax_reference(
        A, offsets, len(groups), _HARDCODED_SIGN_VECTOR
    )
    torch_col, torch_row = reference_group_rht_amax(
        A, offsets, len(groups), _HARDCODED_SIGN_VECTOR
    )

    actual_col, actual_row = _group_rht_amax(
        kernel,
        A,
        _HARDCODED_SIGN_VECTOR,
        offsets,
        len(groups),
        A.shape[0],
        hidden_size,
        1,
    )

    torch.testing.assert_close(actual_col, expected_col, atol=0, rtol=0)
    assert torch.equal(actual_row, expected_row)
    torch.testing.assert_close(actual_col, torch_col, atol=0, rtol=0)
    torch.testing.assert_close(actual_row, torch_row, atol=0, rtol=0)


@_maybe_sm100
@torch.no_grad()
def test_group_rht_amax_persistent_path_bitwise():
    """Large ragged VARYING_FIRST_DIM groups take the per-group-CTA persistent fast
    path (avg rows/group >= threshold); outputs must still match the per-expert
    kernel bitwise. The small-group test above stays on the tiled kernel."""
    device = torch.device("cuda", 0)
    groups = (2048, 1024, 4096, 1152)  # 128-aligned, avg >> 1024 -> persistent
    hidden_size = 2048
    A, offsets, _ = _build_packed(groups, hidden_size, device, seed=91)

    expected_col, expected_row = _group_rht_amax_reference(
        A, offsets, len(groups), _HARDCODED_SIGN_VECTOR
    )

    actual_col, actual_row = triton_group_rht_amax(
        A,
        list(_HARDCODED_SIGN_VECTOR),
        offsets,
        len(groups),
        A.shape[0],
        hidden_size,
        1,
    )

    assert torch.equal(actual_col, expected_col)
    assert torch.equal(actual_row, expected_row)


@_maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize(
    "shape",
    get_deepseek_v3_weight_shapes(factorized_experts=2),
    ids=lambda shape: f"{shape.model}-{shape.projection}",
)
@torch.no_grad()
def test_group_rht_amax_deepseek_dimensions(kernel, shape):
    """Real TorchTitan M/N dimensions with E factorized to two experts."""
    _skip_if_unsupported_groups(kernel, shape.experts)
    device = torch.device("cuda", 0)
    groups = (shape.m,) * shape.experts
    A, offsets, _ = _build_packed(groups, shape.n, device, seed=223)
    expected_col, expected_row = _group_rht_amax_reference(
        A, offsets, shape.experts, _HARDCODED_SIGN_VECTOR
    )

    actual_col, actual_row = _group_rht_amax(
        kernel,
        A,
        _HARDCODED_SIGN_VECTOR,
        offsets,
        shape.experts,
        A.shape[0],
        shape.n,
        0,
    )

    torch.testing.assert_close(actual_col, expected_col, atol=0, rtol=0)
    assert torch.equal(actual_row, expected_row)


@_maybe_sm100
@_skip_no_cutedsl
@torch.no_grad()
def test_cutedsl_group_rht_amax_matches_triton():
    """Ragged 128-aligned groups: both amaxes are bitwise equal across backends."""
    device = torch.device("cuda", 0)
    groups = (128, 256, 384, 128)
    hidden_size = 1024
    A, offsets, _ = _build_packed(groups, hidden_size, device, seed=225)

    args = (A, _HARDCODED_SIGN_VECTOR, offsets, len(groups), A.shape[0], hidden_size, 1)
    triton_col, triton_row = _group_rht_amax("triton", *args)
    cutedsl_col, cutedsl_row = _group_rht_amax("cutedsl", *args)

    assert torch.equal(cutedsl_row, triton_row)
    assert torch.equal(cutedsl_col, triton_col)


@_maybe_sm100
@_skip_no_cutedsl
@torch.no_grad()
def test_cutedsl_group_rht_amax_rejects_too_many_groups():
    """The fixed-depth group search caps the group count; exceeding it must raise."""
    device = torch.device("cuda", 0)
    groups = (128,) * 65
    A, offsets, _ = _build_packed(groups, 128, device, seed=11)
    with pytest.raises(ValueError, match="num_tensors must be <= 64"):
        _group_rht_amax(
            "cutedsl",
            A,
            _HARDCODED_SIGN_VECTOR,
            offsets,
            len(groups),
            A.shape[0],
            128,
            1,
        )


@_maybe_sm100
@torch.no_grad()
def test_group_rht_amax_register_fake_shapes():
    """register_fake yields (num_tensors,) float32 outputs under fake mode."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    num_tensors = 3
    with FakeTensorMode():
        A = torch.empty((512, 256), dtype=torch.bfloat16, device="cuda")
        offsets = torch.empty((num_tensors,), dtype=torch.int32, device="cuda")
        col, row = triton_group_rht_amax(
            A,
            list(_HARDCODED_SIGN_VECTOR),
            offsets,
            num_tensors,
            512,
            256,
            1,
        )
    assert col.shape == (num_tensors,)
    assert row.shape == (num_tensors,)
    assert col.dtype == torch.float32
    assert row.dtype == torch.float32
    # The dynamic-RHT kwargs reach both ops' fake kernels.
    with FakeTensorMode():
        A = torch.empty((512, 256), dtype=torch.bfloat16, device="cuda")
        offsets = torch.empty((num_tensors,), dtype=torch.int32, device="cuda")
        signs = torch.empty((128,), dtype=torch.int8, device="cuda")
        ops = [triton_group_rht_amax]
        if cutedsl_nvfp4_kernels_available():
            ops.append(cutedsl_group_rht_amax)
        for op in ops:
            col, row = op(
                A,
                [],
                offsets,
                num_tensors,
                512,
                256,
                1,
                sign_tensor=signs,
                dynamic_rht=True,
            )
            assert col.shape == (num_tensors,)
            assert row.shape == (num_tensors,)


@_maybe_sm100
@torch.no_grad()
def test_group_rht_amax_validates_packed_shape():
    """The grouped amax wrapper applies shared packed-shape validation."""
    device = torch.device("cuda", 0)
    groups = (128,)
    A, offsets, _ = _build_packed(groups, 256, device, seed=7)
    with pytest.raises(ValueError, match="packed_sequence_length must match"):
        triton_group_rht_amax(
            A,
            list(_HARDCODED_SIGN_VECTOR),
            offsets,
            len(groups),
            256,
            256,
            1,
        )


@_maybe_sm100
@torch.no_grad()
def test_group_rht_amax_rejects_non_tensorwise_scaling():
    device = torch.device("cuda", 0)
    groups = (128,)
    A, offsets, _ = _build_packed(groups, 256, device, seed=7)
    with pytest.raises(ValueError, match="only ScalingType.TensorWise"):
        triton_group_rht_amax(
            A,
            list(_HARDCODED_SIGN_VECTOR),
            offsets,
            len(groups),
            A.shape[0],
            A.shape[1],
            1,
            int(F.ScalingType.RowWise),
        )


# --- dynamic_rht: RHT-128 with a resampled sign buffer ----------------------
#
# Same op, same kernel; only the RHT matrix source differs. These cover what the
# static path cannot: a 128-wide transform, and a sign vector that must not be
# memoized by value.


def _dynamic_signs(seed, n=128):
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (n,), generator=generator, dtype=torch.int8)
    return (bits * 2 - 1).cuda()


@_maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@pytest.mark.parametrize("groups", [(256,), (128, 128), (256, 128, 384, 128)])
@torch.no_grad()
def test_group_rht_amax_dynamic_matches_the_reference(kernel, groups):
    device = torch.device("cuda", 0)
    hidden_size = 512
    A, offsets, _ = _build_packed(groups, hidden_size, device, seed=223)
    signs = _dynamic_signs(0)

    col, row = _group_rht_amax(
        kernel,
        A,
        [],
        offsets,
        len(groups),
        A.shape[0],
        hidden_size,
        1,
        sign_tensor=signs,
        dynamic_rht=True,
    )
    ref_col, ref_row = reference_group_row_cast_col_rht_amax(
        A, signs, offsets, len(groups)
    )
    torch.testing.assert_close(col, ref_col, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(row, ref_row, atol=0, rtol=0)


@_maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_group_rht_amax_dynamic_follows_an_in_place_resample(kernel):
    """The failure mode dynamic_rht exists to prevent.

    The cadence manager updates the sign buffer with ``copy_`` so its address survives
    graph capture -- which also means its ``id()`` never changes. A path that memoized
    on the buffer would keep returning the matrix built from the first contents, and
    the transform would stop cancelling with no error anywhere.
    """
    device = torch.device("cuda", 0)
    groups = (256,)
    A, offsets, _ = _build_packed(groups, 512, device, seed=11)
    signs = _dynamic_signs(0)

    def run():
        return _group_rht_amax(
            kernel,
            A,
            [],
            offsets,
            1,
            A.shape[0],
            512,
            1,
            sign_tensor=signs,
            dynamic_rht=True,
        )

    before_col, before_row = run()
    signs.copy_(_dynamic_signs(1))
    after_col, after_row = run()

    assert before_col.item() != after_col.item(), "columnwise amax ignored the resample"
    assert before_row.item() == after_row.item(), "rowwise amax is not transformed"


@_maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_group_rht_amax_dynamic_excludes_padded_rows(kernel):
    """Rows past logical_packed_length are uninitialized capacity, never data."""
    device = torch.device("cuda", 0)
    groups = (128, 128)
    A, offsets, _ = _build_packed(groups, 512, device, seed=3)
    capacity = torch.full((512, 512), float("inf"), device=device, dtype=torch.bfloat16)
    capacity[:256] = A

    got = _group_rht_amax(
        kernel,
        capacity,
        [],
        offsets,
        2,
        512,
        512,
        1,
        logical_packed_length=offsets[-1:],
        sign_tensor=_dynamic_signs(0),
        dynamic_rht=True,
    )
    assert torch.isfinite(torch.stack(got)).all(), "spare capacity leaked into an amax"


@_maybe_sm100
@pytest.mark.parametrize("kernel", _KERNELS)
@torch.no_grad()
def test_group_rht_amax_rejects_an_inconsistent_sign_pair(kernel):
    device = torch.device("cuda", 0)
    A, offsets, _ = _build_packed((128,), 256, device, seed=7)
    args = (kernel, A, list(_HARDCODED_SIGN_VECTOR), offsets, 1, A.shape[0], 256, 1)
    with pytest.raises(ValueError, match="dynamic_rht=True requires a sign_tensor"):
        _group_rht_amax(*args, dynamic_rht=True)
    with pytest.raises(ValueError, match="only used when dynamic_rht=True"):
        _group_rht_amax(*args, sign_tensor=_dynamic_signs(0))


# (groups, hidden_size, spare capacity rows) -- the spare rows hold NaN and must never be
# read; ``max_groups`` is the cutedsl group cap, ``hidden_7168`` gives each CTA several tiles.
_DYNAMIC_BITWISE_CASES = [
    pytest.param((128, 256, 384, 128), 1024, 0, id="ragged"),
    pytest.param((256,), 512, 0, id="single"),
    pytest.param((128,) * 64, 128, 0, id="max_groups"),
    pytest.param((256, 0, 384), 512, 0, id="empty_group"),
    pytest.param((128, 128), 512, 256, id="capacity_tail"),
    pytest.param((3072, 0, 1024, 2048, 128, 896), 7168, 512, id="hidden_7168"),
]


@_maybe_sm100
@_skip_no_cutedsl
@pytest.mark.parametrize("sign_dtype", [torch.int8, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("groups, hidden_size, spare_rows", _DYNAMIC_BITWISE_CASES)
@torch.no_grad()
def test_cutedsl_group_rht_amax_dynamic_matches_triton(
    groups, hidden_size, spare_rows, sign_dtype
):
    """Both amaxes are bitwise equal to the Triton op's on the RHT-128 dynamic path."""
    device = torch.device("cuda", 0)
    A, offsets, _ = _build_packed(groups, hidden_size, device, seed=31)
    capacity = torch.full(
        (A.shape[0] + spare_rows, hidden_size),
        float("nan"),
        device=device,
        dtype=torch.bfloat16,
    )
    capacity[: A.shape[0]] = A
    signs = _dynamic_signs(0).to(sign_dtype)
    args = (capacity, [], offsets, len(groups), capacity.shape[0], hidden_size, 1)
    kwargs = dict(
        logical_packed_length=offsets[-1:], sign_tensor=signs, dynamic_rht=True
    )

    expected = _group_rht_amax("triton", *args, **kwargs)
    got = _group_rht_amax("cutedsl", *args, **kwargs)
    for name, e, g in zip(("col_amax", "row_amax"), expected, got):
        assert torch.equal(e, g), f"{name} differs from the Triton op"


@_maybe_sm100
@_skip_no_cutedsl
@torch.no_grad()
def test_cutedsl_group_rht_amax_dynamic_rejects_a_16_sign_tensor():
    """Only the (128,) sign buffer is served on the dynamic path (Triton also takes (16,))."""
    device = torch.device("cuda", 0)
    A, offsets, _ = _build_packed((128,), 256, device, seed=7)
    with pytest.raises(ValueError, match=r"sign_tensor must be a \(128,\) tensor"):
        cutedsl_group_rht_amax(
            A,
            [],
            offsets,
            1,
            A.shape[0],
            256,
            1,
            sign_tensor=_dynamic_signs(0, n=16),
            dynamic_rht=True,
        )
