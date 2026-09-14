# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Dense NVFP4 linear under the V2 and V1_REQUANT recipes. Design doc §15 and §16.

The load-bearing structural claim of these recipes is that a dense linear *is* the
degenerate one-group case of the grouped kernels. Several tests below exist to hold
that claim rather than to check numerics: that no non-grouped kernel is reachable,
and that the linear path and the grouped path agree at ``E = 1``.
"""


import pytest
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

from torchao.prototype.moe_training.nvfp4_training import (
    nvfp4_linear as nvfp4_linear_mod,
)
from torchao.prototype.moe_training.nvfp4_training import nvfp4_linear_v2 as v2_mod
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
    cutedsl_prepare_for_cuda_graph,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_recipe import NVFP4Recipe
from torchao.prototype.moe_training.nvfp4_training.nvfp4_rht_cadence import (
    resample_nvfp4_rht_signs,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
    NVFP4Linear,
    NVFP4TrainingConfig,
)
from torchao.quantization import quantize_
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.quantization.utils import compute_error

from ._v2_marks import kernel_gate, kernel_skip, maybe_sm100

# Every grouped kernel these recipes call must be implemented before the numerics
# tests mean anything. V1_REQUANT needs only the three no-RHT ops; V2 needs all of them.
_V1_REQUANT_KERNELS_IMPLEMENTED = True
_V2_KERNELS_IMPLEMENTED = True
_needs_v1_requant = kernel_gate(
    _V1_REQUANT_KERNELS_IMPLEMENTED, "the §11.1/§11.6/§11.7 kernels"
)
_needs_v2 = kernel_gate(_V2_KERNELS_IMPLEMENTED, "the §11.1-§11.9 kernels")

# For tests that cover both recipes. Gating the whole test on V2 would keep the
# V1_REQUANT half unreachable through all of Phase A, which is exactly the half that
# has to hold before a V1_REQUANT convergence run.
_BOTH_RECIPES = [
    pytest.param(
        NVFP4Recipe.V1_REQUANT,
        marks=kernel_skip(
            _V1_REQUANT_KERNELS_IMPLEMENTED, "the §11.1/§11.6/§11.7 kernels"
        ),
        id="v1_requant",
    ),
    pytest.param(
        NVFP4Recipe.V2,
        marks=kernel_skip(_V2_KERNELS_IMPLEMENTED, "the §11.1-§11.9 kernels"),
        id="v2",
    ),
]

_KERNEL_PREFERENCES = [
    pytest.param(KernelPreference.AUTO, id="auto"),
    pytest.param(KernelPreference.TRITON, id="triton"),
    pytest.param(
        KernelPreference.CUTEDSL,
        marks=pytest.mark.skipif(
            not cutedsl_nvfp4_kernels_available(),
            reason="requires the CuteDSL runtime",
        ),
        id="cutedsl",
    ),
]

# For tests that run BOTH backends in one body.
_requires_cutedsl = pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(), reason="requires the CuteDSL runtime"
)

_M = _K = _N = 256

# Ops that belong to recipe V1 only. If any of these is dispatched while a
# V1_REQUANT or V2 layer runs, the "linear is grouped at num_tensors=1" design has
# been quietly abandoned somewhere.
_LINEAR_ONLY_OPS = {
    "torchao::triton_rht_amax",
    "torchao::triton_rht_quantize_row_col",
    "torchao::triton_weight_quantize_2d",
    "torchao::cutedsl_rht_amax",
    "torchao::cutedsl_rht_quantize_row_col",
    "torchao::cutedsl_weight_quantize_2d",
}

# Every quantize and amax op each recipe dispatches, without its backend prefix.
_KERNELS = {
    NVFP4Recipe.V2: {
        "group_rht_amax",
        "group_rht_quantize_row_col",
        "group_row_cast_quantize",
        "group_row_rht_col_rht_amax",
        "group_row_rht_col_rht_quantize_ms_eden",
        "group_col_rht_requant_amax",
        "group_col_rht_requantize",
        "group_weight_amax",
    },
    NVFP4Recipe.V1_REQUANT: {
        "group_rht_amax",
        "group_rht_quantize_row_col",
        "group_row_cast_quantize",
        "group_col_cast_requant_amax",
        "group_col_cast_requantize",
        "group_weight_amax",
    },
}


def _expected_ops(recipe, kernel_preference):
    """The exact ``torchao::`` op set a recipe dispatches on the resolved backend.

    The weight amax has no CuteDSL twin and is Triton on every path.
    """
    if kernel_preference is KernelPreference.AUTO:
        kernel_preference = (
            KernelPreference.CUTEDSL
            if cutedsl_nvfp4_kernels_available()
            else KernelPreference.TRITON
        )
    if kernel_preference is KernelPreference.TRITON:
        return {"torchao::triton_" + k for k in _KERNELS[recipe]}
    return {"torchao::triton_group_weight_amax"} | {
        "torchao::cutedsl_" + k for k in _KERNELS[recipe] - {"group_weight_amax"}
    }


class _RecordOps(TorchDispatchMode):
    """Record every ``torchao::`` op dispatched inside the block."""

    def __init__(self):
        self.names = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.name() if hasattr(func, "name") else str(func)
        if name.startswith("torchao::"):
            self.names.add(name.split(".")[0])
        return func(*args, **(kwargs or {}))


def _layer(recipe, *, seed=0, bias=False, kernel_preference=KernelPreference.AUTO):
    torch.manual_seed(seed)
    return NVFP4Linear(
        _K,
        _N,
        bias=bias,
        kernel_preference=kernel_preference,
        device="cuda",
        dtype=torch.bfloat16,
        recipe=recipe,
    )


def _inputs(seed=0):
    torch.manual_seed(seed)
    x = torch.randn(_M, _K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    return x


# ---------------------------------------------------------------------------
# Structural tests -- these run today
# ---------------------------------------------------------------------------


def test_the_v2_module_imports_no_linear_only_kernel():
    """Static half of the "no non-grouped op is reachable" claim.

    Cheap, runs without a GPU, and catches the mistake at review time rather than at
    dispatch time.
    """
    imported = {
        name
        for name in vars(v2_mod)
        if name.startswith(
            ("triton_rht", "triton_weight", "cutedsl_rht", "cutedsl_weight")
        )
    }
    assert imported == set(), f"linear-only kernels imported: {sorted(imported)}"


def test_every_quantize_op_it_imports_is_grouped():
    """The positive form: everything it calls is a ``*_group_*`` op."""
    quantizers = {
        name
        for name in vars(v2_mod)
        if name.startswith(("triton_", "cutedsl_"))
        and not name.startswith(("triton_group", "cutedsl_group"))
    }
    assert quantizers == set(), f"non-grouped kernels imported: {sorted(quantizers)}"


@maybe_sm100
@pytest.mark.parametrize("kernel_preference", _KERNEL_PREFERENCES)
@pytest.mark.parametrize("recipe", [NVFP4Recipe.V1_REQUANT, NVFP4Recipe.V2])
def test_torch_compile_survives_a_prior_eager_forward(recipe, kernel_preference):
    """The regression the per-device offsets buffer caused.

    That buffer held a real one-element tensor and filled it in place every forward
    and backward, for a stable address across CUDA-graph replays. Dynamo traces under
    a FakeTensorMode in every compile mode, so once anything had allocated it --
    an eager forward, or prepare_for_cuda_graph, which prewarmed it even for a
    V1-only setup that never read it -- filling it while tracing raised, and
    torch.compile was dead for the rest of the process.

    Eager first, then compile, is the order that failed; compiling from a cold
    process always worked, which is why it survived the existing tests.
    """
    # NVFP4Linear.forward is one code object for every recipe and backend, so each
    # parameter here is a recompile of it against Dynamo's per-process limit of 8,
    # which test_nvfp4_linear.py's compile tests share when the files run together.
    torch._dynamo.reset()
    layer = _layer(recipe, kernel_preference=kernel_preference)
    x = torch.randn(_M, _K, device="cuda", dtype=torch.bfloat16)
    layer(x).sum().backward()
    layer.weight.grad = None

    out = torch.compile(layer)(x)
    assert out.shape == (_M, _N)
    assert torch.isfinite(out).all()


@maybe_sm100
@pytest.mark.parametrize("recipe", [NVFP4Recipe.V1_REQUANT, NVFP4Recipe.V2])
def test_shape_constraints_are_enforced(recipe):
    layer = _layer(recipe)
    with pytest.raises(ValueError, match="divisible by 128"):
        layer(torch.randn(100, _K, device="cuda", dtype=torch.bfloat16))


@maybe_sm100
@pytest.mark.parametrize("recipe", list(NVFP4Recipe))
def test_buffers_match_the_recipe(recipe):
    """V2 is the only recipe that draws a second sign vector, because it is the only
    one that rotates the dgrad axis."""
    layer = _layer(recipe)
    names = {n for n, _ in layer.named_buffers()}
    assert "_rht_sign_vector" in names and "_sr_seed" in names
    if recipe is NVFP4Recipe.V2:
        assert "_dgrad_rht_sign_vector" in names
        assert layer._rht_sign_vector.numel() == 128
    else:
        assert "_dgrad_rht_sign_vector" not in names
        assert layer._rht_sign_vector.numel() == 16


@maybe_sm100
def test_v1_default_is_unchanged():
    """An unchanged config must still build exactly the V1 layer it always did."""
    model = nn.Sequential(nn.Linear(_K, _N, bias=False)).cuda().bfloat16()
    quantize_(model, NVFP4TrainingConfig())
    assert model[0].recipe is NVFP4Recipe.V1
    assert model[0]._rht_sign_vector.numel() == 16
    assert not hasattr(model[0], "_dgrad_rht_sign_vector")


@maybe_sm100
def test_v2_sign_buffers_resample_in_place():
    """§15's "same tensor identity, updated in place" requirement.

    The addresses must survive resampling or a CUDA graph captured around the step
    would replay against freed memory.
    """
    model = nn.Sequential(nn.Linear(_K, _N, bias=False)).cuda().bfloat16()
    quantize_(model, NVFP4TrainingConfig(recipe=NVFP4Recipe.V2))
    layer = model[0]
    w_ptr = layer._rht_sign_vector.data_ptr()
    d_ptr = layer._dgrad_rht_sign_vector.data_ptr()

    updated = resample_nvfp4_rht_signs(model, seed=1, step=0, microbatch=0)
    assert updated == 2
    first_w = layer._rht_sign_vector.clone()
    first_d = layer._dgrad_rht_sign_vector.clone()

    resample_nvfp4_rht_signs(model, seed=1, step=0, microbatch=1)
    assert not torch.equal(first_w, layer._rht_sign_vector), (
        "wgrad resamples per microbatch"
    )
    assert torch.equal(first_d, layer._dgrad_rht_sign_vector), (
        "dgrad holds within a step"
    )

    resample_nvfp4_rht_signs(model, seed=1, step=1, microbatch=0)
    assert not torch.equal(first_d, layer._dgrad_rht_sign_vector), (
        "dgrad resamples per step"
    )

    assert layer._rht_sign_vector.data_ptr() == w_ptr
    assert layer._dgrad_rht_sign_vector.data_ptr() == d_ptr


@maybe_sm100
def test_static_recipes_are_never_resampled():
    """V1 and V1_REQUANT signs are fixed for the run; touching them would leak one
    cached RHT matrix per step through the value-keyed cache."""
    model = nn.Sequential(nn.Linear(_K, _N, bias=False)).cuda().bfloat16()
    quantize_(model, NVFP4TrainingConfig(recipe=NVFP4Recipe.V1_REQUANT))
    before = model[0]._rht_sign_vector.clone()
    assert resample_nvfp4_rht_signs(model, seed=1, step=5, microbatch=3) == 0
    assert torch.equal(before, model[0]._rht_sign_vector)


# ---------------------------------------------------------------------------
# Numerics -- gated on the kernel bodies
# ---------------------------------------------------------------------------


@_needs_v1_requant
@torch.no_grad()
def test_v1_requant_forward_vs_bf16():
    layer = _layer(NVFP4Recipe.V1_REQUANT)
    x = _inputs()
    got = layer(x)
    want = x @ layer.weight.t()
    assert compute_error(want.float(), got.float()) > 15.0


@_needs_v2
@torch.no_grad()
def test_v2_forward_vs_bf16():
    layer = _layer(NVFP4Recipe.V2)
    x = _inputs()
    got = layer(x)
    want = x @ layer.weight.t()
    assert compute_error(want.float(), got.float()) > 15.0


@maybe_sm100
@pytest.mark.parametrize("kernel_preference", _KERNEL_PREFERENCES)
@pytest.mark.parametrize("recipe", _BOTH_RECIPES)
def test_gradients_vs_bf16_autograd(recipe, kernel_preference):
    """SQNR against bf16 autograd on each backend. This bound, not ``torch.equal``,
    is the only cross-backend statement possible for the V1_REQUANT backward: its
    stochastic-rounding cast of ``dy`` draws a different Philox stream on CuteDSL."""
    layer = _layer(recipe, kernel_preference=kernel_preference)
    x = _inputs()
    layer(x).sum().backward()

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = layer.weight.detach().clone().requires_grad_(True)
    (x_ref @ w_ref.t()).sum().backward()

    assert compute_error(x_ref.grad.float(), x.grad.float()) > 10.0
    assert compute_error(w_ref.grad.float(), layer.weight.grad.float()) > 10.0


@maybe_sm100
@pytest.mark.parametrize("recipe", _BOTH_RECIPES)
def test_no_non_grouped_kernel_is_dispatched(recipe):
    """The runtime half of the "linear is grouped at num_tensors=1" claim.

    Every quantize and amax op that runs must be a grouped one. This is the test that
    fails if someone later "optimizes" the dense path onto a dedicated linear kernel
    without also keeping the two numerics in agreement.
    """
    layer = _layer(recipe)
    x = _inputs()
    with _RecordOps() as recorder:
        layer(x).sum().backward()
    assert recorder.names, "no torchao op was recorded; the probe is not working"
    leaked = recorder.names & _LINEAR_ONLY_OPS
    assert not leaked, f"non-grouped kernels dispatched: {sorted(leaked)}"


@maybe_sm100
@pytest.mark.parametrize("recipe", _BOTH_RECIPES)
@torch.no_grad()
def test_linear_matches_the_grouped_path_at_one_group(recipe):
    """The converse: driving the grouped entrypoint with ``offs = [M]`` must give the
    same answer as the dense one, bitwise."""
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm_v2 import (
        nvfp4_v1_requant_grouped_mm,
        nvfp4_v2_grouped_mm,
    )

    layer = _layer(recipe)
    x = _inputs()
    dense = layer(x)
    offs = torch.tensor([_M], dtype=torch.int32, device="cuda")
    w3d = layer.weight.detach().unsqueeze(0)
    if recipe is NVFP4Recipe.V2:
        grouped = nvfp4_v2_grouped_mm(
            x.detach(),
            w3d,
            wgrad_rht=layer._rht_sign_vector,
            dgrad_rht=layer._dgrad_rht_sign_vector,
            sr_seed=layer._sr_seed,
            offs=offs,
        )
    else:
        grouped = nvfp4_v1_requant_grouped_mm(
            x.detach(),
            w3d,
            sign_vector=layer.rht_sign_vector,
            sr_seed=layer._sr_seed,
            offs=offs,
        )
    torch.testing.assert_close(dense, grouped, atol=0, rtol=0)


@_needs_v2
def test_saved_tensors_hold_no_bf16_activation_or_weight_transpose():
    """§15 invariant 7: forward saves packed codes and scales only.

    Checked by size: anything the size of a bf16 activation or weight would show up
    immediately, since FP4 codes are a quarter of bf16 and the scales a sixteenth.
    """
    layer = _layer(NVFP4Recipe.V2)
    x = _inputs()
    out = layer(x)
    saved = out.grad_fn.saved_tensors
    for t in saved:
        assert t.dtype in (
            torch.uint8,
            torch.float8_e4m3fn,
            torch.float32,
            torch.int8,
            torch.int64,
        ), f"unexpected saved dtype {t.dtype}"
    total = sum(t.numel() * t.element_size() for t in saved)
    bf16_activation = x.numel() * 2
    bf16_weight = layer.weight.numel() * 2
    assert total < bf16_activation + bf16_weight, (
        f"saved {total} bytes; a bf16 activation plus weight would be "
        f"{bf16_activation + bf16_weight}"
    )


@_needs_v2
def test_v2_backward_is_reproducible_for_a_fixed_rng_state(monkeypatch):
    """MS-EDEN draws fresh offsets per backward, so reproducibility has to be pinned
    by fixing the state the op receives."""
    fixed = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    monkeypatch.setattr(v2_mod, "_backward_rng_state", lambda sr_seed: fixed)

    grads = []
    for _ in range(2):
        layer = _layer(NVFP4Recipe.V2)
        x = _inputs()
        layer(x).sum().backward()
        grads.append((x.grad.clone(), layer.weight.grad.clone()))
    assert torch.equal(grads[0][0], grads[1][0])
    assert torch.equal(grads[0][1], grads[1][1])


@_needs_v2
@_requires_cutedsl
@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
def test_v2_backends_are_bitwise_for_a_fixed_rng_state(bias, monkeypatch):
    """A V2 step on CuteDSL reproduces the Triton step bitwise: every V2 op has a
    bitwise twin, MS-EDEN included, once the Philox state it receives is pinned."""
    fixed = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    monkeypatch.setattr(v2_mod, "_backward_rng_state", lambda sr_seed: fixed)

    results = []
    for kernel_preference in (KernelPreference.TRITON, KernelPreference.CUTEDSL):
        layer = _layer(NVFP4Recipe.V2, bias=bias, kernel_preference=kernel_preference)
        x = _inputs()
        out = layer(x)
        out.float().square().mean().backward()
        results.append(
            (out, x.grad, layer.weight.grad, layer.bias.grad if bias else None)
        )
    for name, triton, cutedsl in zip(
        ("out", "x.grad", "weight.grad", "bias.grad"), *results
    ):
        if triton is not None:
            assert torch.equal(triton, cutedsl), f"{name} differs across backends"


@_needs_v1_requant
@_requires_cutedsl
@torch.no_grad()
def test_v1_requant_forward_is_bitwise_across_backends():
    """The RHT-16 amax, the RTNE row/col quantize and the rowwise weight cast are
    bitwise twins, so the forward agrees exactly. The backward is the one site that
    does not: its stochastic-rounding cast of ``dy`` draws a different Philox stream
    on CuteDSL (see the NVFP4TrainingConfig reproducibility note)."""
    x = _inputs()
    out_t = _layer(NVFP4Recipe.V1_REQUANT, kernel_preference=KernelPreference.TRITON)(x)
    out_c = _layer(NVFP4Recipe.V1_REQUANT, kernel_preference=KernelPreference.CUTEDSL)(
        x
    )
    assert torch.equal(out_t, out_c)


@maybe_sm100
@pytest.mark.parametrize("kernel_preference", _KERNEL_PREFERENCES)
@pytest.mark.parametrize("recipe", _BOTH_RECIPES)
def test_recipe_dispatches_only_the_resolved_backend(recipe, kernel_preference):
    """``NVFP4Linear`` hands its ``kernel_preference`` to the recipe, and the recipe
    runs every quantize and amax op on the backend it resolves to -- except the weight
    amax, which has no CuteDSL twin and stays Triton."""
    layer = _layer(recipe, kernel_preference=kernel_preference)
    x = _inputs()
    with _RecordOps() as recorder:
        layer(x).sum().backward()
    quantizers = {name for name in recorder.names if "_group_" in name}
    assert quantizers == _expected_ops(recipe, kernel_preference)


@maybe_sm100
@pytest.mark.parametrize("recipe", _BOTH_RECIPES)
@torch.no_grad()
def test_auto_falls_back_to_triton_without_cutedsl(recipe, monkeypatch):
    """AUTO degrades to Triton where CuteDSL cannot run, matching an explicit TRITON
    call bitwise (RTNE forward only; the V1_REQUANT backward stream differs)."""
    x = _inputs()
    expected = _layer(recipe, kernel_preference=KernelPreference.TRITON)(x)
    monkeypatch.setattr(
        nvfp4_linear_mod, "cutedsl_nvfp4_kernels_available", lambda: False
    )
    with _RecordOps() as recorder:
        got = _layer(recipe, kernel_preference=KernelPreference.AUTO)(x)
    assert recorder.names, "no torchao op was recorded; the probe is not working"
    leaked = {n for n in recorder.names if n.startswith("torchao::cutedsl_")}
    assert not leaked, f"AUTO reached CuteDSL without the runtime: {sorted(leaked)}"
    assert torch.equal(got, expected)


@maybe_sm100
@pytest.mark.parametrize("recipe", _BOTH_RECIPES)
@torch.no_grad()
def test_cutedsl_raises_without_the_runtime(recipe, monkeypatch):
    monkeypatch.setattr(
        nvfp4_linear_mod, "cutedsl_nvfp4_kernels_available", lambda: False
    )
    layer = _layer(recipe, kernel_preference=KernelPreference.CUTEDSL)
    with pytest.raises(RuntimeError, match="CUTEDSL requires"):
        layer(_inputs())


@_needs_v2
@_requires_cutedsl
def test_cutedsl_prepare_for_cuda_graph_warms_every_v2_kernel():
    """After the pre-capture warm-up, no CuteDSL kernel these recipes reach compiles
    lazily -- the property ``test_nvfp4_linear.py`` pins for V1, for the eleven caches
    V2 and V1_REQUANT reach through the dense and the grouped entry points.

    One shape suffices: every one of these caches is keyed on the device index and
    the op's flags, never on M or N.
    """
    from torchao.prototype.moe_training.nvfp4_training import (
        _cutedsl_group_kernels_impl,
        _cutedsl_kernels_impl,
    )
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm_v2 import (
        nvfp4_v1_requant_grouped_mm,
        nvfp4_v2_grouped_mm,
    )

    v2 = _layer(NVFP4Recipe.V2)
    v1r = _layer(NVFP4Recipe.V1_REQUANT)
    cutedsl_prepare_for_cuda_graph("cuda", sign_vectors=(v1r.rht_sign_vector,))
    group_impl, impl = _cutedsl_group_kernels_impl, _cutedsl_kernels_impl
    caches = {
        "group_amax": group_impl._compile_group_amax_kernel,
        "group_fused": group_impl._compile_group_fused_kernel,
        "group_row_rht_col_rht_amax": group_impl._compile_group_row_rht_col_rht_amax_kernel,
        "group_ms_eden": group_impl._compile_group_row_rht_col_rht_quantize_ms_eden_kernel,
        "group_col_rht_requant_amax": group_impl._compile_group_col_rht_requant_amax_kernel,
        "group_col_rht_requantize": group_impl._compile_group_col_rht_requantize_kernel,
        "group_row_cast_col_rht_amax": group_impl._compile_group_row_cast_col_rht_amax_kernel,
        "group_row_cast_col_rht_quantize": group_impl._compile_group_row_cast_col_rht_quantize_kernel,
        "row_cast_quantize": impl._compile_row_cast_quantize_kernel,
        "requant_amax": impl._compile_requant_amax_kernel,
        "requantize": impl._compile_requantize_kernel,
    }
    before = {k: c.cache_info().misses for k, c in caches.items()}

    num_experts = 2
    offs = torch.tensor([_M // 2, _M], dtype=torch.int32, device="cuda")
    w3d = torch.randn(
        num_experts, _N, _K, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    # Both arithmetic modes, forward and backward, dense and grouped: the fused RHT
    # quantize is keyed on (sr, fast_math), the others on the device alone.
    for use_fast_math in (True, False):
        v2_mod.nvfp4_linear_v2(
            _inputs(),
            v2.weight,
            None,
            wgrad_rht=v2._rht_sign_vector,
            dgrad_rht=v2._dgrad_rht_sign_vector,
            sr_seed=v2._sr_seed,
            kernel_preference=KernelPreference.CUTEDSL,
            use_fast_math=use_fast_math,
        ).sum().backward()
        v2_mod.nvfp4_linear_v1_requant(
            _inputs(),
            v1r.weight,
            None,
            sign_vector=v1r.rht_sign_vector,
            sr_seed=v1r._sr_seed,
            kernel_preference=KernelPreference.CUTEDSL,
            use_fast_math=use_fast_math,
        ).sum().backward()
        nvfp4_v2_grouped_mm(
            _inputs(),
            w3d,
            wgrad_rht=v2._rht_sign_vector,
            dgrad_rht=v2._dgrad_rht_sign_vector,
            sr_seed=v2._sr_seed,
            offs=offs,
            kernel_preference=KernelPreference.CUTEDSL,
            use_fast_math=use_fast_math,
        ).sum().backward()
        nvfp4_v1_requant_grouped_mm(
            _inputs(),
            w3d,
            sign_vector=v1r.rht_sign_vector,
            sr_seed=v1r._sr_seed,
            offs=offs,
            kernel_preference=KernelPreference.CUTEDSL,
            use_fast_math=use_fast_math,
        ).sum().backward()

    new = {k: c.cache_info().misses - before[k] for k, c in caches.items()}
    assert set(new.values()) == {0}, (
        f"{sum(new.values())} CuteDSL kernel(s) compiled after "
        f"cutedsl_prepare_for_cuda_graph ({new}); they would compile inside a "
        "CUDA-graph capture"
    )
