# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 grouped GEMMs and MoE FFN recipe routing. Design doc §17.

§17 routes FC1 (``w1``/``w3``) to V1_REQUANT and FC2 (``w2``) to V2. The tests that
matter most here are the routing ones: that each layer reaches only its own recipe's
kernels, and that FC1 and FC2 own independent sign vectors and seeds. A shared buffer
between them is an explicit test failure, not a performance detail.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_recipe import NVFP4Recipe
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.quantization.utils import compute_error

from ._v2_marks import CUTEDSL_AVAILABLE, maybe_sm100, requires_cutedsl
from ._v2_test_utils import assert_ms_eden_modes

_needs_v1_requant = requires_cutedsl
_needs_v2 = requires_cutedsl

if CUTEDSL_AVAILABLE:
    from torchao.prototype.moe_training.nvfp4_training import (
        nvfp4_grouped_mm_v2 as gmm_v2_mod,
    )
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm_v2 import (
        nvfp4_v1_requant_grouped_mm,
        nvfp4_v2_grouped_mm,
    )

_KERNEL_PREFERENCES = [
    pytest.param(KernelPreference.AUTO, id="auto"),
    pytest.param(
        KernelPreference.CUTEDSL,
        marks=pytest.mark.skipif(
            not cutedsl_nvfp4_kernels_available(),
            reason="requires the CuteDSL runtime",
        ),
        id="cutedsl",
    ),
]

# For tests that require the CuTeDSL runtime.
_requires_cutedsl = pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(), reason="requires the CuteDSL runtime"
)

_RECIPES = [
    pytest.param(NVFP4Recipe.V1_REQUANT, id="v1_requant"),
    pytest.param(NVFP4Recipe.V2, id="v2"),
]

# Kernel names without their backend prefix: the routing claims below hold on
# either backend.
_MS_EDEN_OP = "group_row_rht_col_rht_quantize_ms_eden"
_SR_CAST_OP = "group_rht_quantize_row_col"
_ROTATED_REQUANT_OP = "group_col_rht_requantize"
_PLAIN_REQUANT_OP = "group_col_cast_requantize"

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
    },
    NVFP4Recipe.V1_REQUANT: {
        "group_rht_amax",
        "group_rht_quantize_row_col",
        "group_row_cast_quantize",
        "group_col_cast_requant_amax",
        "group_col_cast_requantize",
    },
}


def _expected_ops(recipe, kernel_preference):
    return {"torchao::cutedsl_" + k for k in _KERNELS[recipe] - {"group_weight_amax"}}


class _RecordOps(TorchDispatchMode):
    def __init__(self):
        self.names = set()
        self.kernels = set()
        self.quantize_fast_math = {}
        self.ms_eden_inputs = None
        self.ms_eden_outputs = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.name() if hasattr(func, "name") else str(func)
        if name.startswith("torchao::"):
            name = name.split(".")[0]
            self.names.add(name)
            # torchao::<backend>_<kernel> -> <kernel>
            self.kernels.add(name.split("::")[1].split("_", 1)[1])
            if name.endswith(
                (
                    "_group_rht_quantize_row_col",
                    "_group_row_rht_col_rht_quantize_ms_eden",
                )
            ):
                # Dispatcher schemas supply defaults when trailing args are omitted.
                bound = {a.name: a.default_value for a in func._schema.arguments}
                bound.update(zip((a.name for a in func._schema.arguments), args))
                bound.update(kwargs or {})
                is_ms_eden = name.endswith("_group_row_rht_col_rht_quantize_ms_eden")
                stage = "ms_eden" if is_ms_eden else "activation"
                self.quantize_fast_math[stage] = bound["use_fast_math"]
                if is_ms_eden:
                    self.ms_eden_inputs = tuple(
                        bound[a.name]
                        for a in func._schema.arguments
                        if a.name != "use_fast_math"
                    )
        result = func(*args, **(kwargs or {}))
        if name.endswith("_group_row_rht_col_rht_quantize_ms_eden"):
            self.ms_eden_outputs = result
        return result


def _signs(seed=0, n=128):
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (n,), generator=generator, dtype=torch.int8)
    return (bits * 2 - 1).cuda()


def _seed(value):
    return torch.tensor([value], dtype=torch.int64, device="cuda")


def _moe_inputs(group_sizes, D, F_dim, *, seed=0):
    torch.manual_seed(seed)
    E = len(group_sizes)
    x = torch.randn(
        sum(group_sizes), D, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    kw = dict(device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w1 = (torch.randn(E, F_dim, D, **kw) * 0.02).detach().requires_grad_(True)
    w3 = (torch.randn(E, F_dim, D, **kw) * 0.02).detach().requires_grad_(True)
    w2 = (torch.randn(E, D, F_dim, **kw) * 0.02).detach().requires_grad_(True)
    offs = torch.cumsum(
        torch.tensor(group_sizes, dtype=torch.int32, device="cuda"),
        0,
        dtype=torch.int32,
    )
    return x, w1, w2, w3, offs


def _ffn(x, w1, w2, w3, offs, state):
    """§17's call shape: FC1 on V1_REQUANT, FC2 on V2, one ``offs`` per forward."""
    gate = nvfp4_v1_requant_grouped_mm(
        x, w1, sign_vector=state["fc1_signs"], sr_seed=state["fc1_seed"], offs=offs
    )
    up = nvfp4_v1_requant_grouped_mm(
        x, w3, sign_vector=state["fc1_signs"], sr_seed=state["fc1_seed"], offs=offs
    )
    hidden = F.silu(gate) * up
    return nvfp4_v2_grouped_mm(
        hidden,
        w2,
        wgrad_rht=state["fc2_wgrad"],
        dgrad_rht=state["fc2_dgrad"],
        sr_seed=state["fc2_seed"],
        offs=offs,
    )


def _state():
    return {
        "fc1_signs": tuple(int(v) for v in _signs(seed=10, n=16).tolist()),
        "fc1_seed": _seed(11),
        "fc2_wgrad": _signs(seed=20),
        "fc2_dgrad": _signs(seed=21),
        "fc2_seed": _seed(22),
    }


def _bf16_ffn(x, w1, w2, w3, group_sizes):
    out, start = [], 0
    for e, size in enumerate(group_sizes):
        xs = x[start : start + size]
        h = F.silu(xs @ w1[e].t()) * (xs @ w3[e].t())
        out.append(h @ w2[e].t())
        start += size
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Numerics and routing -- gated on the kernel bodies
# ---------------------------------------------------------------------------


@_needs_v1_requant
@torch.no_grad()
def test_fc1_alone_vs_bf16_grouped_reference():
    sizes = [128, 128]
    x, w1, _, _, offs = _moe_inputs(sizes, 256, 512)
    got = nvfp4_v1_requant_grouped_mm(
        x.detach(), w1.detach(), sign_vector=(1,) * 16, sr_seed=_seed(1), offs=offs
    )
    want = torch.cat(
        [
            x[s : s + n].detach() @ w1[e].detach().t()
            for e, (s, n) in enumerate(zip([0, 128], sizes))
        ]
    )
    assert compute_error(want.float(), got.float()) > 12.0


@_needs_v2
@torch.no_grad()
def test_fc2_alone_vs_bf16_grouped_reference():
    sizes = [128, 128]
    x, _, w2, _, offs = _moe_inputs(sizes, 512, 256)
    h = torch.randn(sum(sizes), 256, device="cuda", dtype=torch.bfloat16)
    got = nvfp4_v2_grouped_mm(
        h,
        w2.detach(),
        wgrad_rht=_signs(0),
        dgrad_rht=_signs(1),
        sr_seed=_seed(1),
        offs=offs,
    )
    want = torch.cat(
        [
            h[s : s + n] @ w2[e].detach().t()
            for e, (s, n) in enumerate(zip([0, 128], sizes))
        ]
    )
    assert compute_error(want.float(), got.float()) > 12.0


@_needs_v2
def test_full_ffn_forward_and_backward():
    sizes = [128, 256, 128]
    x, w1, w2, w3, offs = _moe_inputs(sizes, 256, 512)
    out = _ffn(x, w1, w2, w3, offs, _state())
    assert out.shape == (sum(sizes), 256)
    out.float().square().mean().backward()
    for name, p in (("x", x), ("w1", w1), ("w2", w2), ("w3", w3)):
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all(), f"{name} gradient has non-finite values"


@_needs_v2
@pytest.mark.parametrize("kernel_preference", _KERNEL_PREFERENCES)
def test_recipe_routing_reaches_only_its_own_kernels(kernel_preference):
    """§17 smoke test 4: FC1 calls no MS-EDEN op; FC2 requantizes with a rotation.

    Both recipes share the forward activation quantizer (``_SR_CAST_OP``) -- V2 drives
    it at RHT-128 with ``dynamic_rht`` -- so the discriminating ops are the backward
    ones: MS-EDEN and the rotated requant belong to V2, the plain requant to
    V1_REQUANT.
    """
    sizes = [128, 128]
    x, w1, w2, w3, offs = _moe_inputs(sizes, 256, 512)
    state = _state()

    with _RecordOps() as fc1:
        nvfp4_v1_requant_grouped_mm(
            x,
            w1,
            sign_vector=state["fc1_signs"],
            sr_seed=state["fc1_seed"],
            offs=offs,
            kernel_preference=kernel_preference,
        ).sum().backward()
    assert _MS_EDEN_OP not in fc1.kernels, "V1_REQUANT must not reach MS-EDEN"
    assert _ROTATED_REQUANT_OP not in fc1.kernels, (
        "V1_REQUANT applies no dgrad rotation"
    )
    assert _PLAIN_REQUANT_OP in fc1.kernels
    assert _SR_CAST_OP in fc1.kernels

    h = torch.randn(
        sum(sizes), 512, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    with _RecordOps() as fc2:
        nvfp4_v2_grouped_mm(
            h,
            w2,
            wgrad_rht=state["fc2_wgrad"],
            dgrad_rht=state["fc2_dgrad"],
            sr_seed=state["fc2_seed"],
            offs=offs,
            kernel_preference=kernel_preference,
        ).sum().backward()
    assert _PLAIN_REQUANT_OP not in fc2.kernels, "V2 requantizes with a rotation"
    assert _MS_EDEN_OP in fc2.kernels and _ROTATED_REQUANT_OP in fc2.kernels
    assert _SR_CAST_OP in fc2.kernels, "V2 shares the forward activation quantizer"


@_needs_v2
def test_recipes_can_be_swapped():
    """§17 extra test: the FC1/FC2 split is a configuration choice, not an assumption.

    Running FC1 on V2 and FC2 on V1_REQUANT must work. If it does not, something has
    hard-coded the routing that §17 describes as a tunable.
    """
    sizes = [128, 128]
    x, w1, w2, w3, offs = _moe_inputs(sizes, 256, 512)
    gate = nvfp4_v2_grouped_mm(
        x, w1, wgrad_rht=_signs(0), dgrad_rht=_signs(1), sr_seed=_seed(1), offs=offs
    )
    out = nvfp4_v1_requant_grouped_mm(
        F.silu(gate),
        w2,
        sign_vector=(1,) * 16,
        sr_seed=_seed(2),
        offs=offs,
    )
    out.sum().backward()
    assert torch.isfinite(x.grad).all()


@_needs_v2
@torch.no_grad()
def test_uneven_groups_match_the_bf16_reference_per_group():
    sizes = [128, 384, 256]
    x, w1, w2, w3, offs = _moe_inputs(sizes, 256, 512)
    got = _ffn(x.detach(), w1.detach(), w2.detach(), w3.detach(), offs, _state())
    want = _bf16_ffn(x.detach(), w1.detach(), w2.detach(), w3.detach(), sizes)
    assert compute_error(want.float(), got.float()) > 8.0


@_needs_v2
def test_single_expert_matches_three_dense_linears():
    """§17 smoke test 5: at ``E = 1`` the FFN reduces to two linears plus one."""
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear_v2 import (
        nvfp4_linear_v1_requant,
        nvfp4_linear_v2,
    )

    sizes = [256]
    x, w1, w2, w3, offs = _moe_inputs(sizes, 256, 512)
    state = _state()
    grouped = _ffn(x.detach(), w1.detach(), w2.detach(), w3.detach(), offs, state)

    gate = nvfp4_linear_v1_requant(
        x.detach(),
        w1.detach()[0],
        sign_vector=state["fc1_signs"],
        sr_seed=state["fc1_seed"],
    )
    up = nvfp4_linear_v1_requant(
        x.detach(),
        w3.detach()[0],
        sign_vector=state["fc1_signs"],
        sr_seed=state["fc1_seed"],
    )
    dense = nvfp4_linear_v2(
        F.silu(gate) * up,
        w2.detach()[0],
        wgrad_rht=state["fc2_wgrad"],
        dgrad_rht=state["fc2_dgrad"],
        sr_seed=state["fc2_seed"],
    )
    torch.testing.assert_close(grouped, dense, atol=0, rtol=0)


@_needs_v2
@_requires_cutedsl
@pytest.mark.parametrize(
    "sizes,pad",
    [([128, 384, 256], False), ([64, 320, 128], True)],
    ids=["aligned", "padded"],
)
def test_v2_auto_matches_explicit_cutedsl_for_a_fixed_rng_state(
    sizes, pad, monkeypatch
):
    """AUTO and explicit CuTeDSL select the same V2 operators and RNG mapping."""
    fixed = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    # The grouped module binds _backward_rng_state at import, so it is pinned here.
    monkeypatch.setattr(gmm_v2_mod, "_backward_rng_state", lambda sr_seed: fixed)
    state = _state()

    results = []
    for kernel_preference in (KernelPreference.AUTO, KernelPreference.CUTEDSL):
        _, _, w2, _, offs = _moe_inputs(sizes, 512, 256)
        h = torch.randn(
            sum(sizes), 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        out = nvfp4_v2_grouped_mm(
            h,
            w2,
            wgrad_rht=state["fc2_wgrad"],
            dgrad_rht=state["fc2_dgrad"],
            sr_seed=state["fc2_seed"],
            offs=offs,
            pad_token_groups_for_grouped_mm=pad,
            kernel_preference=kernel_preference,
        )
        out.float().square().mean().backward()
        results.append((out, h.grad, w2.grad))
    for name, triton, cutedsl in zip(("out", "h.grad", "w2.grad"), *results):
        assert torch.equal(triton, cutedsl), f"{name} differs across backends"


@_needs_v2
@_requires_cutedsl
def test_v2_use_fast_math_controls_activation_and_ms_eden(monkeypatch):
    """One flag selects both stages; compare MS-EDEN modes with the same dy."""
    fixed = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    monkeypatch.setattr(gmm_v2_mod, "_backward_rng_state", lambda sr_seed: fixed)
    sizes = [128, 384, 256]
    state = _state()
    torch.manual_seed(19)
    dy = torch.randn(sum(sizes), 512, device="cuda", dtype=torch.bfloat16)

    def step(**kwargs):
        _, _, w2, _, offs = _moe_inputs(sizes, 512, 256)
        h = torch.randn(
            sum(sizes), 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        with _RecordOps() as recorder:
            out = nvfp4_v2_grouped_mm(
                h,
                w2,
                wgrad_rht=state["fc2_wgrad"],
                dgrad_rht=state["fc2_dgrad"],
                sr_seed=state["fc2_seed"],
                offs=offs,
                kernel_preference=KernelPreference.CUTEDSL,
                **kwargs,
            )
            out.backward(dy)
        return recorder, out, h.grad, w2.grad

    default = step()
    fast = step(use_fast_math=True)
    exact = step(use_fast_math=False)
    for result, expected in ((default, True), (fast, True), (exact, False)):
        assert result[0].quantize_fast_math == {
            "activation": expected,
            "ms_eden": expected,
        }
        assert all(torch.isfinite(t).all() for t in result[1:])
    for name, want, got in zip(("out", "h.grad", "w2.grad"), default[1:], fast[1:]):
        assert torch.equal(want, got), f"{name} differs between omitted and True"
    assert_ms_eden_modes(
        exact[0].ms_eden_inputs,
        exact[0].ms_eden_outputs,
        fast[0].ms_eden_outputs,
    )
    # Activation quantization also changes, so dW is not a scale-only comparison.
    for name, want, got in zip(("out", "h.grad", "w2.grad"), exact[1:], fast[1:]):
        assert compute_error(want.float(), got.float()) > 20.0, name


@maybe_sm100
@torch.no_grad()
def test_v2_use_fast_math_refuses_triton():
    """Explicit TRITON is rejected for V2, including the MS-EDEN fast path."""
    _, _, w2, _, offs = _moe_inputs([128, 128], 512, 256)
    h = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="AUTO or CUTEDSL"):
        nvfp4_v2_grouped_mm(
            h,
            w2.detach(),
            wgrad_rht=_signs(0),
            dgrad_rht=_signs(1),
            sr_seed=_seed(1),
            offs=offs,
            kernel_preference=KernelPreference.TRITON,
            use_fast_math=True,
        )


@_needs_v1_requant
@_requires_cutedsl
@torch.no_grad()
def test_v1_requant_auto_matches_explicit_cutedsl():
    """AUTO and explicit CuTeDSL agree on the lazy weight recipe forward."""
    sizes = [128, 384, 256]
    state = _state()
    outs = []
    for kernel_preference in (KernelPreference.AUTO, KernelPreference.CUTEDSL):
        x, w1, _, _, offs = _moe_inputs(sizes, 256, 512)
        outs.append(
            nvfp4_v1_requant_grouped_mm(
                x,
                w1,
                sign_vector=state["fc1_signs"],
                sr_seed=state["fc1_seed"],
                offs=offs,
                kernel_preference=kernel_preference,
            )
        )
    assert torch.equal(*outs)


@maybe_sm100
@pytest.mark.parametrize("kernel_preference", _KERNEL_PREFERENCES)
@pytest.mark.parametrize("recipe", _RECIPES)
def test_recipe_dispatches_only_the_resolved_backend(recipe, kernel_preference):
    """Every quantization/RHT custom operator belongs to CuTeDSL; raw weight amax uses ATen."""
    sizes = [128, 128]
    x, w1, w2, _, offs = _moe_inputs(sizes, 256, 512)
    state = _state()
    with _RecordOps() as recorder:
        if recipe is NVFP4Recipe.V2:
            h = torch.randn(
                sum(sizes), 512, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            nvfp4_v2_grouped_mm(
                h,
                w2,
                wgrad_rht=state["fc2_wgrad"],
                dgrad_rht=state["fc2_dgrad"],
                sr_seed=state["fc2_seed"],
                offs=offs,
                kernel_preference=kernel_preference,
            ).sum().backward()
        else:
            nvfp4_v1_requant_grouped_mm(
                x,
                w1,
                sign_vector=state["fc1_signs"],
                sr_seed=state["fc1_seed"],
                offs=offs,
                kernel_preference=kernel_preference,
            ).sum().backward()
    assert recorder.names == _expected_ops(recipe, kernel_preference)


@maybe_sm100
@_requires_cutedsl
@torch.no_grad()
def test_cutedsl_rejects_more_than_64_experts():
    """The four token-jagged CuteDSL kernels take at most 64 groups; an explicit
    CUTEDSL request past the cap is refused by the resolver before any kernel runs."""
    x, w1, _, _, offs = _moe_inputs([128] * 65, 256, 256)
    with pytest.raises(ValueError, match="at most 64 experts"):
        nvfp4_v2_grouped_mm(
            x,
            w1,
            wgrad_rht=_signs(0),
            dgrad_rht=_signs(1),
            sr_seed=_seed(1),
            offs=offs,
            kernel_preference=KernelPreference.CUTEDSL,
        )


# ---------------------------------------------------------------------------
# Wrapper layer -- runs today
# ---------------------------------------------------------------------------


@maybe_sm100
def test_fc1_and_fc2_state_must_be_independent():
    """§17: once the recipes differ, FC1 and FC2 can no longer share a sign vector or
    a seed. Pinned as a property of the state the caller assembles."""
    state = _state()
    assert state["fc1_seed"].item() != state["fc2_seed"].item()
    assert len(state["fc1_signs"]) == 16, "V1_REQUANT is RHT-16"
    assert state["fc2_wgrad"].numel() == 128, "V2 is RHT-128"
    assert not torch.equal(state["fc2_wgrad"], state["fc2_dgrad"])
    assert state["fc2_wgrad"].data_ptr() != state["fc2_dgrad"].data_ptr()


@maybe_sm100
@torch.no_grad()
def test_offs_is_required_and_validated():
    x, w1, _, _, offs = _moe_inputs([128, 128], 256, 512)
    with pytest.raises(ValueError, match="offs is required"):
        nvfp4_v1_requant_grouped_mm(
            x.detach(), w1.detach(), sign_vector=(1,) * 16, sr_seed=_seed(1), offs=None
        )
    with pytest.raises(ValueError, match="1D int32"):
        nvfp4_v1_requant_grouped_mm(
            x.detach(),
            w1.detach(),
            sign_vector=(1,) * 16,
            sr_seed=_seed(1),
            offs=offs.long(),
        )
    with pytest.raises(ValueError, match="one group-end offset per expert"):
        nvfp4_v1_requant_grouped_mm(
            x.detach(),
            w1.detach(),
            sign_vector=(1,) * 16,
            sr_seed=_seed(1),
            offs=offs[:1],
        )


@maybe_sm100
@torch.no_grad()
def test_v2_requires_128_element_sign_tensors():
    x, _, w2, _, offs = _moe_inputs([128, 128], 512, 256)
    h = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"wgrad_rht must be a \(128,\) tensor"):
        nvfp4_v2_grouped_mm(
            h,
            w2.detach(),
            wgrad_rht=_signs(0, n=16),
            dgrad_rht=_signs(1),
            sr_seed=_seed(1),
            offs=offs,
        )


@maybe_sm100
@torch.no_grad()
def test_contraction_dimension_mismatch_is_caught():
    x, w1, _, _, offs = _moe_inputs([128, 128], 256, 512)
    bad = torch.randn(2, 512, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="contraction dimensions differ"):
        nvfp4_v1_requant_grouped_mm(
            x.detach(), bad, sign_vector=(1,) * 16, sr_seed=_seed(1), offs=offs
        )


@pytest.mark.parametrize(
    "preference", [KernelPreference.TRITON, KernelPreference.TORCH]
)
def test_lazy_recipes_reject_unsupported_backends(preference):
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_recipe import (
        _require_cutedsl,
    )

    with pytest.raises(ValueError, match="AUTO or CUTEDSL"):
        _require_cutedsl(preference)
