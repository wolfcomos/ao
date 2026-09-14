# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
NVFP4 training configuration and linear module.

Provides NVFP4TrainingConfig for use with quantize_() and an
NVFP4Linear module that performs NVFP4 quantized GEMMs
in both forward and backward passes.

Usage:
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import NVFP4TrainingConfig
    from torchao.quantization import quantize_

    quantize_(model, NVFP4TrainingConfig())
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from torchao.core.config import AOBaseConfig
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    get_wgrad_sign_vector,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import (
    _resolve_use_cutedsl,
    nvfp4_linear,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear_v2 import (
    nvfp4_linear_v1_requant,
    nvfp4_linear_v2,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_recipe import (
    NVFP4Recipe,
    recipe_has_dgrad_rht,
    recipe_rht_size,
    recipe_uses_dynamic_signs,
)
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.quantization.transform_module import register_quantize_module_handler


def _rht_sign_vector_to_tuple(sign_vector: torch.Tensor) -> tuple[int, ...] | None:
    if hasattr(sign_vector, "to_local"):
        sign_vector = sign_vector.to_local()
    if sign_vector.device.type == "meta":
        return None
    return tuple(int(v) for v in sign_vector.detach().cpu().tolist())


def _make_rht_sign_vector(
    sign_vector: torch.Tensor | tuple[int, ...] | list[int] | None,
    device,
    length: int = 16,
) -> torch.Tensor:
    """Materialize an int8 sign buffer of ``length`` elements.

    ``length`` defaults to 16 so existing callers -- including torchtitan's
    ``NVFP4Linear._init_self_buffers`` -- keep working unchanged. Recipe V2 passes
    128.
    """
    if sign_vector is None:
        if device is not None and torch.device(device).type == "meta":
            return torch.empty(length, dtype=torch.int8, device=device)
        return get_wgrad_sign_vector(length, device=device, dtype=torch.int8)

    if isinstance(sign_vector, torch.Tensor):
        if sign_vector.numel() != length:
            raise ValueError(
                f"rht_sign_vector must have {length} elements, "
                f"got {sign_vector.numel()}"
            )
        kwargs = {"dtype": torch.int8}
        if device is not None:
            kwargs["device"] = device
        return sign_vector.detach().to(**kwargs).clone()

    if len(sign_vector) != length:
        raise ValueError(
            f"rht_sign_vector must have {length} elements, got {len(sign_vector)}"
        )
    return torch.tensor(sign_vector, dtype=torch.int8, device=device)


@dataclass
class NVFP4TrainingConfig(AOBaseConfig):
    """Configuration for NVFP4 quantized training.

    When passed to quantize_(), replaces nn.Linear modules with
    NVFP4Linear, which quantizes all three GEMMs (forward
    and backward) to NVFP4.

    Args:
        recipe: Which NVFP4 recipe to run. Defaults to V1, the shipped recipe, so
            an unchanged ``NVFP4TrainingConfig()`` keeps producing exactly what it
            produced before the other two existed.

            V1: RHT-16, stochastic rounding, 2D 16x16 weight quantize. The only
                recipe with a tensor-parallel path.
            V1_REQUANT: V1 with the 2D weight quantize replaced by a 1D 1x16
                quantize plus lazy columnwise requantization in backward. Halves
                the saved weight bytes and reaches forward/dgrad consistency by
                deriving the backward operand from the dequantized forward weight
                rather than from a shared scale byte.
            V2: RHT-128 with independent, resampled ``wgrad``/``dgrad`` sign
                vectors, MS-EDEN on the gradient, and lazy *rotated* requantization.
                Requires a training loop that calls ``resample_nvfp4_rht_signs``
                once per microbatch; without it the sign vectors stay at their
                initial draw, which is correct but forfeits the variance reduction
                resampling buys.

            V1_REQUANT and V2 are single-GPU only for now: ``process_group`` raises.
        kernel_preference: Backend for quantization kernels.
            AUTO: CuteDSL where its runtime allows, Triton otherwise. Both backends
                accept the same shapes, on the tensor-parallel path as on the single-GPU
                one, so the choice is availability alone and there is nothing for AUTO
                to fall back on shape-wise.
            TRITON: Pure-Triton RHT + stochastic rounding path.
            CUTEDSL: CuteDSL kernels for the full quantize path (every amax and
                quantize op the recipe runs except the weight amax, which is Triton
                on every path).
                Requires SM100; in_features divisible by 128 and out_features
                by 128. Under tensor parallel the same constraints apply to each
                per-rank shard, and the per-rank M shard must be divisible by 128.
                Unlike AUTO, an unmet requirement raises instead of falling back.
            Default: AUTO.

            Reproducibility note. AUTO resolves per call site from what the runtime
            offers, so the backend can differ between two nodes running the same code.
            Under stochastic rounding that changes results: the CuteDSL and Triton
            kernels are byte-identical under RTNE but draw *different*
            stochastic-rounding streams (CuteDSL takes one Philox counter per
            16-element block and consumes all four words, rather than reproducing
            Triton's per-packed-byte counter stride). This holds on the **linear and
            grouped paths alike** -- both draw through ``philox4_all``. SR runs in the
            backward pass, so the same seed on a node without the CuteDSL runtime
            yields different gradients -- statistically equivalent, not bitwise equal.
            That applies to V1 and V1_REQUANT, which round ``dy`` stochastically through
            the RHT quantize; V2's MS-EDEN draws the same stream on both backends, so a
            fixed RNG state gives bitwise-equal V2 gradients on either.
            Pin kernel_preference explicitly for runs that must reproduce bitwise
            across machines.
        process_group: Optional ProcessGroup for tensor-parallel TP.
            When set, forward dispatches to the NVFP4 tensor-parallel path on the
            backend kernel_preference resolves to, exactly as the single-GPU path does.
        world_size: TP world size.  Inferred from process_group if None.
        rht_sign_vector: Optional {-1, 1} sign vector of length 16 for the
            randomized Hadamard transform.  When None, each NVFP4Linear draws
            its own random vector.  In multi-rank settings (FSDP) replicas will
            therefore have different bases — harmless for convergence but
            inconsistent across checkpoints.  Callers that require replica
            consistency should broadcast a single vector before calling
            quantize_() and pass it here.  The TP path always enforces
            consistency via _replicate_rht_sign_vector regardless of this field.
        use_fast_math: Match TransformerEngine under ``NVTE_USE_FAST_MATH=1``: the RHT
            quantize consumes the FP32 accumulator directly and takes an approximate
            reciprocal. On by default; both backends implement it and remain bitwise
            identical to TE and to each other. Set False to recover the exact-math
            arithmetic.  Default: True.

    Both defaults moved together, and both change what this config computes:
    ``kernel_preference`` was TRITON and is now AUTO, and ``use_fast_math`` is new and
    defaults on (fast-vs-exact measures 30-32 dB SQNR on the columnwise quantize output
    -- above NVFP4's own ~20 dB quantization noise, but not identical; end to end at the
    linear output the two are ~48 dB apart). A run that must reproduce earlier
    numerics needs both pinned::

        NVFP4TrainingConfig(
            kernel_preference=KernelPreference.TRITON, use_fast_math=False
        )
    """

    recipe: NVFP4Recipe = NVFP4Recipe.V1
    kernel_preference: KernelPreference = KernelPreference.AUTO
    process_group: Optional[object] = field(default=None, compare=False)
    world_size: Optional[int] = None
    rht_sign_vector: Optional[object] = field(default=None, compare=False)
    use_fast_math: bool = True


class NVFP4Linear(nn.Linear):
    """Linear layer with NVFP4 quantized forward and backward GEMMs.

    Drop-in replacement for nn.Linear that quantizes activations, weights,
    and gradients to NVFP4 for all three training GEMMs.

    When process_group is set the forward uses the tensor-parallel protocol
    selected by NVFP4ColwiseParallel or NVFP4RowwiseParallel.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        kernel_preference: KernelPreference = KernelPreference.AUTO,
        process_group=None,
        world_size: Optional[int] = None,
        device=None,
        dtype=None,
        rht_sign_vector: torch.Tensor | tuple[int, ...] | list[int] | None = None,
        use_fast_math: bool = True,
        recipe: NVFP4Recipe = NVFP4Recipe.V1,
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.recipe = recipe
        self.kernel_preference = kernel_preference
        self.use_fast_math = use_fast_math
        self.process_group = process_group
        self.world_size = world_size
        self.tensor_parallel_style = "colwise"
        self.register_buffer(
            "_sr_seed",
            torch.randint(-(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=device),
        )
        # One buffer name across all three recipes, at the recipe's Hadamard size.
        # For V2 this holds wgrad_rht; nvfp4_rht_cadence keys the resample cadence
        # off the length, so a 128-element buffer is resampled and a 16-element one
        # is left alone.
        self.register_buffer(
            "_rht_sign_vector",
            _make_rht_sign_vector(
                rht_sign_vector, device=device, length=recipe_rht_size(recipe)
            ),
            persistent=True,
        )
        if recipe_has_dgrad_rht(recipe):
            self.register_buffer(
                "_dgrad_rht_sign_vector",
                _make_rht_sign_vector(None, device=device, length=128),
                persistent=True,
            )
        self._refresh_rht_sign_vector_tuple()

    def _refresh_rht_sign_vector_tuple(self) -> None:
        # V2 passes the buffer itself to the kernels and resamples it in place, so a
        # cached tuple would go stale on the first resample. Only the static recipes
        # get one.
        if recipe_uses_dynamic_signs(self.recipe):
            self._rht_sign_vector_tuple = None
            return
        self._rht_sign_vector_tuple = _rht_sign_vector_to_tuple(self._rht_sign_vector)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._refresh_rht_sign_vector_tuple()

    @property
    def rht_sign_vector(self) -> tuple[int, ...]:
        if recipe_uses_dynamic_signs(self.recipe):
            raise RuntimeError(
                f"recipe {self.recipe.value} resamples its sign vector, so there is "
                "no stable tuple form; pass the _rht_sign_vector buffer instead"
            )
        if self._rht_sign_vector_tuple is None:
            self._refresh_rht_sign_vector_tuple()
        if self._rht_sign_vector_tuple is None:
            raise RuntimeError("rht_sign_vector is not materialized")
        return self._rht_sign_vector_tuple

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.recipe is not NVFP4Recipe.V1:
            if self.process_group is not None:
                raise NotImplementedError(
                    f"recipe {self.recipe.value} has no tensor-parallel path; "
                    "torchtitan does not use TP NVFP4 linears, so only V1 implements "
                    "one. Use recipe=NVFP4Recipe.V1 or drop process_group."
                )
            if self.recipe is NVFP4Recipe.V2:
                return nvfp4_linear_v2(
                    x,
                    self.weight,
                    self.bias,
                    wgrad_rht=self._rht_sign_vector,
                    dgrad_rht=self._dgrad_rht_sign_vector,
                    sr_seed=self._sr_seed,
                    kernel_preference=self.kernel_preference,
                    use_fast_math=self.use_fast_math,
                )
            return nvfp4_linear_v1_requant(
                x,
                self.weight,
                self.bias,
                sign_vector=self.rht_sign_vector,
                sr_seed=self._sr_seed,
                kernel_preference=self.kernel_preference,
                use_fast_math=self.use_fast_math,
            )
        if self.process_group is not None and self.kernel_preference in (
            KernelPreference.AUTO,
            KernelPreference.TRITON,
            KernelPreference.CUTEDSL,
        ):
            import torch.distributed as dist
            from torch.distributed.tensor import DTensor

            from torchao.prototype.moe_training.nvfp4_training.nvfp4_tensor_parallel import (
                nvfp4_col_parallel_linear,
                nvfp4_row_parallel_linear,
            )

            ws = self.world_size
            if ws is None:
                ws = dist.get_world_size(self.process_group)
            sr_seed = self._sr_seed
            if isinstance(sr_seed, DTensor):
                sr_seed = sr_seed.to_local()
            w = self.weight
            if isinstance(w, DTensor):
                w = w.to_local()
            bias = self.bias
            if isinstance(bias, DTensor):
                bias = bias.to_local()
            tp_linear = (
                nvfp4_row_parallel_linear
                if self.tensor_parallel_style == "rowwise"
                else nvfp4_col_parallel_linear
            )
            return tp_linear(
                x,
                w,
                bias,
                sr_seed=sr_seed,
                tp_group=self.process_group,
                world_size=ws,
                sign_vector=self.rht_sign_vector,
                use_cutedsl=_resolve_use_cutedsl(self.kernel_preference),
                use_fast_math=self.use_fast_math,
            )
        return nvfp4_linear(
            x,
            self.weight,
            self.bias,
            kernel_preference=self.kernel_preference,
            sr_seed=self._sr_seed,
            sign_vector=self.rht_sign_vector,
            use_fast_math=self.use_fast_math,
        )

    @classmethod
    def from_linear(
        cls,
        mod: nn.Linear,
        kernel_preference: KernelPreference = KernelPreference.AUTO,
        process_group=None,
        world_size: Optional[int] = None,
        rht_sign_vector: torch.Tensor | tuple[int, ...] | list[int] | None = None,
        use_fast_math: bool = True,
        recipe: NVFP4Recipe = NVFP4Recipe.V1,
    ) -> "NVFP4Linear":
        if rht_sign_vector is None:
            inherited = getattr(mod, "_rht_sign_vector", None)
            # Only inherit a vector that matches this recipe's Hadamard size; a
            # V1 module re-quantized as V2 must draw a fresh 128-element one.
            if inherited is not None and inherited.numel() == recipe_rht_size(recipe):
                rht_sign_vector = inherited
        new = cls(
            mod.in_features,
            mod.out_features,
            mod.bias is not None,
            kernel_preference=kernel_preference,
            process_group=process_group,
            world_size=world_size,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
            rht_sign_vector=rht_sign_vector,
            use_fast_math=use_fast_math,
            recipe=recipe,
        )
        # Copy weights (don't re-init)
        if mod.weight.device != torch.device("meta"):
            new.weight = mod.weight
            if mod.bias is not None:
                new.bias = mod.bias
        return new


@register_quantize_module_handler(NVFP4TrainingConfig)
def _nvfp4_training_transform(
    module: nn.Module,
    config: NVFP4TrainingConfig,
    parameter_name: Optional[str] = None,
) -> nn.Module:
    """Handler for quantize_(): replaces nn.Linear with NVFP4Linear."""
    if isinstance(module, NVFP4Linear):
        return module
    if isinstance(module, nn.Linear):
        return NVFP4Linear.from_linear(
            module,
            kernel_preference=config.kernel_preference,
            process_group=config.process_group,
            world_size=config.world_size,
            rht_sign_vector=config.rht_sign_vector,
            use_fast_math=config.use_fast_math,
            recipe=config.recipe,
        )
    return module
