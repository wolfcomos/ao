# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for the NVFP4 training kernel benchmarks.

The benchmarks compare the Triton and CuteDSL backends of each quantize op on the
same shapes, reporting **device kernel time** (CUDA self-time) rather than wall clock:
NVFP4 training runs the quantizes under CUDA graphs / ``torch.compile``, so host launch
overhead is amortized and the kernel's device time is the metric that matters.

The activation benches take their per-expert token counts and ``(tokens, dim)`` shapes
from here (``get_deepseek_v3_activation_shapes``); ``deepseek_v3_shapes.py`` stays the
model and weight shapes alone.
"""

from dataclasses import dataclass

import torch
from torch.profiler import ProfilerActivity, profile

from benchmarks.prototype.nvfp4_training.deepseek_v3_shapes import (
    get_deepseek_v3_weight_shapes,
)

# Per-expert tokens per step for a balanced router, EP-group ranks x local batch x seq x
# top_k / experts, from the DeepSeek-V3 training configurations the benches target
# (torchtitan flavors). debugmodel has no such run; 256 keeps its rows at 16 tiles for
# E=4, the launch floor. 16B: 2 nodes x 4 GPUs, ep 8, local batch 4, seq 4096, top_k 6 ->
# 8*4*4096*6/64 = 12288. 671B: 16 nodes x 4 GPUs at ep 32, local batch 8, seq 4096, top_k 8
# -> 32*8*4096*8/256 = 32768 (the EP=64 layout with the same batch would give 65536).
DEEPSEEK_V3_TOKENS_PER_EXPERT = {"debugmodel": 256, "16B": 12288, "671B": 32768}


@dataclass(frozen=True)
class DeepSeekV3ActivationShape:
    model: str
    projection: str
    experts: int
    tokens: int
    dim: int


def get_deepseek_v3_activation_shapes(
    side: str, *, factorized_experts: int | None = None
) -> list[DeepSeekV3ActivationShape]:
    """Return the packed ``x`` (tokens, n) or ``dy`` (tokens, m) per weight shape."""
    return [
        DeepSeekV3ActivationShape(
            shape.model,
            shape.projection,
            shape.experts,
            DEEPSEEK_V3_TOKENS_PER_EXPERT[shape.model],
            {"x": shape.n, "dy": shape.m}[side],
        )
        for shape in get_deepseek_v3_weight_shapes(
            factorized_experts=factorized_experts
        )
    ]


def kernel_time_us(fn, warmup: int = 15, iters: int = 50) -> float:
    """Device kernel time per call (us): summed CUDA self-time averaged over ``iters``.

    Excludes host overhead (so the custom-op dispatch each backend pays is not counted)
    and memcpy/memset, isolating the kernels' device time for an apples-to-apples compare.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total = sum(
        (
            getattr(e, "self_device_time_total", 0)
            or getattr(e, "self_cuda_time_total", 0)
        )
        for e in prof.key_averages()
        if "memcpy" not in e.key.lower() and "memset" not in e.key.lower()
    )
    return total / iters
