# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark the grouped rowwise-cast + columnwise-RHT amax kernel across backends (triton, cutedsl).

One launch over the packed activation x = (E * tokens, dim) bf16 computes per group the
raw rowwise amax of |x_g| and the amax of |x_g^T @ R| with a 128-point randomized Hadamard
transform along the tokens (wgrad signs) -- the V2 forward activation operands, the
``dynamic_rht=True`` path of ``group_rht_amax``. Reports device kernel time (see
bench_utils.kernel_time_us) for each available backend on the DeepSeek-V3 shapes, with the
cutedsl-vs-triton speedup.

    python -m benchmarks.prototype.nvfp4_training.bench_group_row_cast_col_rht_amax
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from tabulate import tabulate
from torch.utils._triton import has_triton
from tqdm import tqdm

from benchmarks.prototype.nvfp4_training.bench_utils import (
    get_deepseek_v3_activation_shapes,
    kernel_time_us,
)
from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
    VARYING_FIRST_DIM,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.utils import is_sm_at_least_100

device = torch.device("cuda")

BACKENDS = ("triton", "cutedsl")

# The target deployment is high expert parallelism, so the small-E shapes are the
# representative ones; the ranking inverts at large E and misleads.
LOCAL_EXPERTS = 4


@dataclass(frozen=True)
class ExperimentConfig:
    experts: int
    tokens: int
    dim: int
    model: str = ""
    projection: str = ""


@dataclass(frozen=True)
class ExperimentResult:
    us: Dict[str, float]  # backend -> device kernel time (us)
    read_bytes: int


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def _signs(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (128,), generator=generator, dtype=torch.int8)
    return (bits * 2 - 1).to(device)


def make_runner(
    backend: str,
    x: torch.Tensor,
    wgrad_rht: torch.Tensor,
    offsets: torch.Tensor,
    num_tensors: int,
    logical_packed_length: torch.Tensor,
) -> Optional[Callable[[], object]]:
    """No-arg callable running ``backend``'s grouped amax op, or None if unavailable."""
    psl, hidden = x.shape
    if backend == "triton":
        if not has_triton():
            return None
        from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_triton import (
            triton_group_rht_amax as op,
        )
    elif backend == "cutedsl":
        if not cutedsl_nvfp4_kernels_available():
            return None
        from torchao.prototype.moe_training.nvfp4_training.group_hadamard_amax_cutedsl import (
            cutedsl_group_rht_amax as op,
        )
    else:
        raise ValueError(f"unknown backend {backend}")

    return lambda: op(
        x,
        [],
        offsets,
        num_tensors,
        psl,
        hidden,
        VARYING_FIRST_DIM,
        logical_packed_length=logical_packed_length,
        sign_tensor=wgrad_rht,
        dynamic_rht=True,
    )


def run_experiment(config: ExperimentConfig) -> Optional[ExperimentResult]:
    E, tokens, dim = config.experts, config.tokens, config.dim
    x = torch.randn((E * tokens, dim), dtype=torch.bfloat16, device=device)
    wgrad_rht = _signs(seed=0)
    offsets = torch.arange(1, E + 1, dtype=torch.int32, device=device) * tokens
    logical_packed_length = offsets[-1:]

    us: Dict[str, float] = {}
    for backend in BACKENDS:
        runner = make_runner(backend, x, wgrad_rht, offsets, E, logical_packed_length)
        if runner is not None:
            us[backend] = kernel_time_us(runner)
    if not us:
        return None
    # amax reads the full bfloat16 input; the 2E scalar outputs are negligible.
    return ExperimentResult(us=us, read_bytes=x.numel() * 2)


def print_results(experiments: List[Experiment]) -> None:
    headers = [
        "model",
        "projection",
        "E",
        "tokens",
        "dim",
        "cutedsl_us",
        "triton_us",
        "speedup",
        "cutedsl_gbps",
    ]
    rows = []
    for e in experiments:
        us = e.result.us
        c, t = us.get("cutedsl"), us.get("triton")
        speedup = f"{t / c:.2f}x" if (c and t) else "n/a"
        ref = c or t
        gbps = (e.result.read_bytes / 1e9) / (ref / 1e6)
        rows.append(
            [
                e.config.model,
                e.config.projection,
                e.config.experts,
                e.config.tokens,
                e.config.dim,
                round(c, 3) if c else "n/a",
                round(t, 3) if t else "n/a",
                speedup,
                round(gbps, 1),
            ]
        )
    print(tabulate(rows, headers=headers))


def main() -> None:
    if not torch.cuda.is_available() or not is_sm_at_least_100():
        raise RuntimeError("Grouped NVFP4 amax requires SM100+")

    torch.random.manual_seed(123)
    configs = [
        ExperimentConfig(
            shape.experts,
            shape.tokens,
            shape.dim,
            model=shape.model,
            projection=shape.projection,
        )
        for shape in get_deepseek_v3_activation_shapes(
            "x", factorized_experts=LOCAL_EXPERTS
        )
    ]
    experiments = []
    for config in tqdm(configs):
        result = run_experiment(config)
        if result is not None:
            experiments.append(Experiment(config=config, result=result))
    print_results(experiments)


if __name__ == "__main__":
    main()
