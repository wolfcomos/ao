# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark the grouped rowwise-cast + columnwise-RHT quantize kernel across backends (triton, cutedsl).

One launch over the packed activation x = (E * tokens, dim) bf16 quantizes per group the
raw rows of x_g and the columns of x_g^T @ R -- a 128-point randomized Hadamard transform
along the tokens (wgrad signs) -- to NVFP4 codes and swizzled E4M3 block scales against the
group's two amaxes: the V2 forward activation operands, the ``dynamic_rht=True`` path of
``group_rht_quantize_row_col``. RTNE with ``use_fast_math=True``, the recipe default (the
baseline table's rows); stochastic rounding is refused on this path by the CuteDSL op.
Reports device kernel time (see bench_utils.kernel_time_us) for each available backend on
the DeepSeek-V3 shapes, with the cutedsl-vs-triton speedup.

    python -m benchmarks.prototype.nvfp4_training.bench_group_row_cast_col_rht_quantize
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from tabulate import tabulate
from torch.utils._triton import has_triton
from tqdm import tqdm

from benchmarks.prototype.nvfp4_training.bench_utils import kernel_time_us
from benchmarks.prototype.nvfp4_training.deepseek_v3_shapes import (
    get_deepseek_v3_activation_shapes,
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
    total_bytes: int


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
    row_amax: torch.Tensor,
    col_amax: torch.Tensor,
    logical_packed_length: torch.Tensor,
) -> Optional[Callable[[], object]]:
    """No-arg callable running ``backend``'s grouped quantize op, or None if unavailable."""
    psl, hidden = x.shape
    if backend == "triton":
        if not has_triton():
            return None
        from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_triton import (
            triton_group_rht_quantize_row_col as op,
        )
    elif backend == "cutedsl":
        if not cutedsl_nvfp4_kernels_available():
            return None
        from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_cutedsl import (
            cutedsl_group_rht_quantize_row_col as op,
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
        row_amax,
        col_amax,
        None,
        False,
        logical_packed_length,
        True,  # use_fast_math: the recipe default, the baseline table's rows
        sign_tensor=wgrad_rht,
        dynamic_rht=True,
    )


def run_experiment(config: ExperimentConfig) -> Optional[ExperimentResult]:
    E, tokens, dim = config.experts, config.tokens, config.dim
    x = torch.randn((E * tokens, dim), dtype=torch.bfloat16, device=device)
    wgrad_rht = _signs(seed=0)
    offsets = torch.arange(1, E + 1, dtype=torch.int32, device=device) * tokens
    logical_packed_length = offsets[-1:]
    # Per-group amaxes (values do not affect timing); compute cheaply from x.
    row_amax = x.view(E, tokens, dim).float().abs().amax(dim=(1, 2)).contiguous()
    col_amax = row_amax.clone()

    us: Dict[str, float] = {}
    for backend in BACKENDS:
        runner = make_runner(
            backend, x, wgrad_rht, offsets, E, row_amax, col_amax, logical_packed_length
        )
        if runner is not None:
            us[backend] = kernel_time_us(runner)
    if not us:
        return None
    psl = E * tokens
    read_bytes = psl * dim * 2  # bfloat16 input
    col_write = dim * (psl // 2) + dim * (psl // 16)  # fp4 codes + fp8 scales
    row_write = psl * (dim // 2) + psl * (dim // 16)
    return ExperimentResult(us=us, total_bytes=read_bytes + col_write + row_write)


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
        gbps = (e.result.total_bytes / 1e9) / (ref / 1e6)
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
        raise RuntimeError("Grouped NVFP4 quantization requires SM100+")

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
