# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark the grouped lazy columnwise weight requant amax across backends (triton, cutedsl).

One launch over the packed forward weight of the (E, M, N) stack -- the rowwise FP4
codes and swizzled e4m3 block scales of ``bench_group_row_cast_quantize``'s op -- rebuilds
W_qdq per expert and returns the per-expert amax of |W_qdq.bf16()|: the amax the
V1_REQUANT backward's dgrad weight operand is requantized against. No sign vector.
Reports device kernel time (see bench_utils.kernel_time_us) for each available backend
on the DeepSeek-V3 shapes, with the cutedsl-vs-triton speedup.

    python -m benchmarks.prototype.nvfp4_training.bench_group_col_cast_requant_amax
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from tabulate import tabulate
from torch.utils._triton import has_triton
from tqdm import tqdm

from benchmarks.prototype.nvfp4_training.bench_utils import kernel_time_us
from benchmarks.prototype.nvfp4_training.deepseek_v3_shapes import (
    get_deepseek_v3_weight_shapes,
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
    m: int
    n: int
    model: str = ""
    projection: str = ""


@dataclass(frozen=True)
class ExperimentResult:
    us: Dict[str, float]  # backend -> device kernel time (us)
    moved_bytes: int


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def make_runner(
    backend: str,
    row_fp4_w: torch.Tensor,
    row_sf_w: torch.Tensor,
    global_amax: torch.Tensor,
    num_tensors: int,
) -> Optional[Callable[[], object]]:
    """No-arg callable running ``backend``'s grouped requant amax op, or None if unavailable."""
    if backend == "triton":
        if not has_triton():
            return None
        from torchao.prototype.moe_training.nvfp4_training.group_col_cast_requantize_triton import (
            triton_group_col_cast_requant_amax as op,
        )
    elif backend == "cutedsl":
        if not cutedsl_nvfp4_kernels_available():
            return None
        from torchao.prototype.moe_training.nvfp4_training.group_col_cast_requantize_cutedsl import (
            cutedsl_group_col_cast_requant_amax as op,
        )
    else:
        raise ValueError(f"unknown backend {backend}")

    return lambda: op(row_fp4_w, row_sf_w, global_amax, num_tensors)


def run_experiment(config: ExperimentConfig) -> Optional[ExperimentResult]:
    E, M, N = config.experts, config.m, config.n
    weights = torch.randn((E, M, N), dtype=torch.bfloat16, device=device)
    global_amax = weights.float().abs().amax(dim=(1, 2))
    # The rowwise codes and scales are inputs, computed once outside the timed region;
    # the two backends' row-cast ops are bitwise identical, so either feeds both amax ops.
    if cutedsl_nvfp4_kernels_available():
        from torchao.prototype.moe_training.nvfp4_training.group_row_cast_quantize_cutedsl import (
            cutedsl_group_row_cast_quantize as cast_op,
        )
    else:
        from torchao.prototype.moe_training.nvfp4_training.group_row_cast_quantize_triton import (
            triton_group_row_cast_quantize as cast_op,
        )
    row_fp4_w, row_sf_w = cast_op(weights, global_amax, E)

    us: Dict[str, float] = {}
    for backend in BACKENDS:
        runner = make_runner(backend, row_fp4_w, row_sf_w, global_amax, E)
        if runner is not None:
            us[backend] = kernel_time_us(runner)
    if not us:
        return None

    elements = E * M * N
    # Packed FP4 codes (elements/2 bytes) and swizzled e4m3 scales (elements/16) in, the
    # (E,) float32 amax out.
    moved_bytes = elements // 2 + elements // 16 + E * 4
    return ExperimentResult(us=us, moved_bytes=moved_bytes)


def print_results(experiments: List[Experiment]) -> None:
    headers = [
        "model",
        "projection",
        "E",
        "M",
        "N",
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
        gbps = (e.result.moved_bytes / 1e9) / (ref / 1e6)
        rows.append(
            [
                e.config.model,
                e.config.projection,
                e.config.experts,
                e.config.m,
                e.config.n,
                round(c, 3) if c else "n/a",
                round(t, 3) if t else "n/a",
                speedup,
                round(gbps, 1),
            ]
        )
    print(tabulate(rows, headers=headers))


def main() -> None:
    if not torch.cuda.is_available() or not is_sm_at_least_100():
        raise RuntimeError("Grouped NVFP4 weight requant amax requires SM100+")

    torch.random.manual_seed(123)
    configs = [
        ExperimentConfig(
            shape.experts,
            shape.m,
            shape.n,
            model=shape.model,
            projection=shape.projection,
        )
        for shape in get_deepseek_v3_weight_shapes(factorized_experts=LOCAL_EXPERTS)
    ]
    experiments = []
    for config in tqdm(configs):
        result = run_experiment(config)
        if result is not None:
            experiments.append(Experiment(config=config, result=result))
    print_results(experiments)


if __name__ == "__main__":
    main()
