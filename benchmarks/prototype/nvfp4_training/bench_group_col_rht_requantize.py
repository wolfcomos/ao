# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark the grouped rotated columnwise weight requantize with CuTeDSL.

One launch over the packed forward weight of the (E, M, N) stack -- the rowwise FP4
codes and swizzled e4m3 block scales of ``bench_group_row_cast_quantize``'s op -- rebuilds
W_qdq per expert, rotates its transpose by R_n (a 128-point randomized Hadamard
transform, the dgrad signs) and requantizes it rowwise along the transposed axis against
the group amaxes of ``bench_group_col_rht_requant_amax``'s op: RTNE FP4 codes and
swizzled e4m3 block scales, the V2 backward's dgrad weight operand. Reports device
kernel time (see bench_utils.kernel_time_us) on the DeepSeek-V3 shapes.

    python -m benchmarks.prototype.nvfp4_training.bench_group_col_rht_requantize
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from tabulate import tabulate
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

BACKENDS = ("cutedsl",)

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


def _signs(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (128,), generator=generator, dtype=torch.int8)
    return (bits * 2 - 1).to(device)


def make_runner(
    backend: str,
    row_fp4_w: torch.Tensor,
    row_sf_w: torch.Tensor,
    global_amax: torch.Tensor,
    amax_rht_w_qdq_t: torch.Tensor,
    dgrad_rht: torch.Tensor,
    num_tensors: int,
) -> Optional[Callable[[], object]]:
    """No-arg callable running ``backend``'s grouped requantize op, or None if unavailable."""
    if backend == "cutedsl":
        if not cutedsl_nvfp4_kernels_available():
            return None
        from torchao.prototype.moe_training.nvfp4_training.group_col_rht_requantize_cutedsl import (
            cutedsl_group_col_rht_requantize as op,
        )
    else:
        raise ValueError(f"unknown backend {backend}")

    return lambda: op(
        row_fp4_w, row_sf_w, global_amax, amax_rht_w_qdq_t, dgrad_rht, num_tensors
    )


def run_experiment(
    config: ExperimentConfig, *, warmup: int = 15, iters: int = 50
) -> Optional[ExperimentResult]:
    if not cutedsl_nvfp4_kernels_available():
        raise RuntimeError("NVFP4 V2 benchmarks require SM100+ and the CuTeDSL runtime")
    E, M, N = config.experts, config.m, config.n
    weights = torch.randn((E, M, N), dtype=torch.bfloat16, device=device)
    global_amax = weights.float().abs().amax(dim=(1, 2))
    dgrad_rht = _signs(seed=0)
    # Prepare the packed forward weight outside the timed region.
    from torchao.prototype.moe_training.nvfp4_training.group_col_rht_requantize_cutedsl import (
        cutedsl_group_col_rht_requant_amax as amax_op,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_row_cast_quantize_cutedsl import (
        cutedsl_group_row_cast_quantize as cast_op,
    )

    row_fp4_w, row_sf_w = cast_op(weights, global_amax, E)
    amax_rht_w_qdq_t = amax_op(row_fp4_w, row_sf_w, global_amax, dgrad_rht, E)

    us: Dict[str, float] = {}
    for backend in BACKENDS:
        runner = make_runner(
            backend, row_fp4_w, row_sf_w, global_amax, amax_rht_w_qdq_t, dgrad_rht, E
        )
        if runner is not None:
            us[backend] = kernel_time_us(runner, warmup=warmup, iters=iters)
    if not us:
        return None

    elements = E * M * N
    # Packed FP4 codes (elements/2 bytes) and swizzled e4m3 scales (elements/16) in, the
    # same on the transposed axis out: 1.125 bytes per element.
    moved_bytes = 2 * (elements // 2 + elements // 16)
    return ExperimentResult(us=us, moved_bytes=moved_bytes)


def print_results(experiments: List[Experiment]) -> None:
    headers = [
        "model",
        "projection",
        "E",
        "M",
        "N",
        "cutedsl_us",
        "cutedsl_gbps",
    ]
    rows = []
    for e in experiments:
        us = e.result.us
        c = us["cutedsl"]
        ref = c
        gbps = (e.result.moved_bytes / 1e9) / (ref / 1e6)
        rows.append(
            [
                e.config.model,
                e.config.projection,
                e.config.experts,
                e.config.m,
                e.config.n,
                round(c, 3) if c else "n/a",
                round(gbps, 1),
            ]
        )
    print(tabulate(rows, headers=headers))


def main() -> None:
    if not torch.cuda.is_available() or not is_sm_at_least_100():
        raise RuntimeError("Grouped NVFP4 weight requantize requires SM100+")

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
