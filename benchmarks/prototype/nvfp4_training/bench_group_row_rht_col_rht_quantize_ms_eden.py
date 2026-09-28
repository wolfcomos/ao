# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark the grouped rowwise-RHT + columnwise-RHT MS-EDEN quantize kernel with CuTeDSL.

One launch over the packed gradient dy = (E * tokens, dim) bf16 quantizes per group
dy_g @ R_n (rowwise, dgrad signs) and dy_g^T @ R_m (columnwise, wgrad signs) with
independent 128-point randomized Hadamard transforms on both axes -- RTNE FP4 codes and
corrected, stochastically rounded E4M3 block scales against the group amaxes of
``bench_group_row_rht_col_rht_amax``'s op -- the V2 backward MS-EDEN operands. Reports
device kernel time (see bench_utils.kernel_time_us) on the DeepSeek-V3 shapes. ``--use-fast-math`` times the CuteDSL
op's ``FAST_PATH`` (hardware stochastic rounding, one Philox draw per 16 scales; a different
random stream from the default software path).

    python -m benchmarks.prototype.nvfp4_training.bench_group_row_rht_col_rht_quantize_ms_eden
    python -m benchmarks.prototype.nvfp4_training.bench_group_row_rht_col_rht_quantize_ms_eden --use-fast-math
"""

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from tabulate import tabulate
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

BACKENDS = ("cutedsl",)

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
    use_fast_math: bool = False


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
    dy: torch.Tensor,
    amax_rht_dy: torch.Tensor,
    amax_rht_dy_t: torch.Tensor,
    dgrad_rht: torch.Tensor,
    wgrad_rht: torch.Tensor,
    offsets: torch.Tensor,
    num_tensors: int,
    rng_state: torch.Tensor,
    logical_packed_length: torch.Tensor,
    use_fast_math: bool = False,
) -> Optional[Callable[[], object]]:
    """No-arg callable running ``backend``'s grouped MS-EDEN quantize op, or None if unavailable.
    ``use_fast_math`` selects hardware stochastic rounding of corrected scales."""
    psl, hidden = dy.shape
    kwargs = {}
    if backend == "cutedsl":
        if not cutedsl_nvfp4_kernels_available():
            return None
        from torchao.prototype.moe_training.nvfp4_training.group_row_rht_col_rht_quantize_ms_eden_cutedsl import (
            cutedsl_group_row_rht_col_rht_quantize_ms_eden as op,
        )

        kwargs = {"use_fast_math": use_fast_math}
    else:
        raise ValueError(f"unknown backend {backend}")

    return lambda: op(
        dy,
        amax_rht_dy,
        amax_rht_dy_t,
        dgrad_rht,
        wgrad_rht,
        offsets,
        num_tensors,
        psl,
        hidden,
        VARYING_FIRST_DIM,
        rng_state,
        logical_packed_length,
        **kwargs,
    )


def run_experiment(
    config: ExperimentConfig, *, warmup: int = 15, iters: int = 50
) -> Optional[ExperimentResult]:
    if not cutedsl_nvfp4_kernels_available():
        raise RuntimeError("NVFP4 V2 benchmarks require SM100+ and the CuTeDSL runtime")
    E, tokens, dim = config.experts, config.tokens, config.dim
    dy = torch.randn((E * tokens, dim), dtype=torch.bfloat16, device=device)
    dgrad_rht, wgrad_rht = _signs(seed=0), _signs(seed=1)
    offsets = torch.arange(1, E + 1, dtype=torch.int32, device=device) * tokens
    logical_packed_length = offsets[-1:]
    rng_state = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device=device)
    # Compute group amaxes once outside the timed quantization region.
    from torchao.prototype.moe_training.nvfp4_training.group_row_rht_col_rht_amax_cutedsl import (
        cutedsl_group_row_rht_col_rht_amax as amax_op,
    )

    amax_rht_dy, amax_rht_dy_t = amax_op(
        dy,
        dgrad_rht,
        wgrad_rht,
        offsets,
        E,
        E * tokens,
        dim,
        VARYING_FIRST_DIM,
        logical_packed_length,
    )

    us: Dict[str, float] = {}
    for backend in BACKENDS:
        runner = make_runner(
            backend,
            dy,
            amax_rht_dy,
            amax_rht_dy_t,
            dgrad_rht,
            wgrad_rht,
            offsets,
            E,
            rng_state,
            logical_packed_length,
            config.use_fast_math,
        )
        if runner is not None:
            us[backend] = kernel_time_us(runner, warmup=warmup, iters=iters)
    if not us:
        return None
    # bf16 in; FP4 codes and E4M3 scales out on both axes, 3.125 bytes per element.
    moved_bytes = dy.numel() * 2 + 2 * (dy.numel() // 2 + dy.numel() // 16)
    return ExperimentResult(us=us, moved_bytes=moved_bytes)


def print_results(experiments: List[Experiment]) -> None:
    headers = [
        "model",
        "projection",
        "E",
        "tokens",
        "dim",
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
                e.config.tokens,
                e.config.dim,
                round(c, 3) if c else "n/a",
                round(gbps, 1),
            ]
        )
    print(tabulate(rows, headers=headers))


def main() -> None:
    if not torch.cuda.is_available() or not is_sm_at_least_100():
        raise RuntimeError("Grouped NVFP4 MS-EDEN quantize requires SM100+")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-fast-math",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Time the CuteDSL op's FAST_PATH (hardware stochastic rounding, one Philox "
        "draw per 16 scales) in the cutedsl column instead of the default path.",
    )
    args = parser.parse_args()

    torch.random.manual_seed(123)
    configs = [
        ExperimentConfig(
            shape.experts,
            shape.tokens,
            shape.dim,
            model=shape.model,
            projection=shape.projection,
            use_fast_math=args.use_fast_math,
        )
        for shape in get_deepseek_v3_activation_shapes(
            "dy", factorized_experts=LOCAL_EXPERTS
        )
    ]
    experiments = []
    for config in tqdm(configs):
        result = run_experiment(config)
        if result is not None:
            experiments.append(Experiment(config=config, result=result))
    if args.use_fast_math:
        print(
            "cutedsl column: use_fast_math=True (hardware SR; different scale RNG from the default path)"
        )
    print_results(experiments)


if __name__ == "__main__":
    main()
