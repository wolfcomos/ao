# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Run the nine CuTeDSL V2 kernel benchmarks and compare fast-math variants.

Each measurement is CUDA kernel self-time, excluding host overhead and copies.
Inputs are regenerated from the same seed for each variant; non-fast/fast order
alternates between passes. Results are medians of independent profiler passes.
"""

import argparse
import importlib
import importlib.metadata
import json
import statistics
from dataclasses import asdict, fields
from pathlib import Path

import torch
from tabulate import tabulate

from benchmarks.prototype.nvfp4_training.bench_utils import (
    get_deepseek_v3_activation_shapes,
)
from benchmarks.prototype.nvfp4_training.deepseek_v3_shapes import (
    get_deepseek_v3_weight_shapes,
)

KERNELS = {
    "group_row_cast_quantize": "weight",
    "group_col_cast_requant_amax": "weight",
    "group_col_cast_requantize": "weight",
    "group_col_rht_requant_amax": "weight",
    "group_col_rht_requantize": "weight",
    "group_row_cast_col_rht_amax": "x",
    "group_row_cast_col_rht_quantize": "x",
    "group_row_rht_col_rht_amax": "dy",
    "group_row_rht_col_rht_quantize_ms_eden": "dy",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("debugmodel", "16B", "671B", "all"), default="all"
    )
    parser.add_argument("--kernel", choices=(*KERNELS, "all"), default="all")
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.experts, args.passes, args.iters) <= 0 or args.warmup < 0:
        parser.error(
            "experts, passes and iters must be positive; warmup must be nonnegative"
        )

    records = []
    for kernel, side in KERNELS.items():
        if args.kernel not in (kernel, "all"):
            continue
        module = importlib.import_module(
            f"benchmarks.prototype.nvfp4_training.bench_{kernel}"
        )
        shapes = (
            get_deepseek_v3_weight_shapes(factorized_experts=args.experts)
            if side == "weight"
            else get_deepseek_v3_activation_shapes(
                side, factorized_experts=args.experts
            )
        )
        for shape in shapes:
            if args.model not in (shape.model, "all"):
                continue
            kwargs = asdict(shape)
            config = module.ExperimentConfig(**kwargs)
            variants = (
                (False, True)
                if any(f.name == "use_fast_math" for f in fields(config))
                else (None,)
            )
            samples = {fast: [] for fast in variants}
            for trial in range(args.passes):
                for fast in variants[:: (-1 if trial % 2 else 1)]:
                    torch.manual_seed(123)
                    config = module.ExperimentConfig(
                        **kwargs,
                        **({"use_fast_math": fast} if len(variants) == 2 else {}),
                    )
                    result = module.run_experiment(
                        config, warmup=args.warmup, iters=args.iters
                    )
                    if result is None:
                        raise RuntimeError(f"No CuTeDSL result for {kernel}")
                    samples[fast].append(result.us["cutedsl"])
            baseline = statistics.median(samples[variants[0]])
            for fast, values in samples.items():
                median = statistics.median(values)
                record = {
                    "kernel": kernel,
                    **kwargs,
                    "use_fast_math": fast,
                    "samples_us": values,
                    "median_us": median,
                    "spread_pct": 100 * (max(values) - min(values)) / median,
                    "baseline_over_variant": baseline / median,
                }
                records.append(record)
                print(json.dumps(record), flush=True)
    if args.output:
        report = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "cutedsl": importlib.metadata.version("nvidia-cutlass-dsl"),
            "warmup": args.warmup,
            "iters": args.iters,
            "passes": args.passes,
            "metric": "CUDA kernel self-time per call; copies and host overhead excluded",
            "records": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        tabulate(
            [
                [
                    r["kernel"],
                    r["model"],
                    r["projection"],
                    "default"
                    if r["use_fast_math"] is None
                    else ("fast" if r["use_fast_math"] else "non-fast"),
                    f"{r['median_us']:.3f}",
                    f"{r['spread_pct']:.2f}",
                    f"{r['baseline_over_variant']:.2f}x",
                ]
                for r in records
            ],
            headers=[
                "kernel",
                "model",
                "projection",
                "variant",
                "us",
                "spread %",
                "baseline/variant",
            ],
        )
    )


if __name__ == "__main__":
    main()
