# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py

import itertools
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8 import (
    mxfp8_quantize_2d_1x32_cutedsl,
    mxfp8_quantize_2d_32x1_cutedsl,
    swiglu_mxfp8_quantize,
)

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000

SCALING_MODE = "rceil"

# E8M0 scales match the eager path bitwise, and forward E4M3 data does too. In
# the backward direction a handful of codes can land one apart, because the
# kernel evaluates sigmoid with rcp.approx plus Markstein refinement and fuses
# the d_silu product differently than eager PyTorch. Same bound as
# test/prototype/moe_training/test_cutedsl_swiglu_mxfp8.py, which measures the
# real rate at ~4e-7.
MAX_DIFFERING_FRACTION = 1e-5


def assert_matches_baseline(actual: torch.Tensor, expected: torch.Tensor, msg: str):
    """Check one fused output against the eager baseline before timing it."""
    assert actual.shape == expected.shape, (
        f"{msg}: shape {actual.shape} vs {expected.shape}"
    )
    assert actual.stride() == expected.stride(), (
        f"{msg}: stride {actual.stride()} vs {expected.stride()}"
    )
    assert actual.dtype == expected.dtype, (
        f"{msg}: dtype {actual.dtype} vs {expected.dtype}"
    )

    # A disabled direction is zero-sized on both sides, and torch.max() has no
    # identity to return for an empty reduction; matching shape, stride and
    # dtype is the whole contract there.
    if actual.numel() == 0:
        return

    # Compare as uint8 so this also works on the column-major (stride (1, M))
    # outputs without materializing a contiguous copy of a multi-GiB tensor.
    a, e = actual.view(torch.uint8), expected.view(torch.uint8)
    if actual.dtype == torch.float8_e8m0fnu:
        assert bool((a == e).all()), f"{msg}: scales are not bitwise identical"
        return

    gap = torch.maximum(a, e) - torch.minimum(a, e)
    max_gap = int(gap.max())
    assert max_gap <= 1, f"{msg}: max E4M3 code gap {max_gap} > 1"
    fraction = int((gap != 0).sum()) / a.numel()
    assert fraction <= MAX_DIFFERING_FRACTION, (
        f"{msg}: {fraction:.3e} of codes differ, limit {MAX_DIFFERING_FRACTION:.3e}"
    )


def swiglu(gated_input: torch.Tensor) -> torch.Tensor:
    """h = silu(gate) * up, computed in fp32 then rounded to bf16."""
    k = gated_input.shape[1] // 2
    gate = gated_input[:, :k].float()
    up = gated_input[:, k:].float()
    return (F.silu(gate) * up).bfloat16()


def swiglu_grads(grad_out: torch.Tensor, gated_input: torch.Tensor) -> torch.Tensor:
    """[dGate | dUp] concatenated into one [M, 2K] tensor, as the kernel emits it."""
    k = gated_input.shape[1] // 2
    gate = gated_input[:, :k].float()
    up = gated_input[:, k:].float()
    grad_out_f = grad_out.float()
    sigmoid_gate = torch.sigmoid(gate)
    d_silu = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    dgate = (grad_out_f * up * d_silu).bfloat16()
    dup = (grad_out_f * gate * sigmoid_gate).bfloat16()
    return torch.cat([dgate, dup], dim=1)


def baseline(gated_input, grad_h, rowwise, colwise):
    """Eager activation, then the standalone quantizers, in the fused API's order.

    Returns the same fixed four-tuple as swiglu_mxfp8_quantize, with zero-sized
    tensors for the disabled directions, so the two can be compared elementwise.
    """
    reference = (
        swiglu(gated_input) if grad_h is None else swiglu_grads(grad_h, gated_input)
    )
    empty_qdata = gated_input.new_empty(0, dtype=torch.float8_e4m3fn)
    empty_scales = gated_input.new_empty(0, dtype=torch.float8_e8m0fnu)

    if rowwise:
        output_rowwise, scales_rowwise = mxfp8_quantize_2d_1x32_cutedsl(
            reference, scaling_mode=SCALING_MODE
        )
    else:
        output_rowwise, scales_rowwise = empty_qdata, empty_scales

    if colwise:
        output_colwise, scales_colwise = mxfp8_quantize_2d_32x1_cutedsl(
            reference, scaling_mode=SCALING_MODE
        )
    else:
        output_colwise, scales_colwise = empty_qdata, empty_scales

    return output_rowwise, output_colwise, scales_rowwise, scales_colwise


def fused(gated_input, grad_h, rowwise, colwise):
    return swiglu_mxfp8_quantize(gated_input, grad_h, rowwise=rowwise, colwise=colwise)


# scaling_layout -> (rowwise, colwise)
LAYOUTS = {
    "rowwise": (True, False),
    "colwise": (False, True),
    "bidirectional": (True, True),
}


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: tuple[int, int]
    direction: str
    scaling_layout: str
    execution_mode: str


@dataclass(frozen=True)
class ExperimentResult:
    # time
    baseline_us: float
    fused_us: float
    # mem bw
    baseline_gbps: float
    fused_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    # M and K must both be multiples of 128. The kernel indexes gmem with 32-bit
    # math, so it also requires 2*K*M <= INT32_MAX; (131072, 8192) sits exactly
    # at that ceiling, which is why M stops there.
    m_values = (128, 256, 512, 1024, 4096, 8192, 16384, 131072)
    k_values = (2048, 7168, 8192)
    directions = ("forward", "backward")
    scaling_layouts = ("rowwise", "colwise", "bidirectional")
    execution_modes = ("eager", "compile")
    configs = []
    for m, k, direction, scaling_layout, execution_mode in itertools.product(
        m_values, k_values, directions, scaling_layouts, execution_modes
    ):
        configs.append(
            ExperimentConfig(
                input_shape=(m, k),
                direction=direction,
                scaling_layout=scaling_layout,
                execution_mode=execution_mode,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    M, K = config.input_shape
    gated_input = torch.randn(M, 2 * K, dtype=torch.bfloat16, device=device)
    grad_h = (
        torch.randn(M, K, dtype=torch.bfloat16, device=device)
        if config.direction == "backward"
        else None
    )
    rowwise, colwise = LAYOUTS[config.scaling_layout]
    args = (gated_input, grad_h, rowwise, colwise)

    if config.execution_mode == "compile":
        baseline_fn = torch.compile(baseline, fullgraph=True)
        fused_fn = torch.compile(fused, fullgraph=True)
    else:
        baseline_fn, fused_fn = baseline, fused

    try:
        # Verify before timing: a kernel that returned uninitialized memory would
        # otherwise report an excellent speedup.
        expected = baseline_fn(*args)
        actual = fused_fn(*args)
        for i, (actual_tensor, expected_tensor) in enumerate(zip(actual, expected)):
            assert_matches_baseline(
                actual_tensor,
                expected_tensor,
                f"{config.direction} {config.scaling_layout} M={M} K={K} output {i}",
            )

        baseline_time_us = benchmark_cuda_function_in_microseconds(baseline_fn, *args)
        fused_time_us = benchmark_cuda_function_in_microseconds(fused_fn, *args)
    finally:
        if config.execution_mode == "compile":
            torch._dynamo.reset()

    # Memory bandwidth calculations. Both implementations are scored against the
    # same ideal byte budget -- one pass over the inputs plus one pass over the
    # quantized outputs -- so the baseline's extra bfloat16 round trip through
    # the activation shows up as lower achieved bandwidth rather than as a
    # larger byte count.
    read_bytes = sum(
        t.numel() * t.element_size() for t in (gated_input, grad_h) if t is not None
    )
    write_bytes = sum(t.numel() * t.element_size() for t in actual)
    total_gb = (read_bytes + write_bytes) / 1e9

    return ExperimentResult(
        baseline_us=baseline_time_us,
        fused_us=fused_time_us,
        baseline_gbps=total_gb / (baseline_time_us / 1e6),
        fused_gbps=total_gb / (fused_time_us / 1e6),
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "input_shape",
        "direction",
        "scaling_layout",
        "execution_mode",
        "baseline_us",
        "fused_us",
        "speedup",
        "baseline_gbps",
        "fused_gbps",
    ]
    rows = []
    for experiment in experiments:
        speedup = experiment.result.baseline_us / experiment.result.fused_us
        rows.append(
            [
                str(experiment.config.input_shape),
                experiment.config.direction,
                experiment.config.scaling_layout,
                experiment.config.execution_mode,
                f"{experiment.result.baseline_us:.2f}",
                f"{experiment.result.fused_us:.2f}",
                f"{speedup:.2f}x",
                f"{experiment.result.baseline_gbps:.1f}",
                f"{experiment.result.fused_gbps:.1f}",
            ]
        )
    print(tabulate(rows, headers=headers))


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))
        # The largest shapes allocate ~70 GiB across the baseline's fp32
        # temporaries; release them before building the next config's inputs.
        torch.cuda.empty_cache()

    # Use Tabulate to print results
    print_results(results)


if __name__ == "__main__":
    main()
