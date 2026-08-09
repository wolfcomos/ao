# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py
#
# The SwiGLU baseline is torchtitan's fused Triton op, so torchtitan must be importable:
#   PYTHONPATH=.:/path/to/torchtitan python \
#     benchmarks/prototype/moe_training/mxfp8/bench_swiglu_mxfp8_unified.py \
#       [--activation swiglu] [--compile]

import argparse
import itertools

import torch
from tabulate import tabulate
from torchtitan.overrides.fused_swiglu import (
    silu_and_mul_backward_op,
    silu_and_mul_op,
)

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8 import (
    gated_mxfp8_backward,
    gated_mxfp8_forward,
    mxfp8_quantize_2d_1x32_cutedsl,
    mxfp8_quantize_2d_32x1_cutedsl,
)

device = torch.device("cuda")

SCALING_MODE = "rceil"

# (M, K): DeepSeekV3 / Llama3 token counts against their FFN hidden sizes.
SHAPES = ((4096, 2048), (4096, 7168), (16384, 7168), (131072, 8192))

# name, rowwise, colwise
LAYOUTS = (("rowwise", True, False), ("colwise", False, True), ("both", True, True))
ACTIVATIONS = ("reglu", "geglu", "swiglu", "qgeglu", "sreglu")


def _activation_and_grad(x, activation):
    """Transformer Engine activation formulas evaluated in FP32."""
    if activation == "reglu":
        return torch.relu(x), (x > 0).float()
    if activation == "geglu":
        act_t = torch.tanh(x * (0.79788456 + 0.03567741 * x * x))
        act = x * (0.5 + 0.5 * act_t)
        grad_t = torch.tanh(0.79788456 * x * (1.0 + 0.044715 * x * x))
        dact = 0.5 * x * (
            (1.0 - grad_t * grad_t) * (0.79788456 + 0.1070322243 * x * x)
        ) + 0.5 * (1.0 + grad_t)
        return act, dact
    if activation == "swiglu":
        s = torch.sigmoid(x)
        return x * s, x * (s * (1.0 - s)) + s
    if activation == "qgeglu":
        ax = 1.702 * x
        s = torch.sigmoid(ax)
        return x * s, ax * (s * (1.0 - s)) + s
    if activation == "sreglu":
        relu = torch.relu(x)
        return relu * relu, torch.relu(2.0 * x)
    raise AssertionError(f"unknown activation {activation}")


def _eager_gated(gated_input, grad_h, activation):
    k = gated_input.shape[1] // 2
    gate = gated_input[:, :k].float()
    up = gated_input[:, k:].float()
    act, dact = _activation_and_grad(gate, activation)
    if grad_h is None:
        return (act * up).bfloat16()
    grad_h_f = grad_h.float()
    return torch.cat(
        [
            ((dact * grad_h_f) * up).bfloat16(),
            (act * grad_h_f).bfloat16(),
        ],
        dim=1,
    )


def baseline(gated_input, grad_h, activation, rowwise, colwise):
    """An activation baseline followed by the standalone MXFP8 quantizers.

    SwiGLU uses torchtitan's fused Triton implementation, making it the honest
    performance baseline. Other activations use the eager TE formulas and are
    useful for kernel timing, but their speedup is not an apples-to-apples fused
    baseline comparison.
    """
    k = gated_input.shape[1] // 2
    gate, up = gated_input[:, :k], gated_input[:, k:]
    if activation != "swiglu":
        reference = _eager_gated(gated_input, grad_h, activation)
    elif grad_h is None:
        reference = silu_and_mul_op(gate, up, None)
    else:
        grad_gate, grad_up = silu_and_mul_backward_op(grad_h, gate, up, None)
        reference = torch.cat([grad_gate, grad_up], dim=1)

    empty_qdata = gated_input.new_empty(0, dtype=torch.float8_e4m3fn)
    empty_scales = gated_input.new_empty(0, dtype=torch.float8_e8m0fnu)
    row = (
        mxfp8_quantize_2d_1x32_cutedsl(reference, scaling_mode=SCALING_MODE)
        if rowwise
        else (empty_qdata, empty_scales)
    )
    col = (
        mxfp8_quantize_2d_32x1_cutedsl(reference, scaling_mode=SCALING_MODE)
        if colwise
        else (empty_qdata, empty_scales)
    )
    return row[0], col[0], row[1], col[1]


def eager_reference(gated_input, grad_h, activation, rowwise, colwise):
    """Ground truth for the correctness gate, in plain PyTorch.

    Not the timing baseline, which is torchtitan's Triton kernel.

    Activation math is FP32 and rounded back to BF16 before the standalone
    quantizer, matching Transformer Engine's numerical contract.
    """
    reference = _eager_gated(gated_input, grad_h, activation)

    empty_qdata = gated_input.new_empty(0, dtype=torch.float8_e4m3fn)
    empty_scales = gated_input.new_empty(0, dtype=torch.float8_e8m0fnu)
    row = (
        mxfp8_quantize_2d_1x32_cutedsl(reference, scaling_mode=SCALING_MODE)
        if rowwise
        else (empty_qdata, empty_scales)
    )
    col = (
        mxfp8_quantize_2d_32x1_cutedsl(reference, scaling_mode=SCALING_MODE)
        if colwise
        else (empty_qdata, empty_scales)
    )
    return row[0], col[0], row[1], col[1]


def fused(gated_input, grad_h, activation, rowwise, colwise):
    if grad_h is None:
        return gated_mxfp8_forward(
            gated_input,
            activation=activation,
            rowwise=rowwise,
            colwise=colwise,
        )
    return gated_mxfp8_backward(
        grad_h,
        gated_input,
        activation=activation,
        rowwise=rowwise,
        colwise=colwise,
    )


# Backward E4M3 only; see the note in eager_reference.
MAX_DIFFERING_FRACTION = 1e-5


def check(actual, expected, msg, exact):
    """Scales and forward data are bitwise exact; backward data within one code."""
    assert actual.shape == expected.shape, f"{msg}: {actual.shape} vs {expected.shape}"
    assert actual.stride() == expected.stride(), f"{msg}: stride mismatch"
    a, e = actual.view(torch.uint8), expected.view(torch.uint8)
    if exact or actual.dtype == torch.float8_e8m0fnu:
        assert bool((a == e).all()), f"{msg}: not bitwise identical"
        return
    # A disabled direction is zero-sized on both sides, and torch.max() has no
    # identity to return for an empty reduction.
    if actual.numel() == 0:
        return
    gap = torch.maximum(a, e) - torch.minimum(a, e)
    assert int(gap.max()) <= 1, f"{msg}: max E4M3 code gap > 1"
    fraction = int((gap != 0).sum()) / a.numel()
    assert fraction <= MAX_DIFFERING_FRACTION, (
        f"{msg}: {fraction:.3e} of codes differ, limit {MAX_DIFFERING_FRACTION:.3e}"
    )


def run(M, K, is_backward, activation, rowwise, colwise, use_compile):
    gated_input = torch.randn(M, 2 * K, dtype=torch.bfloat16, device=device)
    grad_h = (
        torch.randn(M, K, dtype=torch.bfloat16, device=device) if is_backward else None
    )
    args = (gated_input, grad_h, activation, rowwise, colwise)
    if use_compile:
        baseline_fn = torch.compile(baseline, fullgraph=True)
        fused_fn = torch.compile(fused, fullgraph=True)
    else:
        baseline_fn, fused_fn = baseline, fused

    try:
        actual = fused_fn(*args)
        direction = "backward" if is_backward else "forward"
        for i, (a, e) in enumerate(zip(actual, eager_reference(*args))):
            check(a, e, f"M={M} K={K} {direction} output {i}", exact=not is_backward)
        baseline_us = benchmark_cuda_function_in_microseconds(baseline_fn, *args)
        fused_us = benchmark_cuda_function_in_microseconds(fused_fn, *args)
    finally:
        if use_compile:
            torch._dynamo.reset()
    return baseline_us, fused_us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compile",
        action="store_true",
        help="benchmark torch.compile(fullgraph=True) instead of eager",
    )
    parser.add_argument(
        "--activation",
        choices=ACTIVATIONS,
        default="swiglu",
        help="gated activation to benchmark (default: swiglu)",
    )
    args = parser.parse_args()

    torch.random.manual_seed(123)
    rows = []
    for (M, K), is_backward, (layout, rowwise, colwise) in itertools.product(
        SHAPES, (False, True), LAYOUTS
    ):
        baseline_us, fused_us = run(
            M,
            K,
            is_backward,
            args.activation,
            rowwise,
            colwise,
            args.compile,
        )
        rows.append(
            [
                f"({M}, {K})",
                "backward" if is_backward else "forward",
                layout,
                f"{baseline_us:.2f}",
                f"{fused_us:.2f}",
                f"{baseline_us / fused_us:.2f}x",
            ]
        )
        torch.cuda.empty_cache()

    baseline_name = (
        "torchtitan fused" if args.activation == "swiglu" else "PyTorch decomposition"
    )
    print(
        f"\nactivation: {args.activation}; mode: "
        f"{'compile' if args.compile else 'eager'}; baseline: {baseline_name}"
    )
    headers = ["shape", "direction", "scales", "baseline_us", "fused_us", "speedup"]
    print(tabulate(rows, headers=headers))


if __name__ == "__main__":
    main()
