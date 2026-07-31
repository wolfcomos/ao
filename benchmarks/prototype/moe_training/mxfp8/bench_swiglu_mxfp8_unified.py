# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
# this benchmarking script is a modified version of the original script from: https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/utils/benchmark.py
#
# The baseline is torchtitan's fused Triton SwiGLU, so torchtitan must be importable:
#   PYTHONPATH=.:/path/to/torchtitan python \
#     benchmarks/prototype/moe_training/mxfp8/bench_swiglu_mxfp8_unified.py [--compile]

import argparse
import itertools

import torch
import torch.nn.functional as F
from tabulate import tabulate
from torchtitan.overrides.fused_swiglu import (
    silu_and_mul_backward_op,
    silu_and_mul_op,
)

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.moe_training.kernels.mxfp8 import (
    mxfp8_quantize_2d_1x32_cutedsl,
    mxfp8_quantize_2d_32x1_cutedsl,
    swiglu_mxfp8_backward,
    swiglu_mxfp8_forward,
)

device = torch.device("cuda")

SCALING_MODE = "rceil"

# (M, K): DeepSeekV3 / Llama3 token counts against their FFN hidden sizes.
SHAPES = ((4096, 2048), (4096, 7168), (16384, 7168), (131072, 8192))

# name, rowwise, colwise
LAYOUTS = (("rowwise", True, False), ("colwise", False, True), ("both", True, True))


def baseline(gated_input, grad_h, rowwise, colwise):
    """torchtitan's fused Triton SwiGLU, then the standalone MXFP8 quantizers.

    This is the honest thing to beat: the activation is already fused, so the
    remaining win is removing the bfloat16 round trip between it and the cast.
    The backward concatenates [dGate | dUp] because the two Triton outputs are
    separate tensors, whereas the fused kernel writes that layout directly.
    """
    k = gated_input.shape[1] // 2
    gate, up = gated_input[:, :k], gated_input[:, k:]
    if grad_h is None:
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


def eager_reference(gated_input, grad_h, rowwise, colwise):
    """Ground truth for the correctness gate, in plain PyTorch.

    Not the timing baseline. torchtitan's Triton kernel contracts
    `1.0 + gate * (1.0 - sigmoid)` into an FMA, so it differs from eager in a
    few codes per million; the fused kernel matches eager exactly, and that is
    what we assert.
    """
    k = gated_input.shape[1] // 2
    gate, up = gated_input[:, :k].float(), gated_input[:, k:].float()
    if grad_h is None:
        reference = (F.silu(gate) * up).bfloat16()
    else:
        grad_h_f = grad_h.float()
        sig = torch.sigmoid(gate)
        d_silu = sig * (1.0 + gate * (1.0 - sig))
        # grad_h * (gate * sigmoid): the kernel forms silu first, and
        # grad_h * gate * sigmoid would associate the other way.
        reference = torch.cat(
            [
                (grad_h_f * up * d_silu).bfloat16(),
                (grad_h_f * (gate * sig)).bfloat16(),
            ],
            dim=1,
        )

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


def fused(gated_input, grad_h, rowwise, colwise):
    if grad_h is None:
        return swiglu_mxfp8_forward(gated_input, rowwise=rowwise, colwise=colwise)
    return swiglu_mxfp8_backward(grad_h, gated_input, rowwise=rowwise, colwise=colwise)


def check(actual, expected, msg):
    """The fused kernel is bitwise exact against the eager path."""
    assert actual.shape == expected.shape, f"{msg}: {actual.shape} vs {expected.shape}"
    assert actual.stride() == expected.stride(), f"{msg}: stride mismatch"
    a, e = actual.view(torch.uint8), expected.view(torch.uint8)
    assert bool((a == e).all()), f"{msg}: not bitwise identical"


def run(M, K, is_backward, rowwise, colwise, use_compile):
    gated_input = torch.randn(M, 2 * K, dtype=torch.bfloat16, device=device)
    grad_h = (
        torch.randn(M, K, dtype=torch.bfloat16, device=device) if is_backward else None
    )
    args = (gated_input, grad_h, rowwise, colwise)
    if use_compile:
        baseline_fn = torch.compile(baseline, fullgraph=True)
        fused_fn = torch.compile(fused, fullgraph=True)
    else:
        baseline_fn, fused_fn = baseline, fused

    try:
        actual = fused_fn(*args)
        for i, (a, e) in enumerate(zip(actual, eager_reference(*args))):
            check(a, e, f"M={M} K={K} output {i}")
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
    args = parser.parse_args()

    torch.random.manual_seed(123)
    rows = []
    for (M, K), is_backward, (layout, rowwise, colwise) in itertools.product(
        SHAPES, (False, True), LAYOUTS
    ):
        baseline_us, fused_us = run(M, K, is_backward, rowwise, colwise, args.compile)
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

    print(f"\nmode: {'compile' if args.compile else 'eager'}")
    headers = ["shape", "direction", "scales", "baseline_us", "fused_us", "speedup"]
    print(tabulate(rows, headers=headers))


if __name__ == "__main__":
    main()
