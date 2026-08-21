# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Benchmark for the padded-EP-dispatch row permutation: the production eager
# forward/backward vs the improved torchao::_triton_permute_bwd scatter, with
# the pre-improvement kernel body resurrected verbatim for a head-to-head.
#
#   python benchmarks/prototype/moe_training/bench_triton_permute.py

import argparse

import torch
import triton
import triton.language as tl
from tabulate import tabulate

from torchao.prototype.moe_training.ep.kernels import generate_permute_indices
from torchao.prototype.moe_training.ep.permute import (
    _PermuteBF16,
    _triton_permute_bwd,
)

device = torch.device("cuda")


# ---------------------------------------------------------------------------
# The pre-improvement kernel body (torchao::_triton_permute_bwd at be381235),
# resurrected verbatim bench-side for the old-vs-improved head-to-head.
# Deltas the improvement made: 64-bit row arithmetic and a full-range
# destination validity mask (out-of-range treated like -1) -- semantics are
# identical for maps produced by generate_permute_indices.
# ---------------------------------------------------------------------------
@triton.jit
def _old_triton_permute_bwd_kernel(
    grad_ptr,
    permuted_indices_ptr,
    output_buffer_ptr,
    grad_rows,
    grad_cols,
    original_rows,
    original_cols,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    PADDING_VALUE: tl.constexpr = -1,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    row_offsets = row_pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = col_pid * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    dest_rows = tl.load(
        permuted_indices_ptr + row_offsets,
        mask=row_offsets < grad_rows,
        other=PADDING_VALUE,
    )

    read_mask = (row_offsets[:, None] < grad_rows) & (col_offsets[None, :] < grad_cols)
    grad_values = tl.load(
        grad_ptr + row_offsets[:, None] * grad_cols + col_offsets[None, :],
        mask=read_mask,
        other=PADDING_VALUE,
    )

    write_mask = (dest_rows[:, None] != PADDING_VALUE) & (
        col_offsets[None, :] < original_cols
    )
    tl.store(
        output_buffer_ptr + dest_rows[:, None] * original_cols + col_offsets[None, :],
        grad_values,
        mask=write_mask,
    )


def _old_triton_permute_bwd(grad_output, permuted_indices, original_rows):
    grad_rows, grad_cols = grad_output.shape
    output_buffer = grad_output.new_zeros((original_rows, grad_cols))
    grid = (triton.cdiv(grad_rows, 256), triton.cdiv(grad_cols, 256))
    _old_triton_permute_bwd_kernel[grid](
        grad_output,
        permuted_indices,
        output_buffer,
        grad_rows,
        grad_cols,
        original_rows,
        grad_cols,
        BLOCK_ROWS=256,
        BLOCK_COLS=256,
        PADDING_VALUE=-1,
    )
    return output_buffer


def _make_case(num_tokens, hidden, num_local_experts, ep_degree, alignment, skew):
    torch.manual_seed(0)
    num_groups = num_local_experts * ep_degree
    if skew:
        weights = torch.rand(num_groups) ** 3 + 0.01
        counts = (weights / weights.sum() * num_tokens).int()
        counts[-1] = num_tokens - counts[:-1].sum()
    else:
        counts = torch.full((num_groups,), num_tokens // num_groups).int()
        counts[-1] += num_tokens - int(counts.sum())
    counts = counts.to(device=device, dtype=torch.int32)
    # Mirror permute_and_pad's sizing: one aligned pad segment per LOCAL
    # expert (not per group), rounded up to the alignment.
    padded_len = num_tokens + num_local_experts * alignment
    max_len = (padded_len + alignment - 1) // alignment * alignment
    dst_to_src, _, _ = generate_permute_indices(
        counts, num_local_experts, ep_degree, max_len, alignment
    )
    x = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
    return x, dst_to_src


def _timed(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times = torch.tensor(times)
    return (
        times.quantile(0.5).item(),
        times.quantile(0.1).item(),
        times.quantile(0.9).item(),
    )


def _fmt(med, p10, p90):
    return f"{med:.3f} [{p10:.3f},{p90:.3f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    cases = [
        # (name, tokens, hidden, local_experts, ep, alignment, skew)
        ("dsv3_16b_ep4", 98304, 2048, 16, 4, 32, False),
        ("dsv3_16b_ep4_skewed", 98304, 2048, 16, 4, 32, True),
        ("small_ci", 4096, 512, 8, 1, 32, False),
    ]
    rows = []
    for name, tokens, hidden, e_local, ep, align, skew in cases:
        x, dst_to_src = _make_case(tokens, hidden, e_local, ep, align, skew)
        permuted = _PermuteBF16.apply(x, dst_to_src).detach()

        # Production-exact eager closures (what the pre-improvement code ran).
        def eager_fwd():
            x_padded = torch.vstack((x, x.new_zeros((1, hidden))))
            return x_padded[dst_to_src, :]

        def eager_inv():
            out_padded = permuted.new_zeros((tokens + 1, hidden))
            out_padded[dst_to_src, :] = permuted
            return out_padded[:-1]

        x_eager = x.detach().clone().requires_grad_(True)
        x_pair = x.detach().clone().requires_grad_(True)
        grad_seed = torch.randn_like(permuted)

        def eager_fwd_bwd():
            x_eager.grad = None
            xp = torch.vstack((x_eager, x_eager.new_zeros((1, hidden))))
            xp[dst_to_src, :].backward(grad_seed)

        def pair_fwd_bwd():
            x_pair.grad = None
            _PermuteBF16.apply(x_pair, dst_to_src).backward(grad_seed)

        def improved_scatter():
            return _triton_permute_bwd(permuted, dst_to_src, tokens, hidden)

        def old_scatter():
            return _old_triton_permute_bwd(permuted, dst_to_src, tokens)

        # Apples-to-apples sanity before timing.
        assert torch.equal(old_scatter(), improved_scatter())
        assert torch.equal(eager_inv(), improved_scatter())

        rows.append(
            {
                "case": name,
                "impl": "eager (production)",
                "fwd_ms": _fmt(*_timed(eager_fwd, iters=args.iters)),
                "inv_ms": _fmt(*_timed(eager_inv, iters=args.iters)),
                "fwd_bwd_ms": _fmt(*_timed(eager_fwd_bwd, iters=args.iters)),
            }
        )
        rows.append(
            {
                "case": name,
                "impl": "eagerF + triton scatterB",
                "fwd_ms": _fmt(
                    *_timed(lambda: _PermuteBF16.apply(x, dst_to_src), iters=args.iters)
                ),
                "inv_ms": _fmt(*_timed(improved_scatter, iters=args.iters)),
                "fwd_bwd_ms": _fmt(*_timed(pair_fwd_bwd, iters=args.iters)),
            }
        )
        rows.append(
            {
                "case": name,
                "impl": "old scatter kernel",
                "inv_ms": _fmt(*_timed(old_scatter, iters=args.iters)),
            }
        )

    print(tabulate(rows, headers="keys", tablefmt="github"))


if __name__ == "__main__":
    main()
