# NVFP4 V2 grouped kernels: CuteDSL vs Triton

Per-kernel CuteDSL ports of the grouped Triton kernels behind `nvfp4_grouped_mm_v2` (the
V2 recipe and the V1_REQUANT weight path), each measured against its Triton twin with the
scripts in this directory. The tables in `README.md` were measured on a different
environment, so compare numbers within this file only.

## Environment and methodology

- NVIDIA GB200, 152 SMs, SM application clock capped at 1200 MHz (`nvidia-smi`
  applications clock). Absolute times are roughly 1.6x those in `README.md` at equal
  bandwidth efficiency; ratios between shapes and CuteDSL-vs-Triton speedups are
  comparable. Peak bandwidth from device properties is 7936 GB/s (the bench scripts'
  formula); every `pct_peak` below is against it.
- Primary toolchain, used for every table: PyTorch 2.14.0a0, CUDA 13.x (pre-release), Triton 3.8.0,
  nvidia-cutlass-dsl 4.8.0.dev0. The Triton 3.6.0 column of the baseline table comes from
  PyTorch 2.14.0a0, CUDA 13.4, Triton 3.6.0, nvidia-cutlass-dsl 4.6.0; every CuteDSL
  kernel here compiles and is bitwise-checked on both.
- Device kernel time via `bench_utils.kernel_time_us` (15 warmups / 50 iterations, CUDA
  self-time, memcpy/memset excluded) -- op time for everything the custom op launches.
  `E = 4` local experts, DeepSeek-V3 shapes from `deepseek_v3_shapes.py`, medians of three
  full script passes.

## Kernels

### group_row_cast_quantize (`cutedsl_group_row_cast_quantize` vs `triton_group_row_cast_quantize`)

The rowwise 1x16 replacement for the grouped 2D weight quantize in the forward of the
V1_REQUANT and V2 recipes. Both backends consume the `(E,)` weight amax and emit, per
expert, rowwise FP4 codes and swizzled e4m3 scales only: every row of 16 gets its own
scale instead of one per 16x16 tile, and there is no columnwise output -- the dgrad
operand is rebuilt in backward from these codes by `group_col_cast_requantize` or
`group_col_rht_requantize`. RTNE, no RHT; the two backends produce bitwise identical
output, codes and scale factors. Bandwidth counts the bfloat16 read plus the rowwise FP4
codes and swizzled scales, 2.5625 bytes per element against the 2D kernel's 3.125.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_row_cast_quantize
```

| model | projection | E | M | N | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 11.10 | 8.80 | 0.79x | 47.2 |
| debugmodel | down (w2) | 4 | 256 | 256 | 11.09 | 8.77 | 0.79x | 47.3 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 10.20 | 18.10 | 1.77x | 2261.7 |
| 16B | down (w2) | 4 | 2048 | 1408 | 9.82 | 18.12 | 1.85x | 2348.3 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 22.95 | 39.91 | 1.74x | 5117.1 |
| 671B | down (w2) | 4 | 7168 | 2048 | 22.42 | 40.54 | 1.81x | 5237.5 |

CuteDSL wins 1.6-1.7x at 16B and 671B and loses at the debug model (0.74x). At 671B it
sustains 4.7 TB/s, 59% of peak, against Triton's 2.9 TB/s. Both kernels read `A` with 16-B
loads and write codes with 8-B stores; the CuteDSL kernel holds 48 registers with no shared
memory, barriers or shuffles (each thread stores its swizzled scale byte directly), where
the Triton kernel at `num_warps = 8` holds 107 registers and reduces through shared memory,
so it keeps roughly 2.5x the warps resident. The debug model launches 16 CTAs on either
backend and is latency-bound.

## Triton baseline for the nine grouped kernels

The targets for the remaining ports: the five V2 ops (`row_cast_quantize`,
`row_cast_col_rht_amax` / `_quantize` at RHT-128, `row_rht_col_rht_amax` /
`_quantize_ms_eden`) and the four V1_REQUANT columnwise requantizers
(`col_cast_requant_amax` / `_requantize`, `col_rht_requant_amax` / `_requantize`), one
launch per kernel over the local expert stack at `E = 4`, fed the inputs the recipe
builds: 128-aligned uniform token groups with cumulative row-end offsets for the
token-jagged ops, the `(E, N, K)` weight stack for the expert-uniform ones, and the packed
`row_fp4_w` / `row_sf_w` plus the matching `*_requant_amax` output for the requantizers.
Tokens per expert equal the weight row count, as in `bench_group_rht_quantize_row_col`.
The script that produced this table is not checked in; the table is the record.

- Op time, not kernel-only time: the RHT-128 ops build their rotation matrix in torch
  (`get_dynamic_rht_matrix`, 6-9 us at this clock) and the amax ops zero-fill their output
  (about 2 us) inside the timed region. Compare a port of one of these kernels on
  kernel-only time, or make it build the RHT matrix the same way.
- Bytes are the tensors each op reads and writes.
- Triton 3.6.0 fails to compile the two RHT-128 amax ops
  (`TritonNvidiaGPUOptimizeTMemLayoutsPass`: `parent layout must have at least rank >= 2`),
  so those two rows are Triton 3.8 only, and in the 3.6 column the quantizers downstream
  of them were fed amaxes computed in torch with the same RHT-128 chunking.
- `row_cast_col_rht_quantize` runs with `use_fast_math=True`, the recipe default. Its
  `rs` row at RHT-128 is synthetic: the recipe's stochastic-rounding path is the
  V1_REQUANT backward at RHT-16.

`w` is the weight `(E, N, K)`, `x` the packed activation `(E * tokens, K)` and `dy` the
packed gradient `(E * tokens, N)`; `tokens` per expert equals `N`.

| model | projection | kernel | input | rounding | Triton 3.6.0 us | Triton 3.8.0 us | GB/s (3.8) | pct_peak (3.8) |
|---|---|---|---|---|---:|---:|---:|---:|
| 16B | gate/up (w1/w3) | row_cast_quantize | w (4, 1408, 2048) | rtne | 11.99 | 11.77 | 2511.0 | 31.64 |
| 16B | gate/up (w1/w3) | col_cast_requant_amax | w (4, 1408, 2048) | - | 13.59 | 11.68 | 555.7 | 7.00 |
| 16B | gate/up (w1/w3) | col_cast_requantize | w (4, 1408, 2048) | rtne | 82.39 | 56.19 | 230.9 | 2.91 |
| 16B | gate/up (w1/w3) | col_rht_requant_amax | w (4, 1408, 2048) | - | 25.93 | 25.82 | 251.3 | 3.17 |
| 16B | gate/up (w1/w3) | col_rht_requantize | w (4, 1408, 2048) | rtne | 27.04 | 28.09 | 462.0 | 5.82 |
| 16B | gate/up (w1/w3) | row_cast_col_rht_amax | x (5632, 2048) | - | n/a | 35.93 | 642.0 | 8.09 |
| 16B | gate/up (w1/w3) | row_cast_col_rht_quantize | x (5632, 2048) | rtne | 33.07 | 30.74 | 1172.7 | 14.78 |
| 16B | gate/up (w1/w3) | row_cast_col_rht_quantize | x (5632, 2048) | rs | 70.35 | 59.60 | 604.8 | 7.62 |
| 16B | gate/up (w1/w3) | row_rht_col_rht_amax | dy (5632, 1408) | - | n/a | 35.59 | 445.6 | 5.61 |
| 16B | gate/up (w1/w3) | row_rht_col_rht_quantize_ms_eden | dy (5632, 1408) | ms_eden | 50.54 | 51.39 | 482.2 | 6.08 |
| 16B | down (w2) | row_cast_quantize | w (4, 2048, 1408) | rtne | 11.96 | 11.71 | 2523.1 | 31.79 |
| 16B | down (w2) | col_cast_requant_amax | w (4, 2048, 1408) | - | 13.69 | 11.65 | 557.1 | 7.02 |
| 16B | down (w2) | col_cast_requantize | w (4, 2048, 1408) | rtne | 82.26 | 56.13 | 231.2 | 2.91 |
| 16B | down (w2) | col_rht_requant_amax | w (4, 2048, 1408) | - | 25.77 | 25.69 | 252.5 | 3.18 |
| 16B | down (w2) | col_rht_requantize | w (4, 2048, 1408) | rtne | 26.91 | 28.10 | 461.8 | 5.82 |
| 16B | down (w2) | row_cast_col_rht_amax | x (8192, 1408) | - | n/a | 35.90 | 642.6 | 8.10 |
| 16B | down (w2) | row_cast_col_rht_quantize | x (8192, 1408) | rtne | 33.03 | 30.67 | 1175.2 | 14.81 |
| 16B | down (w2) | row_cast_col_rht_quantize | x (8192, 1408) | rs | 70.65 | 59.60 | 604.8 | 7.62 |
| 16B | down (w2) | row_rht_col_rht_amax | dy (8192, 2048) | - | n/a | 54.33 | 617.6 | 7.78 |
| 16B | down (w2) | row_rht_col_rht_quantize_ms_eden | dy (8192, 2048) | ms_eden | 84.62 | 86.60 | 605.4 | 7.63 |
| 671B | gate/up (w1/w3) | row_cast_quantize | w (4, 2048, 7168) | rtne | 54.84 | 52.21 | 2881.8 | 36.31 |
| 671B | gate/up (w1/w3) | col_cast_requant_amax | w (4, 2048, 7168) | - | 47.56 | 38.19 | 865.0 | 10.90 |
| 671B | gate/up (w1/w3) | col_cast_requantize | w (4, 2048, 7168) | rtne | 390.59 | 265.34 | 249.0 | 3.14 |
| 671B | gate/up (w1/w3) | col_rht_requant_amax | w (4, 2048, 7168) | - | 77.86 | 79.42 | 415.9 | 5.24 |
| 671B | gate/up (w1/w3) | col_rht_requantize | w (4, 2048, 7168) | rtne | 90.33 | 105.54 | 626.0 | 7.89 |
| 671B | gate/up (w1/w3) | row_cast_col_rht_amax | x (8192, 7168) | - | n/a | 119.12 | 985.9 | 12.42 |
| 671B | gate/up (w1/w3) | row_cast_col_rht_quantize | x (8192, 7168) | rtne | 128.54 | 112.92 | 1625.0 | 20.48 |
| 671B | gate/up (w1/w3) | row_cast_col_rht_quantize | x (8192, 7168) | rs | 286.14 | 246.95 | 743.1 | 9.36 |
| 671B | gate/up (w1/w3) | row_rht_col_rht_amax | dy (8192, 2048) | - | n/a | 54.45 | 616.3 | 7.77 |
| 671B | gate/up (w1/w3) | row_rht_col_rht_quantize_ms_eden | dy (8192, 2048) | ms_eden | 84.60 | 86.66 | 605.0 | 7.62 |
| 671B | down (w2) | row_cast_quantize | w (4, 7168, 2048) | rtne | 54.99 | 52.35 | 2874.3 | 36.22 |
| 671B | down (w2) | col_cast_requant_amax | w (4, 7168, 2048) | - | 47.59 | 38.18 | 865.2 | 10.90 |
| 671B | down (w2) | col_cast_requantize | w (4, 7168, 2048) | rtne | 390.08 | 266.77 | 247.6 | 3.12 |
| 671B | down (w2) | col_rht_requant_amax | w (4, 7168, 2048) | - | 77.92 | 79.40 | 416.0 | 5.24 |
| 671B | down (w2) | col_rht_requantize | w (4, 7168, 2048) | rtne | 90.54 | 106.01 | 623.2 | 7.85 |
| 671B | down (w2) | row_cast_col_rht_amax | x (28672, 2048) | - | n/a | 119.37 | 983.8 | 12.40 |
| 671B | down (w2) | row_cast_col_rht_quantize | x (28672, 2048) | rtne | 127.63 | 112.59 | 1629.9 | 20.54 |
| 671B | down (w2) | row_cast_col_rht_quantize | x (28672, 2048) | rs | 284.94 | 248.34 | 738.9 | 9.31 |
| 671B | down (w2) | row_rht_col_rht_amax | dy (28672, 7168) | - | n/a | 440.12 | 933.9 | 11.77 |
| 671B | down (w2) | row_rht_col_rht_quantize_ms_eden | dy (28672, 7168) | ms_eden | 865.93 | 801.36 | 801.5 | 10.10 |

`row_cast_quantize` is the only one of the nine near bandwidth (31-36% of peak; the
CuteDSL port above reaches 59%). The no-transform `col_cast_requantize` runs 3-4x slower
than its RHT twin `col_rht_requantize` at 2-3% of peak: the compiled kernel holds 254
registers and 64 KB of shared memory and materialises its 128x128 tile three times (once
per layout conversion), so it is neither memory- nor occupancy-bound; that is the largest
headroom of the nine. The RHT-128 activation and gradient kernels sit at 8-20% of peak,
with 6-9 us of each op spent building the rotation matrix in torch.
