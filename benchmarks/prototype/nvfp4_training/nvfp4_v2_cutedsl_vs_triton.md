# NVFP4 V2 grouped kernels: CuteDSL vs Triton

CuteDSL ports of the nine grouped Triton kernels behind `nvfp4_grouped_mm_v2` (the V2
recipe and the V1_REQUANT weight path), each measured against its Triton twin with the
benches in this directory. Compare numbers within this file only; the tables in
`README.md` come from another environment.

## Methodology

- NVIDIA GB200, 152 SMs, SM application clock capped at 1200 MHz for every table here
  (medians of three passes on 2026-09-16, GPUs 0-3 of one node, the tree of this file's
  commit). SM-bound kernels run about 1.6x slower than at the 1965 MHz maximum;
  memory-bound rows do not scale with the SM clock. Most of these kernels are SM-bound at
  this clock on both backends, so the speedup, not the absolute time, is the
  clock-portable number.
- Peak bandwidth 7936 GB/s (`torch.cuda.get_device_properties` on the GPU that produced
  the first tables; other GB200s on these nodes report 8064). Every GB/s and `pct_peak` is
  against 7936. The project's per-kernel target is 3000 GB/s at full clock.
- Toolchain of record: PyTorch 2.14.0a0, CUDA 13.x (pre-release), Triton 3.8.0, nvidia-cutlass-dsl
  4.8.0.dev0; every bitwise claim is against Triton 3.8.0. On Triton 3.6.0 / CUDA 13.4 /
  nvidia-cutlass-dsl 4.6.0 every CuteDSL kernel compiles and passes its single-backend
  tests, and the four `col_*` weight ops are bitwise against Triton 3.6.0 as well at every
  table shape. Triton 3.6.0 does not compile the two RHT-128 amax ops, so the
  cross-backend tests of the four RHT-128 activation and gradient ops run on 3.8.0 only;
  its MS-EDEN scale bytes differ from 3.8.0's at ~1e-6 of blocks.
- Time is `bench_utils.kernel_time_us` (15 warmups / 50 iterations, CUDA self-time,
  memcpy/memset excluded) summed over every kernel the custom op launches: op time. It
  includes the in-op torch glue an op issues (a sign-matrix build, i.e.
  `get_dynamic_rht_matrix` plus its int8-to-bfloat16 copy, 5-8 us at this clock; a
  `torch.zeros` fill, ~2 us) and excludes the 2-5 us `Memcpy DtoD` the four activation
  CuteDSL ops issue for `logical_packed_length` and the Philox state. Kernel-only ratios
  divide the Triton kernel's profiler self time (the 2026-09-14 record; those Triton
  kernels are unchanged since) by this refresh's CuteDSL op time less its fill.
- Another GPU of the node moves an op by up to ~4%. In 6 Triton cells of this refresh the
  three-pass spread exceeded 2% (up to 4.7% at 671B gate/up of `row_cast_col_rht_amax`);
  the medians stand. The timed loop re-runs one buffer with no L2 flush, so rows whose
  working set fits the 137 MB L2 (the weight tables at the debug model and 16B, the 671B
  `col_*` rows) read partly from cache, measured up to 10% at 6.5-33 MB and none at 117 MB
  and above: their GB/s is an upper bound and their speedup may move when cold.
- Shapes: `E = 4` local experts and the DeepSeek-V3 weights `w = (E, M, N)` of
  `deepseek_v3_shapes.py` (`M` rows, `N` columns; the recipe calls the stack `(E, N, K)`).
  Activations are `x = (E * tokens, dim = N)` and gradients `dy = (E * tokens, dim = M)`
  at the per-expert tokens per step of `bench_utils.py`: 16B 12288 from the ep-8 training layout,
  671B 32768 from the ep-32 layout, debugmodel 256. Debug-model rows are 16-tile
  launch-floor probes: latency ratios, not throughput.

## Results at a glance

Each range spans the gate/up and down rows of the named model; debug-model rows are in the
per-kernel tables only. Speedup is Triton op time over CuteDSL op time. GB/s is the
CuteDSL op at 671B against the 7936 GB/s peak, the fast variant's on its row. "codes
only": the FP4 codes are bitwise, the E4M3 scales come from a different stochastic stream
and are accepted by the distribution tests described under the kernel.

| kernel | bitwise vs Triton 3.8 | 16B speedup | 671B speedup | 671B GB/s |
|---|---|---:|---:|---:|
| row_cast_quantize | yes | 1.67-1.75x | 1.93-1.95x | 5580-5601 |
| row_rht_col_rht_amax | yes | 5.01-5.18x | 5.62-5.75x | 5270-5601 |
| row_rht_col_rht_quantize_ms_eden | yes | 2.61-2.64x | 2.63-2.64x | 2068-2108 |
| row_rht_col_rht_quantize_ms_eden, `fast_path=True` | codes only | 3.19-3.22x | 3.24-3.25x | 2546-2597 |
| col_rht_requant_amax | yes | 1.87-1.89x | 2.39-2.41x | 991-1000 |
| col_rht_requantize | yes | 1.85-1.88x | 2.06-2.29x | 1290-1441 |
| col_cast_requant_amax | yes | 2.20-2.21x | 4.58-4.78x | 3970-4135 |
| col_cast_requantize | yes | 5.00-5.25x | 5.62-5.65x | 1408-1411 |
| row_cast_col_rht_amax | yes | 4.31-4.54x | 4.97-5.13x | 5485-5846 |
| row_cast_col_rht_quantize | yes | 2.11-2.14x | 2.27-2.28x | 4075-4178 |

## End to end

An internal end-to-end comparison of the V2 recipe (DeepSeek-V3 671B, 16 nodes x 4 GB300,
ep-32, bs-8, compiled, unseeded; the cluster's own clocks, not this file's 1200 MHz frame)
ran `kernel_preference=cutedsl` against `triton` at the same torchtitan and torchao pin,
once on the kernels before the perf commits (0556473f1) and once on this record's
(f6f0e0999). The CuteDSL arm was the faster of the two in step TFLOP/s in both runs. The
nine ops are a small share of the step, so the `fast_path` variant of the MS-EDEN quantize
is projected to move step throughput by well under a percent, below what a single
cross-node run pair resolves; it was never run end to end (the V2 recipe has no `fast_path`
knob). Unseeded runs start the two arms from different weights, so loss is a band reading
only; the step-level bitwise check is the seeded 16B gate, not yet run. The per-run
numbers and raw traces are kept in an internal record.

## Kernels

### group_row_cast_quantize

`cutedsl_group_row_cast_quantize` vs `triton_group_row_cast_quantize`: the rowwise 1x16
weight quantize of the V1_REQUANT and V2 forward. Per expert, FP4 codes and swizzled E4M3
scales from the `(E,)` weight amax, RTNE, no RHT, no columnwise output (backward rebuilds
it with `group_col_cast_requantize` / `group_col_rht_requantize`). Each thread issues the
eight 16 B loads of half a 128-row block before quantizing it, and every 128-row block has
its own CTA. Codes and scales are bitwise identical across backends. Bytes: the bfloat16
read plus the codes and scales, 2.5625 per element.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_row_cast_quantize
```

| model | projection | E | M | N | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 3.79 | 3.98 | 1.05x | 177.1 |
| debugmodel | down (w2) | 4 | 256 | 256 | 3.79 | 3.98 | 1.05x | 177.2 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 7.09 | 11.83 | 1.67x | 4167.6 |
| 16B | down (w2) | 4 | 2048 | 1408 | 6.72 | 11.77 | 1.75x | 4399.0 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 26.96 | 52.16 | 1.93x | 5580.4 |
| 671B | down (w2) | 4 | 7168 | 2048 | 26.86 | 52.27 | 1.95x | 5601.0 |

- Op time is kernel time on both backends (no glue).
- Memory-bound at 671B: 70-71% of peak against Triton's 2.9 TB/s, about 5.6 TB/s of a
  measured 5.9-6.2 TB/s streaming ceiling; 16B and the debug model sit on the launch
  floor. REG 76, no spills, no shared memory, barriers or shuffles, six resident CTAs per
  SM; Triton's kernel holds 107 registers and reduces through shared memory.

### group_row_rht_col_rht_amax

`cutedsl_group_row_rht_col_rht_amax` vs `triton_group_row_rht_col_rht_amax`: the V2
backward gradient amax. Per expert, `max|dy_g @ R_n|` (rowwise, the dgrad signs) and
`max|dy_g.t() @ R_m|` (columnwise, the wgrad signs), two RHT-128 transforms with
independent sign vectors. One TMA load per 128x128 tile feeds two tcgen05 UMMA chains (8 +
8 UMMAs against two resident `R^T` tiles built in shared memory from the sign bytes). The
accumulators reduce from TMEM into a persistent accumulator with a last-CTA hand-off, so
the op issues no fill; that accumulator and its arrival ticket are one state per device,
so two launches of this op on different streams of one device would race on it. The amaxes
are bitwise identical across backends. Bytes: the bfloat16 read of `dy`.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_row_rht_col_rht_amax
```

| model | projection | E | tokens | dim | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 8.33 | 26.08 | 3.13x | 62.9 |
| debugmodel | down (w2) | 4 | 256 | 256 | 8.33 | 26.18 | 3.14x | 62.9 |
| 16B | gate/up (w1/w3) | 4 | 12288 | 1408 | 32.55 | 163.04 | 5.01x | 4251.6 |
| 16B | down (w2) | 4 | 12288 | 2048 | 43.76 | 226.80 | 5.18x | 4601.2 |
| 671B | gate/up (w1/w3) | 4 | 32768 | 2048 | 101.87 | 572.00 | 5.62x | 5270.3 |
| 671B | down (w2) | 4 | 32768 | 7168 | 335.47 | 1927.44 | 5.75x | 5601.3 |

- The CuteDSL op is its one kernel; the Triton op adds two sign-matrix builds and two
  fills, 15.5-19.3 us. Kernel-only 4.48-5.72x over the four large rows.
- 671B down: 5601 GB/s, 71% of peak. Tensor-bound at 1200 MHz: ~1.06k cycles per tile at
  the floor of its 16 UMMAs, after a 1.85 us launch. REG 50, no spills.

### group_row_rht_col_rht_quantize_ms_eden

`cutedsl_group_row_rht_col_rht_quantize_ms_eden` vs
`triton_group_row_rht_col_rht_quantize_ms_eden`: the V2 backward gradient quantize. Both
MS-EDEN operands, rowwise `ms_eden(dy_g @ R_n)` and columnwise `ms_eden(dy_g.t() @ R_m)`,
as RTNE FP4 codes plus a corrected, stochastically rounded E4M3 block scale (ceiling 256,
decode numerator 1536), each against its group's amax from `group_row_rht_col_rht_amax`
and its own Philox stream. Reuses the amax kernel's two-chain mainloop. Per 1x16 block the
quantize epilogues read TMEM and compute the pre-correction scale and codes, the two
16-term inner products of the unclamped scaled values with themselves and with the codes,
the correction, and the E4M3 rounding of the corrected scale from the Philox word Triton
draws. The Philox state is copied per launch into one persistent buffer per device, so
concurrent launches on two streams of one device would race on it as well. The default
path (`fast_path=False`) is bitwise identical to the Triton op on all four outputs. Bytes:
the bfloat16 read plus codes and scales on both axes, 3.125 per element.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_row_rht_col_rht_quantize_ms_eden --no-fast-path
python -m benchmarks.prototype.nvfp4_training.bench_group_row_rht_col_rht_quantize_ms_eden --fast-path
```

| model | projection | E | tokens | dim | cutedsl_us | fast_us | triton_us | speedup | fast_speedup | cutedsl_gbps | fast_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 11.36 | 10.64 | 32.48 | 2.86x | 3.05x | 72.1 | 77.0 |
| debugmodel | down (w2) | 4 | 256 | 256 | 11.36 | 10.62 | 32.38 | 2.85x | 3.05x | 72.1 | 77.1 |
| 16B | gate/up (w1/w3) | 4 | 12288 | 1408 | 113.01 | 92.90 | 298.77 | 2.64x | 3.22x | 1913.7 | 2328.0 |
| 16B | down (w2) | 4 | 12288 | 2048 | 160.33 | 131.28 | 419.08 | 2.61x | 3.19x | 1962.0 | 2396.1 |
| 671B | gate/up (w1/w3) | 4 | 32768 | 2048 | 405.63 | 329.50 | 1066.58 | 2.63x | 3.24x | 2068.0 | 2545.8 |
| 671B | down (w2) | 4 | 32768 | 7168 | 1392.65 | 1130.74 | 3673.81 | 2.64x | 3.25x | 2108.2 | 2596.5 |

- `cutedsl_us` and `speedup` are the default path, `fast_us` and `fast_speedup` the
  `fast_path=True` variant. The CuteDSL op is its one kernel (the signed `R^T` tiles are
  built in shared memory); the Triton op adds two sign-matrix builds, 12.0-15.1 us. Both
  kernels are SM-bound (27-33% of peak at 671B down). The default path runs one CTA per SM
  with an issue-limited epilogue at ~4.4k cycles per tile (two-point over the 671B rows)
  and the tensor pipe ~23% busy. REG 96, no spills, 3472 SASS instructions (the fast
  variant 3296); Triton 3.8's kernel holds 247 registers, no spills, 5576 SASS
  instructions.
- `fast_path=True` (default `False`; the recipe calls the Triton op and never reaches it)
  rounds the corrected scales in hardware: one `cvt.rs.satfinite.e4m3x4.f32` per warp-tile
  in place of four Philox rounds and four software roundings, for 18% less op time at 16B
  and 19% at 671B. It is a different stochastic stream, 16 random bits per value instead
  of 20, with the two values of a pair sharing one half-word, one of them bit-reversed.
  Its scale bytes are therefore not bitwise with Triton's: about a third differ by one
  E4M3 step, always the other neighbour of the same corrected scale, while the codes are
  identical. Accepted by distribution (`test_fast_path_*`): every scale lands on one of
  the reference's two E4M3 neighbours, and the round-up rate equals the fractional
  position within 0.01 per bin over 2^20 scales per axis. The tests also check
  unbiasedness over 64 draws, SQNR within 0.1 dB of the default path, determinism, and
  rng-slice isolation.
- 26c3f8116 feeds the unclamped scaled values into the correction on both backends (the
  clamped ones hid the saturated amax element on half the blocks, a one-sided ~0.3% shrink
  of the dequantized operand overall, ~0.7% on those blocks). Against the fe57e17d5 record
  it moved the CuteDSL default path -4% and the fast path -7% (3592 to 3472 and 3424 to
  3296 SASS instructions) and the Triton op +3%: its kernel keeps the unclamped tile live
  beside the clamped copy it packs, 5424 to 5576 SASS instructions and 243 to 247
  registers, no spills either way.
- 4496c9da9 speeds up the fast path only, inside its `fast_path` branch: the correction
  ratio through `div.approx.ftz.f32` instead of `div.full.f32`; the two 16-term inner
  products as fused `fma.rn.f32x2` chains instead of Triton's multiply-then-add tree; the
  codes decoded straight to bf16 pairs (`cvt.rn.bf16x2.e2m1x2`) and the cross product as
  `fma.rn.f32.bf16` of the bf16-exact scaled values, so the f16-to-f32 widening of the
  dequantized values is gone; and on the row chain one Philox draw per 16-scale group
  instead of one per tile. Codes unchanged; the corrected scale moves at float-rounding
  level ahead of the stochastic E4M3 rounding (no fast-path scale byte differs from the
  previous fast path at the recipe shapes; the `test_fast_path_*` items and the adversarial
  probe hold). Measured on a GB200 at 2062 MHz, paired three-pass medians (this table is
  at 1200 MHz): 671B down 704.5 to 636.2 us (-9.7%, 4168 to 4615 GB/s, 3.6x Triton), 671B
  gate/up -9.1%, 16B down -7.6%, 16B gate/up -6.7%; warp instructions -12%, the shared
  FMA-heavy pipe 63 to 44% busy, 3296 SASS instructions as before, REG 96 to 95, no
  spills. The default kernel's PTX and SASS are unchanged.

### group_col_rht_requant_amax

`cutedsl_group_col_rht_requant_amax` vs `triton_group_col_rht_requant_amax`: the V2
backward weight amax. Per expert, `max|W_qdq.t() @ R_n|` from the packed forward weight
(`group_row_cast_quantize`'s codes and scales). 256 producer threads stage each 128x128
tile's codes and scale words through a four-slot `cp.async` ring three tiles ahead and
dequantize them into a four-deep SW128 shared-memory ring (2688 numerator, bfloat16
rounding, the sign folded into the per-row decode scale). One MMA warp runs eight UMMAs
against the resident sign-free `H128` and four epilogue warps atomic-max into the zeroed
`(E,)`. The producer is `group_col_rht_requantize`'s, as the Triton twins share
`_load_rht_requant_weight_tile`. The amaxes are bitwise identical across backends. Bytes:
the codes and scales read, 0.5625 per weight element, plus the amax.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_col_rht_requant_amax
```

| model | projection | E | M | N | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 9.70 | 15.86 | 1.64x | 15.2 |
| debugmodel | down (w2) | 4 | 256 | 256 | 9.52 | 15.53 | 1.63x | 15.5 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 13.94 | 26.12 | 1.87x | 465.4 |
| 16B | down (w2) | 4 | 2048 | 1408 | 13.78 | 26.09 | 1.89x | 471.0 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 33.03 | 79.72 | 2.41x | 1000.0 |
| 671B | down (w2) | 4 | 7168 | 2048 | 33.33 | 79.73 | 2.39x | 991.1 |

- The CuteDSL op is its kernel plus one fill (1.7 us); the Triton op adds a sign-matrix
  build to the fill, 7.4 us. Kernel-only 1.50-2.28x over the four large rows.
- Issue-bound in the SIMT dequantize producer: ~1.2k cycles per tile against the 512-cycle
  floor of its eight UMMAs (tensor pipe ~42% active); 991-1000 GB/s at 671B, 12-13% of
  peak. The debug model pays the ring's three-tile prologue once per CTA. REG 38, no
  spills. Not shipped: a three-slot ring of its own, 1.04-1.10x faster on this op but
  slower on the requantize op that shares the producer; an in-kernel removal of the fill,
  bitwise, +2% at 671B.

### group_col_rht_requantize

`cutedsl_group_col_rht_requantize` vs `triton_group_col_rht_requantize`: the V2 backward
dgrad weight operand. Per expert, `W_qdq.t() @ R_n` requantized rowwise along the
transposed axis against the amaxes above, RTNE codes `(E, N, M//2)` and swizzled scales
`(E, N//128, M//64, 32, 16)`, 2688 numerator. Reuses the amax kernel's producer and chain
(accumulator zero-filled in TMEM per tile); eight epilogue warps run the RTNE 1x16
quantize on four blocks per warp per tile with the TMEM load ahead of the block chain.
Codes and scales are bitwise identical across backends (both ops fed the same amax).
Bytes: the codes and scales read and written, 1.125 per weight element.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_col_rht_requantize
```

| model | projection | E | M | N | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 8.75 | 13.50 | 1.54x | 33.7 |
| debugmodel | down (w2) | 4 | 256 | 256 | 8.74 | 13.41 | 1.53x | 33.7 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 15.32 | 28.30 | 1.85x | 847.2 |
| 16B | down (w2) | 4 | 2048 | 1408 | 15.06 | 28.33 | 1.88x | 861.6 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 45.83 | 105.11 | 2.29x | 1441.4 |
| 671B | down (w2) | 4 | 7168 | 2048 | 51.20 | 105.33 | 2.06x | 1290.3 |

- The CuteDSL op is its one kernel; the Triton op adds a sign-matrix build, 5.6-6.2 us.
  Kernel-only 1.47-2.17x over the four large rows.
- Issue-bound in the same producer with the RTNE epilogue on top: ~1.9k cycles per tile at
  671B gate/up and ~2.3k at 671B down against the 512-cycle floor; 1290-1441 GB/s at 671B,
  16-18% of peak. REG 94, no spills.

### group_col_cast_requant_amax

`cutedsl_group_col_cast_requant_amax` vs `triton_group_col_cast_requant_amax`: the
V1_REQUANT backward weight amax. Per expert, `max|bf16(W_qdq)|` from the packed forward
weight. A CUDA-core streaming reduction with no MMA, TMA or pipeline shared memory. A fast
pass keeps, per block, the scale of any block that contains a magnitude-7 code (whose
dequantized value is exactly `6 * |sf|`) and takes the max; a tile with a block lacking
such a code falls back to an exact pass over the CTA's tiles, never taken on
quantizer-produced weights. The per-expert decode multiply and the one bfloat16 rounding
happen once per CTA, which equals Triton's per-element rounding because `x -> bf16(x *
dec)` is monotone. A NaN or inf expert amax reports 0.0 as Triton's does. The amaxes are
bitwise identical across backends; a NaN scale byte, which the quantizer never emits, is
NaN on both with a different payload. Bytes: the codes and scales read, 0.5625 per weight
element, plus the amax.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_col_cast_requant_amax
```

| model | projection | E | M | N | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 4.45 | 6.04 | 1.36x | 33.1 |
| debugmodel | down (w2) | 4 | 256 | 256 | 4.46 | 6.20 | 1.39x | 33.0 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 5.33 | 11.77 | 2.21x | 1217.5 |
| 16B | down (w2) | 4 | 2048 | 1408 | 5.33 | 11.74 | 2.20x | 1218.0 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 8.32 | 38.12 | 4.58x | 3970.4 |
| 671B | down (w2) | 4 | 7168 | 2048 | 7.99 | 38.14 | 4.78x | 4135.5 |

- Both ops are one kernel plus the `torch.zeros((E,))` fill (1.8-2.0 us), 36% of the 16B
  op. Kernel-only (op less the fill on both sides) 2.9-6.0x over the four large rows.
- 671B: 3970-4135 GB/s, 50-52% of peak; less the fill and its ~1.85 us launch floor the
  kernel streams at ~7.2-7.8 TB/s, the memory floor of this access pattern. 16B: 1218
  GB/s, where the fill and the launch floor dominate; 6.5 MB cannot reach the 3000 GB/s
  target at any SM clock, since the launch floor plus the fill already exceed the 2.2 us
  such a rate allows. Uniform random code bytes, which take the exact pass, cost 17.1-17.5
  us at 671B, still 2.2x Triton. REG 33, no spills.

### group_col_cast_requantize

`cutedsl_group_col_cast_requantize` vs `triton_group_col_cast_requantize`: the V1_REQUANT
backward dgrad weight operand. `W_qdq.bf16().t()` requantized rowwise along the transposed
axis, RTNE codes `(E, N, M//2)` and swizzled scales, 2688 numerator. A CUDA-core
shared-memory transpose: a warp's coalesced loads rebuild eight weight rows into an
XOR-swizzled bfloat16 tile (272 B pitch, bank-conflict-free). Each thread then gathers the
1x16 blocks of four output rows and runs the RTNE 1x16 quantize, and the four lanes of a
row fill its 64 B of codes with one 16 B store each. The gather, not a tcgen05 identity
chain, preserves the sign of exact zeros, which keeps the contract bitwise. Codes and
scales are bitwise identical across backends (671B shapes, NaN / inf / zero / negative
expert amax, an all-`-0` expert, random code and scale bytes, degenerate `E` / `M` / `N =
0`, CUDA-graph capture); a NaN or inf expert amax yields `+0`. Bytes: the codes and scales
read and written, 1.125 per weight element.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_col_cast_requantize
```

| model | projection | E | M | N | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 4.96 | 12.01 | 2.42x | 59.5 |
| debugmodel | down (w2) | 4 | 256 | 256 | 4.96 | 12.00 | 2.42x | 59.4 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 11.23 | 56.14 | 5.00x | 1155.5 |
| 16B | down (w2) | 4 | 2048 | 1408 | 10.68 | 56.07 | 5.25x | 1214.5 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 46.91 | 263.84 | 5.62x | 1408.2 |
| 671B | down (w2) | 4 | 7168 | 2048 | 46.81 | 264.36 | 5.65x | 1411.4 |

- Op time is kernel time on both backends (no glue, no memcpy).
- Latency-bound SIMT transpose at five resident CTAs of 128 threads per SM (34 KB of
  shared memory each, under a `min_blocks_per_mp` launch bound): 1408-1411 GB/s at 671B,
  18% of peak. REG 94, no spills. Not shipped: a one-tile-ahead register prefetch, bitwise
  and a wash.

### group_row_cast_col_rht_amax

`cutedsl_group_rht_amax` vs `triton_group_rht_amax` at `dynamic_rht=True`: the V2 forward
activation amax. Per expert, the raw rowwise `max|x_g|` and `max|x_g.t() @ R|`, a
128-point randomized Hadamard transform along the tokens with the live wgrad sign buffer
(`sign_tensor`; the op's static RHT-16 path is unchanged). One TMA load per tile feeds a
tcgen05 chain (8 UMMAs against the resident `R^T`, its signs folded into the TMA-loaded
`H128` in shared memory, reduced from TMEM by four warps) and eight row warps reducing the
raw amax from the same stage with packed bfloat16 maxes. The amaxes are bitwise identical
across backends (64 groups, empty groups, spare capacity rows, int8 / bfloat16 / float32
sign buffers). Bytes: the bfloat16 read of `x`.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_row_cast_col_rht_amax
```

| model | projection | E | tokens | dim | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 8.28 | 18.96 | 2.29x | 63.3 |
| debugmodel | down (w2) | 4 | 256 | 256 | 8.29 | 18.93 | 2.28x | 63.3 |
| 16B | gate/up (w1/w3) | 4 | 12288 | 2048 | 42.71 | 194.07 | 4.54x | 4714.0 |
| 16B | down (w2) | 4 | 12288 | 1408 | 31.61 | 136.39 | 4.31x | 4378.9 |
| 671B | gate/up (w1/w3) | 4 | 32768 | 7168 | 321.43 | 1649.80 | 5.13x | 5846.0 |
| 671B | down (w2) | 4 | 32768 | 2048 | 97.87 | 486.16 | 4.97x | 5485.3 |

- The CuteDSL op is its kernel plus one fill (the column amax and a ready flag; CTA 0
  zeroes the row amax in-kernel), ~2 us; the Triton op adds a sign-matrix build to two
  fills, 9.7-12.8 us. Kernel-only 4.23-5.14x over the four large rows.
- 671B: 5485-5846 GB/s, 69-74% of peak, the HBM share of a tile now within reach of its
  8-UMMA chain and the row warps' shared-memory pass. REG 61, no spills.

### group_row_cast_col_rht_quantize

`cutedsl_group_rht_quantize_row_col` vs `triton_group_rht_quantize_row_col` at
`dynamic_rht=True`: the V2 forward activation quantize. Both NVFP4 operands, the raw rows
of `x_g` and the columns of `x_g.t() @ R`, as RTNE codes with E4M3 block scales against
the two group amaxes above. Reuses the amax kernel's mainloop; eight col warps quantize
the accumulator from TMEM under the requantize kernel's zero-fill discipline (an
exact-zero column sum keeps Triton's `+0` nibble), eight row warps run the RHT-16 fused
kernel's rowwise epilogue on the bfloat16 stage. All four outputs are bitwise identical
across backends, exact and fast math (Triton 3.6.0 too at the smoke shapes, both backends
fed the CuteDSL amaxes). Stochastic rounding is refused on this path (`ValueError`).
`use_fast_math=True`, the recipe default. Bytes: the bfloat16 read plus codes and scales
on both axes, 3.125 per element.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_row_cast_col_rht_quantize
```

| model | projection | E | tokens | dim | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 7.51 | 15.10 | 2.01x | 109.1 |
| debugmodel | down (w2) | 4 | 256 | 256 | 7.53 | 15.19 | 2.02x | 108.9 |
| 16B | gate/up (w1/w3) | 4 | 12288 | 2048 | 87.94 | 185.15 | 2.11x | 3577.3 |
| 16B | down (w2) | 4 | 12288 | 1408 | 61.53 | 131.77 | 2.14x | 3515.0 |
| 671B | gate/up (w1/w3) | 4 | 32768 | 7168 | 702.75 | 1602.31 | 2.28x | 4177.9 |
| 671B | down (w2) | 4 | 32768 | 2048 | 205.84 | 467.36 | 2.27x | 4075.2 |

- The CuteDSL op is its one kernel; the Triton op adds a sign-matrix build, 5.9-8.3 us.
  Kernel-only 2.01-2.27x over the four large rows.
- 671B: 4075-4178 GB/s, 51-53% of peak; the two quantize epilogues bound the kernel at
  this clock, not the UMMA chain or HBM. REG 76, no spills.

## Triton baseline for the nine grouped kernels

The Triton 3.6.0 and 3.8.0 op times for the nine ports: the activation rows were
re-measured on 2026-09-14 with the four checked-in benches at the methodology's token
counts, the weight rows are the earlier record of a script that is not checked in. One
launch per kernel over the `E = 4` expert stack, fed the inputs the recipe builds
(128-aligned token groups with cumulative row-end offsets, the `(E, M, N)` weight stack,
the packed `row_fp4_w` / `row_sf_w` plus the matching `*_requant_amax` output for the
requantizers). The per-kernel tables above carry the Triton 3.8.0 op times of their own
passes, which agree with this table within the pass-to-pass spread; the MS-EDEN rows here
predate 26c3f8116.

- Triton 3.6.0 does not compile the two RHT-128 amax ops
  (`TritonNvidiaGPUOptimizeTMemLayoutsPass`), so those rows are 3.8.0 only and the 3.6.0
  column's downstream quantizers were fed amaxes computed in torch with the same RHT-128
  chunking.

| model | projection | kernel | input | rounding | Triton 3.6.0 us | Triton 3.8.0 us | GB/s (3.8) | pct_peak (3.8) |
|---|---|---|---|---|---:|---:|---:|---:|
| 16B | gate/up (w1/w3) | row_cast_quantize | w (4, 1408, 2048) | rtne | 11.99 | 11.77 | 2511.0 | 31.64 |
| 16B | gate/up (w1/w3) | col_cast_requant_amax | w (4, 1408, 2048) | - | 13.59 | 11.68 | 555.7 | 7.00 |
| 16B | gate/up (w1/w3) | col_cast_requantize | w (4, 1408, 2048) | rtne | 82.39 | 56.19 | 230.9 | 2.91 |
| 16B | gate/up (w1/w3) | col_rht_requant_amax | w (4, 1408, 2048) | - | 25.93 | 25.82 | 251.3 | 3.17 |
| 16B | gate/up (w1/w3) | col_rht_requantize | w (4, 1408, 2048) | rtne | 27.04 | 28.09 | 462.0 | 5.82 |
| 16B | gate/up (w1/w3) | row_cast_col_rht_amax | x (49152, 2048) | - | n/a | 194.21 | 1036.7 | 13.06 |
| 16B | gate/up (w1/w3) | row_cast_col_rht_quantize | x (49152, 2048) | rtne | 210.76 | 184.91 | 1701.2 | 21.44 |
| 16B | gate/up (w1/w3) | row_rht_col_rht_amax | dy (49152, 1408) | - | n/a | 163.76 | 845.2 | 10.65 |
| 16B | gate/up (w1/w3) | row_rht_col_rht_quantize_ms_eden | dy (49152, 1408) | ms_eden | 309.63 | 289.51 | 747.0 | 9.41 |
| 16B | down (w2) | row_cast_quantize | w (4, 2048, 1408) | rtne | 11.96 | 11.71 | 2523.1 | 31.79 |
| 16B | down (w2) | col_cast_requant_amax | w (4, 2048, 1408) | - | 13.69 | 11.65 | 557.1 | 7.02 |
| 16B | down (w2) | col_cast_requantize | w (4, 2048, 1408) | rtne | 82.26 | 56.13 | 231.2 | 2.91 |
| 16B | down (w2) | col_rht_requant_amax | w (4, 2048, 1408) | - | 25.77 | 25.69 | 252.5 | 3.18 |
| 16B | down (w2) | col_rht_requantize | w (4, 2048, 1408) | rtne | 26.91 | 28.10 | 461.8 | 5.82 |
| 16B | down (w2) | row_cast_col_rht_amax | x (49152, 1408) | - | n/a | 136.31 | 1015.4 | 12.79 |
| 16B | down (w2) | row_cast_col_rht_quantize | x (49152, 1408) | rtne | 149.12 | 131.66 | 1642.7 | 20.70 |
| 16B | down (w2) | row_rht_col_rht_amax | dy (49152, 2048) | - | n/a | 227.81 | 883.8 | 11.14 |
| 16B | down (w2) | row_rht_col_rht_quantize_ms_eden | dy (49152, 2048) | ms_eden | 436.25 | 406.08 | 774.7 | 9.76 |
| 671B | gate/up (w1/w3) | row_cast_quantize | w (4, 2048, 7168) | rtne | 54.84 | 52.21 | 2881.8 | 36.31 |
| 671B | gate/up (w1/w3) | col_cast_requant_amax | w (4, 2048, 7168) | - | 47.56 | 38.19 | 865.0 | 10.90 |
| 671B | gate/up (w1/w3) | col_cast_requantize | w (4, 2048, 7168) | rtne | 390.59 | 265.34 | 249.0 | 3.14 |
| 671B | gate/up (w1/w3) | col_rht_requant_amax | w (4, 2048, 7168) | - | 77.86 | 79.42 | 415.9 | 5.24 |
| 671B | gate/up (w1/w3) | col_rht_requantize | w (4, 2048, 7168) | rtne | 90.33 | 105.54 | 626.0 | 7.89 |
| 671B | gate/up (w1/w3) | row_cast_col_rht_amax | x (131072, 7168) | - | n/a | 1648.42 | 1139.9 | 14.36 |
| 671B | gate/up (w1/w3) | row_cast_col_rht_quantize | x (131072, 7168) | rtne | 1866.77 | 1602.74 | 1831.9 | 23.08 |
| 671B | gate/up (w1/w3) | row_rht_col_rht_amax | dy (131072, 2048) | - | n/a | 575.00 | 933.7 | 11.77 |
| 671B | gate/up (w1/w3) | row_rht_col_rht_quantize_ms_eden | dy (131072, 2048) | ms_eden | 1114.48 | 1034.84 | 810.6 | 10.21 |
| 671B | down (w2) | row_cast_quantize | w (4, 7168, 2048) | rtne | 54.99 | 52.35 | 2874.3 | 36.22 |
| 671B | down (w2) | col_cast_requant_amax | w (4, 7168, 2048) | - | 47.59 | 38.18 | 865.2 | 10.90 |
| 671B | down (w2) | col_cast_requantize | w (4, 7168, 2048) | rtne | 390.08 | 266.77 | 247.6 | 3.12 |
| 671B | down (w2) | col_rht_requant_amax | w (4, 7168, 2048) | - | 77.92 | 79.40 | 416.0 | 5.24 |
| 671B | down (w2) | col_rht_requantize | w (4, 7168, 2048) | rtne | 90.54 | 106.01 | 623.2 | 7.85 |
| 671B | down (w2) | row_cast_col_rht_amax | x (131072, 2048) | - | n/a | 485.77 | 1105.2 | 13.93 |
| 671B | down (w2) | row_cast_col_rht_quantize | x (131072, 2048) | rtne | 537.75 | 466.62 | 1797.7 | 22.65 |
| 671B | down (w2) | row_rht_col_rht_amax | dy (131072, 7168) | - | n/a | 1935.77 | 970.7 | 12.23 |
| 671B | down (w2) | row_rht_col_rht_quantize_ms_eden | dy (131072, 7168) | ms_eden | 3868.53 | 3564.01 | 823.8 | 10.38 |

`row_cast_quantize` is the only one of the nine near bandwidth (31-36% of peak; the port
reaches 70-71%). `col_cast_requantize` runs 2-2.5x slower than its RHT twin on Triton
3.8.0 (3-4x on 3.6.0) at 2-3% of peak: 254 registers, 64 KB of shared memory and the tile
materialised three times, the largest headroom of the nine. The RHT-128 activation and
gradient kernels sit at 9-23% of peak, with 6-9 us of each op spent in the sign-matrix
build.
