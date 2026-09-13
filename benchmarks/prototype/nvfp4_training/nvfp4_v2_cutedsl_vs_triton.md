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

### group_row_rht_col_rht_amax (`cutedsl_group_row_rht_col_rht_amax` vs `triton_group_row_rht_col_rht_amax`)

The V2 backward gradient amax: one pass over the packed gradient
`dy = (E * tokens, hidden)` returns, per expert, the amax of `|dy_g @ R_n|` (rowwise, the
dgrad signs) and of `|dy_g.t() @ R_m|` (columnwise, the wgrad signs) -- two RHT-128
transforms with independent sign vectors. The CuteDSL kernel loads every 128x128 tile into
shared memory once (TMA) and applies both transforms to that one copy through two tcgen05
UMMA chains (8 + 8 UMMAs against two resident `R^T` operand tiles; the row chain reads the
same bytes through a K-major view), reducing both per-group amaxes from TMEM in the same
pass. The two backends produce bitwise identical amaxes (`torch.equal` at every shape
below). Bandwidth counts the bfloat16 read of `dy`; the `2E` scalar outputs are not
counted.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_row_rht_col_rht_amax
```

| model | projection | E | tokens | hidden | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 20.31 | 26.67 | 1.31x | 25.8 |
| debugmodel | down (w2) | 4 | 256 | 256 | 20.30 | 26.74 | 1.32x | 25.8 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 1408 | 23.17 | 37.07 | 1.60x | 684.4 |
| 16B | down (w2) | 4 | 2048 | 2048 | 25.93 | 55.74 | 2.15x | 1293.9 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 2048 | 25.94 | 55.74 | 2.15x | 1293.7 |
| 671B | down (w2) | 4 | 7168 | 7168 | 98.14 | 442.72 | 4.51x | 4188.3 |

- Op time includes the same torch glue on both backends -- two sign-to-matrix builds and
  two `torch.zeros` fills -- 14.10-15.60 us for CuteDSL and 15.78-19.28 us for Triton. At
  671B gate/up, `dy (8192, 2048)`: CuteDSL two `torch.mul(h128, sign[None, :])` 10.23 us
  (the int8 sign operand puts them on ATen's casting `gpu_kernel_impl`; with bfloat16
  signs the pair takes 6.77 us) + two fills 3.94 us; Triton two `get_dynamic_rht_matrix`
  7.01 us + their two int8->bfloat16 copies 4.93 us + two fills 3.98 us. Kernel-only
  (profiler self CUDA time of the main kernel), CuteDSL vs Triton: 8.78 vs 21.10 us at 16B
  gate/up (2.40x), 11.55 vs 39.82 us at 16B down (3.45x), 11.55 vs 38.85 us at 671B
  gate/up (3.36x), 82.48 vs 423.51 us at 671B down (5.13x). The metric excludes the
  1.99-2.44 us `Memcpy DtoD` of the CuteDSL impl's `logical_packed_length.clone()`, which
  the Triton op does not issue.
- The 671B gate: gate/up `dy (8192, 2048)` 25.94 us against <= 22 us -- over by 3.94 us
  (17.9%); down `dy (28672, 7168)` 98.14 us against <= 105 us -- under by 6.86 us (6.5%).
  At gate/up the kernel is 11.55 us, under the gate on its own, and the glue 14.18 us,
  55% of the op: the miss is the glue, the two int8-sign `torch.mul` at 5.1 us each; the
  kernel's fixed floor is 6.04-6.08 us (the debug model: 16 CTAs, one tile each).
- At 671B down the op reads 411 MB at 4188.3 GB/s, 52.8% of the 7936 GB/s peak; the kernel
  alone runs at 4984 GB/s, 62.8%. At the 1200 MHz application clock the kernel is
  tensor-bound: every 128x128 tile costs 16 `(128, 128, 16)` bfloat16 UMMAs, and the
  steady state derived from the 671B rows after a 6.06 us fixed floor is 1099-1118 cycles
  per tile, inside the 1024-1100 the UMMA rate predicts.
- On the CUDA 13.4 / nvidia-cutlass-dsl 4.6.0 / Triton 3.6.0 toolchain the CuteDSL column
  is 0.97-1.00x of the values above (25.76 us at 671B gate/up, 95.42 us at 671B down); the
  Triton twin does not compile there (`TritonNvidiaGPUOptimizeTMemLayoutsPass`, as in the
  baseline table), so the Triton column is Triton 3.8.0 only.
- `cuobjdump -res-usage` of the compiled kernel: REG 50, STACK 0, SHARED 1024 (static),
  LOCAL 0, at 384 threads and no `setmaxnreg`. Its SASS holds 16 `UTCHMMA` (the 8 + 8
  UMMAs of the two chains), 2 `UTCBAR` and no `HMMA`, `LDSM` or `STSM`: the dynamic shared
  memory is the 5-stage TMA ring of 32 KB `dy` tiles plus the two resident 32 KB `R^T`
  operands, and both epilogues reduce the accumulators from TMEM through registers (16
  `LDTM`) with no shared-memory staging (the 5 `LDS` / 2 `STS` address the static struct).

### group_row_rht_col_rht_quantize_ms_eden (`cutedsl_group_row_rht_col_rht_quantize_ms_eden` vs `triton_group_row_rht_col_rht_quantize_ms_eden`)

The V2 backward gradient quantize: one pass over the packed gradient
`dy = (E * tokens, hidden)` emits both MS-EDEN operands -- rowwise `ms_eden(dy_g @ R_n)` (the
dgrad signs) and columnwise `ms_eden(dy_g.t() @ R_m)` (the wgrad signs) -- RTNE FP4 codes with
a corrected, stochastically rounded E4M3 block scale (ceiling 256, decode numerator 1536),
each against its group's amax from `group_row_rht_col_rht_amax` and its own Philox stream.
The CuteDSL kernel is the amax kernel's two-chain tcgen05 mainloop (one TMA load per 128x128
tile, 8 + 8 UMMAs against the two resident `R^T` operands) with the two amax epilogues
replaced by MS-EDEN quantize epilogues reading the accumulators from TMEM: per 1x16 block
the pre-correction scale and RTNE codes, the decoded codes, the two 16-term inner products,
the correction, and the stochastic E4M3 rounding of the corrected scale from the same Philox
word Triton draws. Codes and scales are bitwise identical across backends at every shape
below (`torch.equal` on all four outputs, including 671B down). Bandwidth counts the
bfloat16 read plus the FP4 codes and swizzled scales on both axes, 3.125 bytes per element.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_row_rht_col_rht_quantize_ms_eden
```

| model | projection | E | tokens | hidden | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 20.41 | 32.19 | 1.58x | 40.1 |
| debugmodel | down (w2) | 4 | 256 | 256 | 20.37 | 32.30 | 1.59x | 40.2 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 1408 | 33.63 | 51.67 | 1.54x | 736.8 |
| 16B | down (w2) | 4 | 2048 | 2048 | 46.38 | 86.84 | 1.87x | 1130.3 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 2048 | 46.33 | 86.81 | 1.87x | 1131.6 |
| 671B | down (w2) | 4 | 7168 | 7168 | 391.43 | 800.95 | 2.05x | 1640.8 |

- Op time includes the same torch glue on both backends -- the two sign-to-matrix builds, no
  fills -- 10.0-11.7 us for CuteDSL (two `torch.mul(h128, sign[None, :])` at ~5 us each on
  ATen's casting `gpu_kernel_impl`, the int8 sign operand) and 12.2-14.5 us for Triton (two
  `get_dynamic_rht_matrix` 7.1-8.8 us plus their two int8->bfloat16 copies 5.0-5.7 us).
  Kernel-only (profiler self CUDA time of the main kernel), CuteDSL vs Triton: 10.41 vs
  20.06 us at the debug model (1.93x), 23.61 vs 39.59 us at 16B gate/up (1.68x), 36.33 vs
  74.56 us at 16B down (2.05x), 36.34 vs 74.43 us at 671B gate/up (2.05x), 379.58 vs
  783.31 us at 671B down (2.06x). The metric excludes the CuteDSL impl's two `Memcpy DtoD`
  (`logical_packed_length.clone()` and the `rng_state` copy into the persistent Philox
  buffer), 3.9-5.0 us of wall time per op that the Triton op does not issue.
- The 671B gate: down `dy (28672, 7168)` 391.43 us against the re-set <= 400 us -- PASS,
  8.6 us (2.1%) under -- and against the provisional <= 300 us -- FAIL, 1.30x over; the op
  is 2.05x Triton 3.8's 800.95 us and the kernel 2.06x (379.6 vs 783.3 us). Gate/up
  `dy (8192, 2048)` 46.33 us against the reported <= 44 us -- FAIL, 2.3 us (5.3%) over;
  1.87x Triton (kernel 36.3 vs 74.4 us). At gate/up the kernel is 36.3 us, under the gate
  on its own, and the glue 10.0 us, 22% of the op: the miss is the glue, the two int8-sign
  `torch.mul` at ~5 us each, as for the amax twin. At 671B down the glue is 3% of the op,
  so both verdicts there are the kernel's.
- At 671B down the op moves 642 MB at 1640.8 GB/s, 20.7% of the 7936 GB/s peak (the kernel
  alone 1692 GB/s): SM-bound, not memory-bound. The steady state from the two 671B rows (7
  and 83 tiles on the busiest CTA, kernel-only 36.3 and 379.6 us) is 4.52 us = 5.4k cycles
  per 128x128 tile at the 1200 MHz application clock (ncu on the 671B down launch: 5.3k
  SM-active cycles per tile) against the 1024-cycle UMMA floor -- the tensor pipe is 19%
  active. The first version of this epilogue -- 12 warps, one warp per TMEM quadrant per
  chain running all 8 blocks of its lane as ~1850 straight-line SASS per chain -- took
  11.1k cycles per tile, Triton's ~11k, i.e. parity, and ncu showed the reason: issue slots
  active 36.5%, 6.85 warp-cycles per issued instruction of which 3.27 (48%) were
  `no_instruction` and 0.05 math-pipe throttle. Two ~30 KB bodies per SM sub-partition (a
  col warp's and a row warp's) run through its instruction cache: fetch-bound, not
  pipe-bound. Scratch variants of the same numerics measured the way down (152-CTA
  two-point launches at hidden 1024): rolling the 8 blocks into a runtime loop, same
  per-block math, 8.6k cycles per tile; two blocks per trip with the codes stored as
  16-byte `st.global.v4`, 6.4k; layout B -- 8 + 8 epilogue warps, the two warps of a
  quadrant splitting a lane's blocks 0-3 / 4-7, 4.5 warps per sub-partition instead of
  2.5 -- with each warp's four blocks as one straight body, the shipped form, 5.4k. Six
  instruction substitutions that keep every rounding op are -8.7% of that (5.9k
  -> 5.4k cycles, 414 -> 378 us kernel: the step that takes the op across 400 us): the
  Philox multiply-high/low pairs as one wide multiply, `rcp.rn.f32` for the correctly
  rounded reciprocal of the E4M3 scale, the widened E4M3 scale reused by the correction,
  the redundant zero-divisor guard dropped (a zero `<v, q>` makes the ratio inf or NaN,
  which the finiteness test already rejects), the E4M3 byte read from bits 31, 26:20 of the
  stochastically rounded word before the 2^120 rescale (exact while the corrected scale is
  at most 448, which finite input guarantees: stored scale <= 256, correction <= 1.23), and
  the bf16 round plus exact widening as one zero-paired `cvt.rn.bf16x2.f32` with the block
  amax taken from the widened words. ncu of the measured kernel at 671B down: issue slots
  active 70%, ALU pipe 66%, FMA-heavy 50%, 4.5 warps active / 1.6 eligible per
  sub-partition, 3.7k warp-instructions issued per sub-partition per tile; stalls per issued
  instruction not-selected 1.35, long scoreboard 1.18 (the in-place `tcgen05.ld`),
  math-pipe throttle 1.10, wait 0.86, no-instruction 0.15. The remaining headroom is
  ALU-class instruction count (`LOP3`/`FMNMX`/`PRMT`/`SHF`), not warps or fetch; a second
  CTA per SM is not available to a tcgen05 kernel (`EIATTR_TCGEN05_1CTA_USED`, occupancy
  1). Both kernels are SM-bound, so the ratio, not the absolute time, is the clock-portable
  number.
- On the CUDA 13.4 / nvidia-cutlass-dsl 4.6.0 / Triton 3.6.0 toolchain the CuteDSL column is
  1.01-1.06x of the values above (20.63 / 20.63 / 34.40 / 47.41 / 47.37 / 415.81 us, medians
  of three passes; REG 96, 3527 SASS, the same UMMA/LDTM/store/math counts, 6% slower at
  671B down). The Triton column is Triton 3.8.0, the toolchain of record: Triton 3.6.0's own
  MS-EDEN scale bytes, rowwise and columnwise, differ from 3.8.0's by one E4M3 step at ~1e-6
  of the blocks (the codes agree), so the bitwise sentence above is against Triton 3.8.0; the
  baseline table's Triton 3.6.0 rows for this op are 50.54 / 84.62 / 84.60 / 865.93 us.
- `cuobjdump -res-usage` of the compiled kernel: REG 96, STACK 0, SHARED 1024 (static),
  LOCAL 0 at 640 threads (20 warps: MMA, TMA, two idle, 8 + 8 epilogue), launch-bounded to
  one CTA per SM with no `setmaxnreg` -- 96 x 640 = 61440 of the SM's 65536 registers, no
  spill. Its 3542 SASS hold 16 `UTCHMMA` (the 8 + 8 UMMAs
  of the two chains), 8 `LDTM` (one per block of each chain's four-block body, 155-192
  SASS between consecutive `LDTM`), 4 `STG.128` (the code stores, two per lane per tile per
  chain) + 4 `STG.U8` (the rowwise scale bytes) and no `HMMA`, `LDL` or `STL`; the epilogue
  is 272 `F2FP`, 156 `FMNMX` + 56 `FMNMX3`, 192 `FMUL2` + 112 `FADD2` (the packed products
  and the two 16-term trees), 135 `IMAD.WIDE` (Philox; the 8 `IMAD.HI` are its
  launch-uniform prep) and 35 `MUFU.RCP` + 8 `FCHK` (the per-group `div.rn` only). Triton
  3.8's kernel for the same op holds 253 registers in use (255 allocated) with 319 spill ops
  (160 `LDL` + 159 `STL`, STACK 632) in 7824 SASS: the twin removes the spills and the
  shared-memory operand staging, and its epilogue issues from a four-block body that fits
  the instruction cache. The times above were measured before `_sr_e4m3_byte` gained its
  sign select (one `shr.u32` more per block), a form byte-identical on every valid input.

### group_col_rht_requant_amax (`cutedsl_group_col_rht_requant_amax` vs `triton_group_col_rht_requant_amax`)

The V2 backward weight amax: one pass over the packed forward weight of the `(E, M, N)`
stack -- the rowwise FP4 codes and swizzled e4m3 scales `group_row_cast_quantize` emitted
-- returns, per expert, the amax of `|W_qdq.t() @ R_n|`, the RHT-128 transform (dgrad
signs) of the dequantized weight's transpose. The CuteDSL kernel rebuilds `W_qdq` on chip:
256 producer threads decode each 128x128 tile's codes and scales in the Triton kernel's
order (2688 numerator, rounded to bfloat16, the +-1 sign folded into the per-row decode
scale) into an MN-major SW128 A stage of a four-deep ring, one MMA warp runs the stage
through eight `(128, 128, 16)` bfloat16 UMMAs against the resident sign-free `H128`, and
four epilogue warps atomic-max the bfloat16-rounded accumulator into the zeroed `(E,)`.
The requantize op below shares this producer, as the Triton twins share
`_load_rht_requant_weight_tile`. The two backends produce bitwise identical amaxes
(`torch.equal` at every shape below). Bandwidth counts the packed FP4 codes and swizzled
scales read, 0.5625 bytes per weight element, plus the `(E,)` amax written.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_col_rht_requant_amax
```

| model | projection | E | M | N | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 8.90 | 15.00 | 1.69x | 16.6 |
| debugmodel | down (w2) | 4 | 256 | 256 | 8.89 | 15.17 | 1.71x | 16.6 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 15.45 | 25.73 | 1.67x | 420.0 |
| 16B | down (w2) | 4 | 2048 | 1408 | 15.03 | 25.68 | 1.71x | 431.8 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 43.77 | 79.29 | 1.81x | 754.5 |
| 671B | down (w2) | 4 | 7168 | 2048 | 43.51 | 79.28 | 1.82x | 759.1 |

- Op time includes the torch glue each backend issues per call: CuteDSL the one
  `torch.zeros((E,))` fill, 1.73-1.76 us (the signs enter as the int8 vector and the
  sign-free `H128` is the cached constant); Triton `get_dynamic_rht_matrix` -- an
  int8->bfloat16 copy at 2.2-2.3 us and the sign product at 3.4-3.5 us -- plus the same
  fill, 7.42-7.45 us. Kernel-only (profiler self CUDA time of the main kernel), CuteDSL vs
  Triton: 7.15 vs 8.01 us at the debug model (1.12x), 13.67 vs 18.31 us at 16B gate/up
  (1.34x), 13.28 vs 18.34 us at 16B down (1.38x), 42.06 vs 71.54 us at 671B gate/up
  (1.70x), 41.75 vs 71.53 us at 671B down (1.71x). Neither op issues a memcpy.
- The 671B gate, op time <= 30 us: gate/up `w (4, 2048, 7168)` 43.77 us -- FAIL, 13.77 us
  (1.46x) over; down `w (4, 7168, 2048)` 43.51 us -- FAIL, 13.51 us (1.45x) over. The
  kernel alone is 42.06 / 41.75 us, over the gate on its own, and the glue 1.76 us (4% of
  the op): the miss is the kernel's. The kernel is producer-bound: the producer threads'
  register-prefetched code and scale loads are latency-exposed (ncu of the producer probe
  on a 2062 MHz node: 1.79 long-scoreboard stalls per issued instruction), so a tile costs
  ~1.8k cycles against the 512-cycle floor of its eight UMMAs and the 660-720-cycle
  producer issue budget the design assumed. A bitwise `cp.async` staging ring for the
  producer's codes and scale words, measured on that 2062 MHz node, takes the kernel from
  1.67x to 2.02x Triton kernel-only at 671B (22.6 vs 27.3 us there) and is not shipped in
  this commit.
- At 671B the op reads 33.0 MB at 754.5-759.1 GB/s, 9.5-9.6% of the 7936 GB/s peak (the
  kernel alone 785-791 GB/s, 9.9-10.0%): neither memory- nor tensor-bound. The steady
  state from the 671B and 16B rows of each projection (24 and 5 tiles on the busiest of
  the 152 persistent CTAs, kernel-only 42.06 / 13.67 and 41.75 / 13.28 us) is 1.49-1.50 us
  = 1.79-1.80k cycles per 128x128 tile at the 1200 MHz application clock against the
  512-cycle UMMA floor -- the tensor pipe is 29% active. The debug model (16 CTAs, one
  tile each) is the kernel's fixed floor, 7.1 us; the same two-point figure for Triton's
  kernel is 3.36k cycles per tile.
- On the CUDA 13.4 / nvidia-cutlass-dsl 4.6.0 / Triton 3.6.0 toolchain the CuteDSL column
  is 0.97-1.00x of the values above (8.87 / 8.85 / 15.28 / 14.87 / 42.66 / 42.48 us, one
  pass; REG 53) and the Triton column 0.97-1.01x (15.11 / 14.89 / 25.13 / 25.02 / 77.12 /
  77.30 us: Triton 3.6.0 compiles this op, as the baseline table records). The twin's
  amaxes are bitwise equal to Triton 3.6.0's as well (`torch.equal` at every shape above).
- `cuobjdump -res-usage` of the compiled kernel: REG 54, STACK 0, SHARED 1024 (static),
  LOCAL 0 at 416 threads (13 warps: the MMA warp, four epilogue, eight producer) with no
  `setmaxnreg`. Its 1232 SASS hold 8 `UTCHMMA` (the tile's eight UMMAs, one tile body in
  the persistent loop), 2 `UTMALDG` (the one-shot `H128` load), 8 `LDTM` and no `HMMA`,
  `LDSM`, `STSM`, `LDL` or `STL`; the producer's rolled word loop is 171 instructions
  (2736 bytes) per pass with 64 `FMUL` and no `FFMA` or `.FTZ` -- the dequantize
  multiplies round as Triton's do (the kernel's 22 `FFMA` and 12 `.FTZ` are the `div.rn`
  expansions of the per-expert scales, as Triton's `tl.div_rn`). The dynamic shared memory
  is the four 32 KB MN-major SW128 A stages plus the resident 32 KB `H128`.

### group_col_rht_requantize (`cutedsl_group_col_rht_requantize` vs `triton_group_col_rht_requantize`)

The V2 backward dgrad weight operand: one pass over the same packed forward weight
requantizes, per expert, `W_qdq.t() @ R_n` rowwise along the transposed axis against the
group amaxes of `group_col_rht_requant_amax` -- RTNE FP4 codes `(E, N, M//2)` and swizzled
e4m3 block scales `(E, N//128, M//64, 32, 16)`, decoded with the 2688 cast numerator. The
CuteDSL kernel is the amax kernel's producer and tcgen05 chain (the same on-chip
dequantize into the SW128 A ring, eight UMMAs per tile against the resident `H128`) with
the accumulator zero-filled in TMEM before each tile's chain, as Triton's kernel
accumulates all eight K steps into a zeroed accumulator, and the amax epilogue replaced by
eight epilogue warps running the landed RTNE 1x16 quantize helpers on four blocks per warp
per tile read from TMEM, writing 64-byte code rows and 4-byte swizzled scale words. Codes
and scales are bitwise identical across backends at every shape below (`torch.equal` on
both outputs, the two ops fed the same amax). Bandwidth counts the packed FP4 codes and
swizzled scales read plus the same written on the transposed axis, 1.125 bytes per weight
element.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_col_rht_requantize
```

| model | projection | E | M | N | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 7.90 | 13.10 | 1.66x | 37.3 |
| debugmodel | down (w2) | 4 | 256 | 256 | 7.89 | 13.19 | 1.67x | 37.4 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 18.04 | 28.19 | 1.56x | 719.3 |
| 16B | down (w2) | 4 | 2048 | 1408 | 17.49 | 28.21 | 1.61x | 742.0 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 61.64 | 104.91 | 1.70x | 1071.8 |
| 671B | down (w2) | 4 | 7168 | 2048 | 65.31 | 105.14 | 1.61x | 1011.5 |

- Op time: the CuteDSL op is its one kernel launch -- no fill, no sign matrix, no memcpy;
  the Triton op adds `get_dynamic_rht_matrix` (the int8->bfloat16 copy at 2.3 us and the
  sign product at 3.3-3.9 us), 5.64-6.16 us per call. Kernel-only (profiler self CUDA time
  of the main kernel), CuteDSL vs Triton: 7.92 vs 7.50 us at the debug model (0.95x -- the
  one cell Triton's kernel wins; the op still wins on the glue), 18.02 vs 22.44 us at 16B
  gate/up (1.24x), 17.49 vs 22.48 us at 16B down (1.28x), 61.86 vs 99.26 us at 671B
  gate/up (1.60x), 65.61 vs 99.50 us at 671B down (1.52x).
- The 671B gate, op time <= 52 us (2x the Triton 3.8.0 baseline): gate/up `w (4, 2048,
  7168)` 61.64 us -- FAIL, 9.64 us (1.19x) over; down `w (4, 7168, 2048)` 65.31 us --
  FAIL, 13.31 us (1.26x) over; against the port plan's <= 30 us -- FAIL, 2.05x and 2.18x
  over. The op is its kernel, so both verdicts are the kernel's; it is 1.70x / 1.61x
  Triton 3.8's 104.91 / 105.14 us. The kernel is bound by the producer the amax kernel is
  bound by, with the eight epilogue warps' RTNE chain on top: 2.77-3.04k cycles per tile
  against the amax kernel's 1.79-1.80k, so the epilogue does not overlap the producer
  fully. The two 671B shapes differ by 6% on this kernel (61.86 vs 65.61 us) and by 0.2%
  on Triton's.
- At 671B the op moves 66.1 MB at 1071.8 / 1011.5 GB/s, 13.5% / 12.7% of the 7936 GB/s
  peak. The steady state from the 671B and 16B rows of each projection (24 and 5 tiles on
  the busiest of the 152 persistent CTAs, kernel-only 61.86 / 18.02 and 65.61 / 17.49 us)
  is 2.31-2.53 us = 2.77-3.04k cycles per 128x128 tile at the 1200 MHz application clock
  against the 512-cycle UMMA floor -- the tensor pipe is 17-19% active. The debug model
  (16 CTAs, one tile each) is the kernel's fixed floor, 7.9 us; the same two-point figure
  for Triton's kernel is 4.85-4.87k cycles per tile. Both kernels are SM-bound, so the
  ratio, not the absolute time, is the clock-portable number.
- On the CUDA 13.4 / nvidia-cutlass-dsl 4.6.0 / Triton 3.6.0 toolchain the CuteDSL column
  is 1.02-1.11x of the values above (8.13 / 8.11 / 18.49 / 17.93 / 66.60 / 72.66 us, one
  pass; REG 71, 8-11% slower at 671B) and Triton 3.6.0 is faster than 3.8.0 on this op
  (13.12 / 13.12 / 26.51 / 26.44 / 89.72 / 89.85 us, 0.85x at 671B, as in the baseline
  table), so the twin is 1.24-1.62x there. Codes and scales are bitwise equal to Triton
  3.6.0's as well (`torch.equal` at every shape above).
- `cuobjdump -res-usage` of the compiled kernel: REG 80, STACK 0, SHARED 1024 (static),
  LOCAL 0 at 544 threads (17 warps: the MMA warp, eight epilogue, eight producer) with no
  `setmaxnreg` -- 80 x 544 = 43520 of the SM's 65536 registers. Its 1568 SASS hold 8
  `UTCHMMA`, 2 `UTMALDG`, 12 `STTM` (the TMEM zero fill), 4 `LDTM`, 2 `STG.128` + 1 `STG`
  (the code row and the scale words) and no `HMMA`, `LDSM`, `STSM`, `LDL` or `STL`; the
  producer loop is the amax kernel's (171 instructions per pass, 64 `FMUL`, no `FFMA` or
  `.FTZ`), and the whole kernel holds 105 `FMUL`, 105 `F2FP` and 14 `MUFU.RCP` + 8 `FCHK`
  (the `div.rn` expansions) against the amax kernel's 64 / 33 / 7 + 2 -- the difference is
  the RTNE epilogue.

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
