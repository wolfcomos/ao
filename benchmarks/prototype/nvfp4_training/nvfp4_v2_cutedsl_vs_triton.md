# NVFP4 V2 grouped kernels: CuteDSL vs Triton

Per-kernel CuteDSL ports of the grouped Triton kernels behind `nvfp4_grouped_mm_v2` (the
V2 recipe and the V1_REQUANT weight path), each measured against its Triton twin with the
scripts in this directory. The tables in `README.md` were measured on a different
environment, so compare numbers within this file only.

## Environment and methodology

- NVIDIA GB200, 152 SMs, SM application clock capped at 1200 MHz (`nvidia-smi`
  applications clock; every table in this file, the re-measured ones included, is at this
  local cap). Absolute times of SM-bound kernels are roughly 1.6x those in `README.md`
  (the 1965 / 1200 MHz clock ratio); memory-bound rows, such as `row_cast_quantize` at 671B,
  do not scale with the SM clock. Peak bandwidth 7936 GB/s is memory bus width / 8 x memory
  clock x 2 from `torch.cuda.get_device_properties` -- the `get_peak_mem_bw_gbps` formula
  of the sibling `bench_group_rht_quantize_row_col.py` / `bench_quantize_2d.py`; the nine
  benches here print GB/s only -- as read on the GPU that produced the first tables
  (7936-bit bus x 4000 MHz; other GB200 GPUs on these nodes report an 8064-bit bus, 8064
  GB/s, and `README.md`'s environment 7928.1). Every `pct_peak` below is against 7936.
- Primary toolchain, used for every table: PyTorch 2.14.0a0, CUDA 13.x (pre-release), Triton 3.8.0,
  nvidia-cutlass-dsl 4.8.0.dev0. The Triton 3.6.0 column of the baseline table comes from
  PyTorch 2.14.0a0, CUDA 13.4, Triton 3.6.0, nvidia-cutlass-dsl 4.6.0; every CuteDSL
  kernel here compiles and passes its single-backend tests on both. The bitwise claims are
  against Triton 3.8.0: the two RHT-128 amax twins have no Triton 3.6.0 counterpart and the
  MS-EDEN scale bytes differ from Triton 3.6.0's at ~1e-6 of blocks -- see each kernel's
  toolchain bullet. The Triton 3.6.0-toolchain figures in those bullets are single passes
  unless stated.
- Device kernel time via `bench_utils.kernel_time_us` (15 warmups / 50 iterations, CUDA
  self-time, memcpy/memset excluded) -- op time for every kernel the custom op launches.
  The exclusion drops the `logical_packed_length.clone()` (and, for MS-EDEN, the
  `rng_state` copy) `Memcpy DtoD` of the four activation CuteDSL ops, 2.0-2.4 us (3.9-5.0 us
  MS-EDEN) per call that the Triton ops do not issue; each of those sections quotes it. The
  timed loop re-runs one buffer with no L2 flush: rows whose working set is under the
  137 MB L2 (the weight tables at the debug model and 16B, the 671B `col_*` rows) read
  partly from cache -- measured up to 10% at 6.5-33 MB and none at >= 117 MB -- so their
  GB/s is an upper bound, and the two backends do not always lose the same share when cold.
- `E = 4` local experts (the 671B EP-64 layout; the V2 training runs use 8 local experts at
  ep 32 / ep 8), DeepSeek-V3 shapes from `deepseek_v3_shapes.py`. Weight tables: `w` is
  `(E, M, N)`, `M` rows (out features), `N` columns (in features). Activation tables:
  `x = (E * tokens, dim)` with `dim = N` (weight columns) and `dy = (E * tokens, dim)` with
  `dim = M` (weight rows); `tokens` per expert is the recipe's per-step count for a balanced
  router, EP-group ranks x local batch x seq x top_k / experts -- 16B 8 x 4 x 4096 x 6 / 64
  = 12288 (2 nodes x 4 GPUs, ep 8, local batch 4, seq 4096), 671B 32 x 8 x 4096
  x 8 / 256 = 32768 (16 nodes x 4 GPUs, ep 32, local batch 8, seq 4096; the EP-64 layout
  with the same batch would give 65536), debugmodel 256 (no such run; 16 tiles at `E = 4`).
  Debug-model rows are 16-tile launch-floor probes on either backend, so their speedup is a
  latency ratio, not a throughput one.
- Medians of three full script passes; the same op re-measured on another GPU of the node
  moves up to ~4% (op-dependent), so read speedups at that precision. The four activation
  tables and the baseline's activation rows were re-measured at the recipe token counts, and
  the `row_cast_quantize` table with its own bench, on 2026-09-14 (GPU 3 of the same node,
  1200 MHz local cap; Triton 3.6.0 in the CUDA 13.4 container on the same GPU).

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
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 5.28 | 4.47 | 0.85x | 127.2 |
| debugmodel | down (w2) | 4 | 256 | 256 | 5.28 | 4.45 | 0.84x | 127.2 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 7.15 | 11.77 | 1.65x | 4134.4 |
| 16B | down (w2) | 4 | 2048 | 1408 | 6.94 | 11.72 | 1.69x | 4257.1 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 32.09 | 52.10 | 1.62x | 4688.6 |
| 671B | down (w2) | 4 | 7168 | 2048 | 31.89 | 52.26 | 1.64x | 4719.0 |

CuteDSL wins 1.6-1.7x at 16B and 671B and loses at the debug model (0.85x; the Triton cell
there lands at 3.95 or 4.47 us from one pass to the next, so 0.75-0.85x). At 671B it
sustains 4.7 TB/s, 59% of peak, against Triton's 2.9 TB/s. Both kernels read `A` with 16-B
loads and write codes with 8-B stores; the CuteDSL kernel holds 48 registers with no shared
memory, barriers or shuffles (each thread stores its swizzled scale byte directly), where
the Triton kernel at `num_warps = 8` holds 107 registers and reduces through shared memory,
so it keeps roughly 2.5x the warps resident. The debug model launches 16 CTAs on either
backend and is latency-bound.

### group_row_rht_col_rht_amax (`cutedsl_group_row_rht_col_rht_amax` vs `triton_group_row_rht_col_rht_amax`)

The V2 backward gradient amax: one pass over the packed gradient
`dy = (E * tokens, dim)` returns, per expert, the amax of `|dy_g @ R_n|` (rowwise, the
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

| model | projection | E | tokens | dim | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 19.73 | 25.83 | 1.31x | 26.6 |
| debugmodel | down (w2) | 4 | 256 | 256 | 19.74 | 25.95 | 1.31x | 26.6 |
| 16B | gate/up (w1/w3) | 4 | 12288 | 1408 | 47.56 | 163.76 | 3.44x | 2910.4 |
| 16B | down (w2) | 4 | 12288 | 2048 | 59.47 | 227.81 | 3.83x | 3385.2 |
| 671B | gate/up (w1/w3) | 4 | 32768 | 2048 | 120.56 | 575.00 | 4.77x | 4453.1 |
| 671B | down (w2) | 4 | 32768 | 7168 | 366.71 | 1935.77 | 5.28x | 5124.0 |

- Op time includes the same torch glue on both backends -- two sign-to-matrix builds and
  two `torch.zeros` fills -- 13.8-16.5 us for CuteDSL and 15.5-19.3 us for Triton. At
  671B gate/up, `dy (131072, 2048)`: CuteDSL two `torch.mul(h128, sign[None, :])` 11.61 us
  (the int8 sign operand puts them on ATen's casting `gpu_kernel_impl`; with bfloat16
  signs the pair takes 6.77 us) + two fills 3.93 us; Triton two `get_dynamic_rht_matrix`
  8.74 us + their two int8->bfloat16 copies 5.86 us + two fills 4.67 us. Kernel-only
  (profiler self CUDA time of the main kernel), CuteDSL vs Triton: 32.10 vs 145.95 us at
  16B gate/up (4.55x), 44.06 vs 208.27 us at 16B down (4.73x), 105.20 vs 555.25 us at 671B
  gate/up (5.28x), 350.12 vs 1919.80 us at 671B down (5.48x). The metric excludes the
  1.98-2.74 us `Memcpy DtoD` of the CuteDSL impl's `logical_packed_length.clone()`, which
  the Triton op does not issue.
- The 671B gates of the port plan were set at the earlier record's shapes, which are no
  longer in the table: gate/up `dy (8192, 2048)` 25.94 us against <= 22 us -- over by
  3.94 us (17.9%); down `dy (28672, 7168)` 98.14 us against <= 105 us -- under by 6.86 us
  (6.5%). At that gate/up shape the kernel was 11.55 us, under the gate on its own, and the
  glue 14.18 us, 55% of the op: the miss is the glue, the two int8-sign `torch.mul` at
  5.1 us each. At the recipe token counts the same glue is 13% of the 671B gate/up op
  (15.5 of 120.7 us) and 5% at 671B down; the kernel's fixed floor is 5.96 us (the debug
  model: 16 CTAs, one tile each).
- At 671B down the op reads 1879 MB at 5124.0 GB/s, 64.6% of the 7936 GB/s peak; the kernel
  alone runs at 5367 GB/s, 67.6%. At the 1200 MHz application clock the kernel is
  tensor-bound: every 128x128 tile costs 16 `(128, 128, 16)` bfloat16 UMMAs, and the
  steady state derived from the 671B rows (108 and 378 tiles on the busiest of the 152
  CTAs) after the 5.96 us fixed floor is 1093-1103 cycles per tile, inside the 1024-1100
  the UMMA rate predicts.
- On the CUDA 13.4 / nvidia-cutlass-dsl 4.6.0 / Triton 3.6.0 toolchain the CuteDSL column
  is 0.97-1.00x of the values above (117.07 us at 671B gate/up, 354.27 us at 671B down, one
  pass); the Triton twin does not compile there (`TritonNvidiaGPUOptimizeTMemLayoutsPass`,
  as in the baseline table), so the Triton column is Triton 3.8.0 only.
- `cuobjdump -res-usage` of the compiled kernel: REG 50, STACK 0, SHARED 1024 (static),
  LOCAL 0, at 384 threads and no `setmaxnreg`. Its SASS holds 16 `UTCHMMA` (the 8 + 8
  UMMAs of the two chains), 2 `UTCBAR` and no `HMMA`, `LDSM` or `STSM`: the dynamic shared
  memory is the 5-stage TMA ring of 32 KB `dy` tiles plus the two resident 32 KB `R^T`
  operands, and both epilogues reduce the accumulators from TMEM through registers (16
  `LDTM`) with no shared-memory staging (the 5 `LDS` / 2 `STS` address the static struct).

### group_row_rht_col_rht_quantize_ms_eden (`cutedsl_group_row_rht_col_rht_quantize_ms_eden` vs `triton_group_row_rht_col_rht_quantize_ms_eden`)

The V2 backward gradient quantize: one pass over the packed gradient
`dy = (E * tokens, dim)` emits both MS-EDEN operands -- rowwise `ms_eden(dy_g @ R_n)` (the
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

| model | projection | E | tokens | dim | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 20.39 | 32.16 | 1.58x | 40.2 |
| debugmodel | down (w2) | 4 | 256 | 256 | 20.36 | 32.27 | 1.58x | 40.2 |
| 16B | gate/up (w1/w3) | 4 | 12288 | 1408 | 142.91 | 289.51 | 2.03x | 1513.3 |
| 16B | down (w2) | 4 | 12288 | 2048 | 200.20 | 406.08 | 2.03x | 1571.3 |
| 671B | gate/up (w1/w3) | 4 | 32768 | 2048 | 493.43 | 1034.84 | 2.10x | 1700.0 |
| 671B | down (w2) | 4 | 32768 | 7168 | 1699.15 | 3564.01 | 2.10x | 1727.9 |

- Op time includes the same torch glue on both backends -- the two sign-to-matrix builds, no
  fills -- 9.9-12.1 us for CuteDSL (two `torch.mul(h128, sign[None, :])` at 5-6 us each on
  ATen's casting `gpu_kernel_impl`, the int8 sign operand) and 12.0-15.1 us for Triton (two
  `get_dynamic_rht_matrix` 7.2-9.0 us plus their two int8->bfloat16 copies 4.8-6.2 us).
  Kernel-only (profiler self CUDA time of the main kernel), CuteDSL vs Triton: 10.37 vs
  19.99 us at the debug model (1.93x), 131.43 vs 275.23 us at 16B gate/up (2.09x), 188.38
  vs 390.82 us at 16B down (2.07x), 481.50 vs 1017.77 us at 671B gate/up (2.11x), 1688.12
  vs 3539.96 us at 671B down (2.10x). The metric excludes the CuteDSL impl's two `Memcpy
  DtoD` (`logical_packed_length.clone()` and the `rng_state` copy into the persistent
  Philox buffer), 4.0-5.2 us of wall time per op that the Triton op does not issue.
- The 671B gates of the port plan were set at the earlier record's shapes, which are no
  longer in the table: down `dy (28672, 7168)` 391.43 us against the re-set <= 400 us --
  PASS, 8.6 us (2.1%) under -- and against the provisional <= 300 us -- FAIL, 1.30x over;
  the op was 2.05x Triton 3.8's 800.95 us and the kernel 2.06x (379.6 vs 783.3 us).
  Gate/up `dy (8192, 2048)` 46.33 us against the reported <= 44 us -- FAIL, 2.3 us (5.3%)
  over; 1.87x Triton (kernel 36.3 vs 74.4 us). At that gate/up shape the kernel was
  36.3 us, under the gate on its own, and the glue 10.0 us, 22% of the op: the miss is the
  glue, the two int8-sign `torch.mul` at ~5 us each, as for the amax twin. At the recipe
  token counts the glue is 2% of the 671B gate/up op (12.1 of 493.5 us) and 1% at 671B
  down, so the 2.10x of both 671B rows is the kernel's (2.11x and 2.10x kernel-only).
- At 671B down the op moves 2936 MB at 1727.9 GB/s, 21.8% of the 7936 GB/s peak (the kernel
  alone 1739 GB/s): SM-bound, not memory-bound. The steady state from the two 671B rows (108
  and 378 tiles on the busiest CTA, kernel-only 481.5 and 1688.1 us) is 4.47 us = 5.4k cycles
  per 128x128 tile at the 1200 MHz application clock (ncu on the earlier record's 671B down
  launch, `dy (28672, 7168)`: 5.3k SM-active cycles per tile) against the 1024-cycle UMMA
  floor -- the tensor pipe is 19% active. The first version of this epilogue -- 12 warps, one
  warp per TMEM quadrant per
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
  0.99-1.02x of the values above (20.21 / 20.19 / 145.45 / 202.91 / 497.96 / 1717.28 us,
  medians of three passes; REG 96, 3527 SASS, the same UMMA/LDTM/store/math counts, 1% slower
  at 671B down). The Triton column is Triton 3.8.0, the toolchain of record: Triton 3.6.0's own
  MS-EDEN scale bytes, rowwise and columnwise, differ from 3.8.0's by one E4M3 step at ~1e-6
  of the blocks (the codes agree), so the bitwise sentence above is against Triton 3.8.0; the
  baseline table's Triton 3.6.0 rows for this op are 309.63 / 436.25 / 1114.48 / 3868.53 us.
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
- The 671B gate, op time <= 52 us (half the Triton 3.8.0 baseline, a 2x speedup): gate/up `w (4, 2048,
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

### group_col_cast_requant_amax (`cutedsl_group_col_cast_requant_amax` vs `triton_group_col_cast_requant_amax`)

The V1_REQUANT backward weight amax: one pass over the packed forward weight of the `(E,
M, N)` stack -- the rowwise FP4 codes and swizzled e4m3 scales `group_row_cast_quantize`
emitted -- returns, per expert, `max|bf16(W_qdq)|`, the amax of the dequantized weight (a
transpose does not change the set of elements). The CuteDSL kernel is a streaming
CUDA-core reduction with no MMA, TMA or pipeline SMEM: 128 threads per 128x128 tile read
the 8 KB of codes as 16 B vectors and the two 512 B scale atoms as one 16 B word run per
thread, take each 1x16 block's largest magnitude code with bit masks (never a per-nibble
decode), multiply it by the block scale -- exact in f32 -- and reduce the exact products;
the multiply by the per-expert decode scale and the one bfloat16 rounding happen once per
CTA, which equals Triton's per-element rounding because `x -> bf16(x * dec)` is monotone
on `x >= 0`. A NaN or inf expert amax reports 0.0 as Triton's does. The two backends
produce bitwise identical amaxes (`torch.equal` at every shape below, with the tensor's
own amax and an over-bounding one). Bandwidth counts the packed FP4 codes and swizzled
scales read, 0.5625 bytes per weight element, plus the `(E,)` amax written.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_col_cast_requant_amax
```

| model | projection | E | M | N | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 4.87 | 6.06 | 1.24x | 30.3 |
| debugmodel | down (w2) | 4 | 256 | 256 | 4.86 | 6.22 | 1.28x | 30.3 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 6.19 | 11.76 | 1.90x | 1048.7 |
| 16B | down (w2) | 4 | 2048 | 1408 | 6.17 | 11.72 | 1.90x | 1052.1 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 14.44 | 38.13 | 2.64x | 2287.2 |
| 671B | down (w2) | 4 | 7168 | 2048 | 14.24 | 38.12 | 2.68x | 2320.2 |

- Op time includes the one `torch.zeros((E,))` fill both backends issue per call,
  1.74-1.78 us (CuteDSL) and 1.74-1.77 us (Triton) at this clock; two device launches per
  call on each side, no memcpy. Kernel-only (profiler self CUDA time of the main kernel),
  CuteDSL vs Triton: 3.11 vs 4.39 us at the debug model (1.41x), 4.43 vs 10.04 us at 16B
  gate/up (2.27x), 4.42 vs 10.01 us at 16B down (2.26x), 12.68 vs 36.41 us at 671B gate/up
  (2.87x), 12.45 vs 36.39 us at 671B down (2.92x).
- The port plan's gates (§1.6 / M1), op time <= 8 us at 16B: 6.19 / 6.17 us -- PASS (0.77x
  of the budget); <= 12 us at 671B: 14.44 / 14.24 us -- FAIL, 2.44 / 2.24 us (1.20x /
  1.19x) over. The fill is 12.1-12.4% of the 671B op and the kernel alone (12.68 / 12.45
  us) is over the gate on its own by 0.68 / 0.45 us, so the miss is the fill plus a little
  of the kernel; kernel-only is reported alongside as the plan asks.
- At 671B the op reads 33.0 MB at 2287-2320 GB/s, 28.8-29.2% of the 7936 GB/s peak (the
  kernel alone 2605-2653 GB/s, 32.8-33.4%); at 16B 6.49 MB at 1049-1052 GB/s (13.2-13.3%;
  the kernel alone 1466 GB/s, 18.5%), where the fill (28% of the op) and the launch floor
  dominate. Debug model to 16B to 671B the kernel scales 3.1 -> 4.4 -> 12.6 us for 0.15 ->
  6.5 -> 33.0 MB: the debug row is the fixed floor and the incremental rate between the
  two large rows is 3215-3306 GB/s (40.5-41.7% of peak).
- On the CUDA 13.4 / nvidia-cutlass-dsl 4.6.0 / Triton 3.6.0 toolchain the CuteDSL column
  is 0.99-1.01x of the values above (4.91 / 4.90 / 6.16 / 6.18 / 14.43 / 14.25 us, one
  pass; REG 33 and the same 480 SASS on both DSLs) and Triton 3.6.0 is 1.04-1.25x slower
  on this op (6.50 / 6.49 / 13.39 / 13.39 / 47.49 / 47.50 us, as the baseline table
  records), so the twin is 1.32-3.33x there. The amaxes are bitwise equal to Triton
  3.6.0's as well (`torch.equal` at every shape above and on the smoke sets).
- `cuobjdump -res-usage` of the compiled kernel: REG 33, STACK 0, SHARED 1024 (static; the
  four-word warp-max scratch), LOCAL 0 at 128 threads. Its 480 SASS hold 5 `LDG.E.128` + 1
  `LDG.E` (the four code pieces, the scale words, the expert amax), 9 `FMUL`, 7 `FMNMX`, 5
  `SHFL`, one `STS` / `LDS.128` / `BAR.SYNC` (the cross-warp max) and no `LDL` / `STL`,
  `HMMA` or `UTCHMMA`; the 22 `FFMA`, 3 `MUFU.RCP` + 1 `MUFU.RSQ`, 2 `FCHK` and 8 `.FTZ`
  are the `div.rn` expansion of `_global_scale`, as Triton's `tl.div_rn`. A NaN scale byte
  (0x7f / 0xff, which §11.1 never emits) is NaN in both backends with a different payload
  (0x7fff0000 here, 0x7fc00000 in Triton).

### group_col_cast_requantize (`cutedsl_group_col_cast_requantize` vs `triton_group_col_cast_requantize`)

The V1_REQUANT backward dgrad weight operand: one pass over the same packed forward weight
requantizes, per expert, `W_qdq.bf16().t()` rowwise along the transposed axis against the
group amaxes of `group_col_cast_requant_amax` -- RTNE FP4 codes `(E, N, M//2)` and
swizzled e4m3 block scales `(E, N//128, M//64, 32, 16)`, decoded with the 2688 cast
numerator. The CuteDSL kernel is a CUDA-core SMEM transpose, no MMA: 128 threads per
128x128 tile, thread `m` rebuilds weight row `m` (four 16 B code loads, two scale words,
the Triton reconstruction op for op with `_dequant_e2m1x8_bf16x2x4`, `+0` for a NaN / inf
expert amax) into a padded bf16 SMEM tile (row pitch 272 B, so the 16 B row stores and the
column gathers are bank-conflict-free); after the barrier thread `t` owns output rows `2 *
(t % 64)`, `+ 1` (the low and high bf16 of one column word) over four of the eight blocks
along `m`, gathers each 1x16 block with 16 `LDS`, runs the landed RTNE 1x16 quantize
helpers and writes two 32 B code runs (`st.global.v4`) and four scale bytes per row in the
§11.1 swizzle. The transpose being a plain gather is load-bearing for the bitwise
contract: an identity-matrix `tcgen05` chain (the col_rht kernel with `B = I`) reproduces
every code and scale except the sign of exact zeros, because `+0 + (-0)` is `+0` on the
accumulator where Triton's `tl.trans` carries `-0` through to the 0x8 nibble -- measured
on 23 of 23 real-weight cases, with the mismatch count equal to the number of `-0`
nibbles. Codes and scales are bitwise identical across backends at every shape below
(`torch.equal` on both outputs, each backend fed its own bitwise amax). Bandwidth counts
the packed FP4 codes and swizzled scales read plus the same written on the transposed
axis, 1.125 bytes per weight element.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_col_cast_requantize
```

| model | projection | E | M | N | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 5.26 | 12.01 | 2.28x | 56.0 |
| debugmodel | down (w2) | 4 | 256 | 256 | 5.26 | 12.01 | 2.28x | 56.0 |
| 16B | gate/up (w1/w3) | 4 | 1408 | 2048 | 17.11 | 56.12 | 3.28x | 758.3 |
| 16B | down (w2) | 4 | 2048 | 1408 | 19.23 | 56.06 | 2.92x | 674.9 |
| 671B | gate/up (w1/w3) | 4 | 2048 | 7168 | 58.45 | 263.85 | 4.51x | 1130.3 |
| 671B | down (w2) | 4 | 7168 | 2048 | 63.37 | 264.36 | 4.17x | 1042.5 |

- Op time is the one kernel launch for both backends -- no fill, no sign matrix, no memcpy
  -- so op time is kernel time here; the profiler split reproduces the table within 0.5%
  (CuteDSL 5.27 / 5.25 / 17.12 / 19.22 / 58.47 / 63.08 us, Triton 12.02 / 12.01 / 56.12 /
  56.06 / 265.27 / 266.00 us).
- The port plan's gates (§1.8 / M2, targets the plan marks *inferred*), op time <= 8 us at
  16B: 17.11 / 19.23 us -- FAIL, 2.14x / 2.40x over; <= 25 us at 671B: 58.45 / 63.37 us --
  FAIL, 2.34x / 2.53x over; the plan's "anything <= 12 us at 16B beats §11.1's own Triton
  time" -- FAIL. Against the identity-`tcgen05` route ruled out above (18.06 / 17.52 /
  61.83 / 65.40 us kernel-only, the col_rht kernel's time, one pass on another GPU of the
  node) this kernel is 0.95x / 1.10x / 0.95x / 0.97x -- the same speed with no tensor pipe
  and the sign of zero right. A register prefetch of the next tile's codes and scale words
  before the epilogue (the col_rht producer's one-tile-ahead pattern) was measured bitwise
  and a wash (+10% / +6% / -1.5% / +0.4% / -3.6% over the five timed shapes) and is not
  shipped.
- At 671B the op moves 66.1 MB at 1130.3 / 1042.5 GB/s, 14.2% / 13.1% of the 7936 GB/s
  peak; at 16B 13.0 MB at 758 / 675 GB/s (9.6% / 8.5%). The grid is `(N//128, GRID_Y, E)`
  with `GRID_Y` sized for four resident CTAs per SM (34 KB of SMEM, 96 registers at 128
  threads): 640 / 616 / 672 / 640 CTAs over the four large rows, 4.05-4.42 per SM, so
  every row of the table runs in one wave with 1-2 tiles per CTA at 16B and 5-6 at 671B.
  The two-point steady state from the gate/up rows (58.47 - 17.12 us over the four extra
  tiles of the busiest CTA) is 10.3 us = 12.4k cycles per tile per CTA at the 1200 MHz
  application clock, ~2.8-3.0k cycles per tile per SM with the 4.2-4.4 resident CTAs --
  the col_rht kernel's 2.77-3.04k with no UMMA in the loop. The kernel is issue- and
  latency-bound on its 16-18 resident warps per SM (the `LDS` -> unpack -> `_quant16`
  chains), not memory-bound. The down rows are slower than gate/up at equal bytes, 12% at
  16B (19.23 vs 17.11 us; 11 column tiles x 14 y-slots over 16 row blocks against 16 x 10
  over 11) and 8% at 671B (63.37 vs 58.45 us), where Triton's rows agree within 0.2%; the
  cause was not isolated.
- On the CUDA 13.4 / nvidia-cutlass-dsl 4.6.0 / Triton 3.6.0 toolchain the CuteDSL column
  is 1.02-1.07x of the values above (5.37 / 5.38 / 17.82 / 20.57 / 61.69 / 65.67 us, one
  pass; REG 80 and 1544 SASS there against 96 / 1528) and Triton 3.6.0 is 1.44-1.48x
  slower than 3.8.0 on this op (17.31 / 17.29 / 82.41 / 82.26 / 389.27 / 389.51 us, as the
  baseline table records), so the twin is 3.21-6.31x there. Codes and scales are bitwise
  equal to Triton 3.6.0's as well (`torch.equal` at every shape above and on the 43-case
  smoke set: 671B shapes, NaN / inf / -inf / 0 / negative expert amax, halved and doubled
  requant amax, the +-6 representable stack, an all-zero and an all-`-0` (0x88) expert,
  random code and scale bytes with 0 / -0 / subnormal scales, 512 one-tile experts,
  degenerate E / M / N = 0, CUDA-graph capture and two replays).
- `cuobjdump -res-usage` of the compiled kernel: REG 96, STACK 0, SHARED 1024 (static; the
  34816 B tile is dynamic), LOCAL 0 at 128 threads. Its 1528 SASS hold 4 `LDG.E.128` and 4
  `LDG.E` (the codes, the scale words, the two expert amaxes), 16 `STS.128` (the row
  stores), 64 `LDS` (the four 16-deep column gathers), 4 `STG.E.128` + 8 `STG.E.U8` (the
  code runs and scale bytes), 2 `BAR.SYNC`, 273 `FMUL`, 26 `FMNMX`, 212 `F2FP` (64
  `F16.E2M1.UNPACK_B`, 64 `BF16.F32.PACK_AB`, 64 `SATFINITE.E2M1.F32.PACK_AB_MERGE_C`, 12
  `F16.E4M3.UNPACK_B`, 8 `SATFINITE.E4M3.F32.PACK_AB_MERGE_C`) and no `SHFL`, `LDL` /
  `STL`, `HMMA` or `UTCHMMA`; the 72 `FFMA`, 13 `MUFU.RCP` + 1 `MUFU.RSQ`, 12 `FCHK` and 8
  `.FTZ` are the `div.rn` expansions of the per-expert and per-block scales, as Triton's
  `tl.div_rn`.

### group_row_cast_col_rht_amax (`cutedsl_group_rht_amax` vs `triton_group_rht_amax`, `dynamic_rht=True`)

The V2 forward activation amax: one pass over the packed activation `x = (E * tokens, dim)`
returns, per expert, the raw rowwise amax `max|x_g|` and the columnwise amax of
`|x_g.t() @ R|`, a 128-point randomized Hadamard transform along the tokens with the live
wgrad sign buffer (`sign_tensor`, `dynamic_rht=True`; the static RHT-16 path of the same op
is unchanged). The CuteDSL kernel loads every 128x128 tile into shared memory once (TMA) and
feeds it to two consumer groups: one tcgen05 UMMA chain (8 UMMAs against the resident
`R^T` operand, the col chain of `group_row_rht_col_rht_amax`) whose accumulator four warps
reduce from TMEM, and eight row warps that read the same bytes through a plain
`(dim, token, stage)` view and reduce the raw amax -- the RHT-16 kernels' two-consumer
stage. The two backends produce bitwise identical amaxes (`torch.equal` on both outputs at
every shape below, 64 groups, empty groups, spare capacity rows and int8 / bfloat16 / float32
sign buffers). Bandwidth counts the bfloat16 read of `x`; the `2E` scalar outputs are not
counted.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_row_cast_col_rht_amax
```

| model | projection | E | tokens | dim | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 14.64 | 18.92 | 1.29x | 35.8 |
| debugmodel | down (w2) | 4 | 256 | 256 | 14.63 | 18.88 | 1.29x | 35.8 |
| 16B | gate/up (w1/w3) | 4 | 12288 | 2048 | 55.39 | 194.21 | 3.51x | 3634.8 |
| 16B | down (w2) | 4 | 12288 | 1408 | 42.51 | 136.31 | 3.21x | 3256.0 |
| 671B | gate/up (w1/w3) | 4 | 32768 | 7168 | 392.54 | 1648.42 | 4.20x | 4786.9 |
| 671B | down (w2) | 4 | 32768 | 2048 | 121.94 | 485.77 | 3.98x | 4402.9 |

- Op time includes the same torch glue on both backends -- one sign-to-matrix build and two
  `torch.zeros` fills -- 9.0-11.4 us for CuteDSL and 9.7-12.8 us for Triton. At 671B gate/up,
  `x (131072, 7168)`: CuteDSL `torch.mul(h128, sign[None, :])` 6.84 us (the int8 sign
  operand puts it on ATen's casting `gpu_kernel_impl`) + two fills 4.58 us; Triton
  `get_dynamic_rht_matrix` 5.06 us + its int8->bfloat16 copy 3.03 us + two fills 4.61 us.
  Kernel-only (profiler self CUDA time of the main kernel), CuteDSL vs Triton: 44.66 vs
  182.76 us at 16B gate/up (4.09x), 31.96 vs 125.13 us at 16B down (3.91x), 380.85 vs
  1641.66 us at 671B gate/up (4.31x), 110.94 vs 474.71 us at 671B down (4.28x). The metric
  excludes the 2.02-2.78 us `Memcpy DtoD` of the CuteDSL impl's
  `logical_packed_length.clone()`, which the Triton op does not issue.
- The op gates of the port plan were set at the earlier record's shapes, which are no longer
  in the table: 16B `x (5632, 2048)` 18.59 us against <= 16 us -- over by 2.59 us (16.2%);
  671B `x (8192, 7168)` 38.75 us against <= 32 us -- over by 6.75 us (21.1%). At both shapes
  the kernel alone was under the gate (9.51 and 28.26 us) and the glue 8.9-10.4 us, 27-48%
  of the op: the miss is the glue, the int8-sign `torch.mul` plus the two fills. At the
  recipe token counts the same glue is 19% of the 16B gate/up op (10.5 of 55.2 us) and 3%
  at 671B gate/up (11.4 of 392.3 us). Flipping the sign bits of the cached unsigned `H128`
  in shared memory by the two idle warps would remove the `torch.mul`; the fills are the
  Triton op's contract (zero-initialised outputs, atomic max).
- At 671B gate/up the op reads 1879 MB at 4786.9 GB/s, 60.3% of the 7936 GB/s peak; the
  kernel alone runs at 4934 GB/s, 62.2% (16B gate/up: 4508 GB/s kernel-only). Per tile,
  derived from the 671B kernel-only times over 57344 and 16384 tiles on 152 CTAs after the
  5.72 us fixed floor (the debug-model row less its glue): 0.97-0.99 us, ~1170-1190 cycles
  at 1200 MHz -- above both the 753-cycle HBM share of a 32 KB tile and the ~550 cycles of
  the 8-UMMA chain, so the row warps' shared-memory pass (4 x 16-wide reads per thread) and
  the two-consumer stage recycle are what to profile next; the design's 750-900-cycle
  forecast was not reached.
- On the CUDA 13.4 / nvidia-cutlass-dsl 4.6.0 / Triton 3.6.0 toolchain the CuteDSL kernel
  compiles and passes every single-backend test (oracle, in-place resample, padded rows); the
  Triton op does not compile there (`TritonNvidiaGPUOptimizeTMemLayoutsPass`, as in the
  baseline table), so the cross-backend `torch.equal` items and the Triton column are
  Triton 3.8.0 only.
- `cuobjdump -res-usage` of the compiled kernel: REG 34, STACK 0, SHARED 1024 (static),
  LOCAL 0, at 512 threads and no `setmaxnreg`. Its SASS (1632 lines) holds 8 `UTCHMMA` (the
  single chain), 2 `UTCBAR`, 8 `LDTM` (the col epilogue's TMEM reads), 4 `REDG.E.MAX` (the
  per-group bit-pattern max flushes), 118 `FMNMX`, 12 `LDS` / 2 `STS` (the static struct
  and the row warps' 16-byte stage reads) and no `HMMA`, `LDSM`, `STSM`, `STTM`, `LDL` or
  `STL`. The landed kernels' cubins are byte-identical before and after this change or
  differ only by ptxas's recorded two-outcome uniform-register assignment (the MS-EDEN cubin
  in every compile session, the RHT-128 gradient amax cubin in one: 12 `UIADD3` / `UTCHMMA`
  operand pairs with `UR52` and `UR54` swapped, no instruction change; the flip reproduces
  between two compiles of the unchanged tree).

### group_row_cast_col_rht_quantize (`cutedsl_group_rht_quantize_row_col` vs `triton_group_rht_quantize_row_col`, `dynamic_rht=True`)

The V2 forward activation quantize: one pass over the packed activation
`x = (E * tokens, dim)` emits both NVFP4 operands -- the raw rows of `x_g` and the columns
of `x_g.t() @ R` (the 128-point randomized Hadamard transform along the tokens, live wgrad
signs) -- as RTNE FP4 codes with E4M3 block scales against the group's two amaxes from
`group_rht_amax`. The CuteDSL kernel is the amax kernel's mainloop (one TMA load per 128x128
tile, 8 UMMAs against the resident `R^T` operand, the row warps on the same shared-memory
stage) with the two amax epilogues replaced by the quantize epilogues: eight col warps, two
per TMEM quadrant, quantize the accumulator from TMEM under the requantize kernel's zero-fill
discipline (the epilogue zero-fills its TMEM chunks and every UMMA accumulates, so an
exact-zero column sum keeps Triton's `+0` nibble), and eight row warps run the RHT-16 fused
kernel's rowwise epilogue on the raw bfloat16 tile. Codes and scales are bitwise identical
across backends at every shape below, exact and fast math (`torch.equal` on all four outputs,
including 64 groups, an empty group, spare capacity rows, an all-zero group under all `-1`
signs and 671B down). Stochastic rounding is refused on this path (`ValueError`); the
baseline table's `rs` rows for this op are synthetic. Bandwidth counts the bfloat16 read plus
the FP4 codes and swizzled scales on both axes, 3.125 bytes per element; `use_fast_math=True`,
the recipe default and the baseline table's rows.

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_row_cast_col_rht_quantize
```

| model | projection | E | tokens | dim | cutedsl_us | triton_us | speedup | cutedsl_gbps |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| debugmodel | gate/up (w1/w3) | 4 | 256 | 256 | 12.70 | 14.90 | 1.17x | 64.5 |
| debugmodel | down (w2) | 4 | 256 | 256 | 12.56 | 14.99 | 1.19x | 65.2 |
| 16B | gate/up (w1/w3) | 4 | 12288 | 2048 | 103.24 | 184.91 | 1.79x | 3046.9 |
| 16B | down (w2) | 4 | 12288 | 1408 | 74.00 | 131.66 | 1.78x | 2922.6 |
| 671B | gate/up (w1/w3) | 4 | 32768 | 7168 | 812.62 | 1602.74 | 1.97x | 3613.0 |
| 671B | down (w2) | 4 | 32768 | 2048 | 242.88 | 466.62 | 1.92x | 3453.8 |

- Op time includes the same torch glue on both backends -- the sign-to-matrix build, no
  fills -- 5.2-6.9 us for CuteDSL (`torch.mul(h128, sign[None, :])`, the int8 sign operand on
  ATen's casting `gpu_kernel_impl`) and 5.9-8.3 us for Triton (`get_dynamic_rht_matrix`
  3.5-5.2 us plus its int8->bfloat16 copy 2.4-3.2 us). Kernel-only (profiler self CUDA time
  of the main kernel), CuteDSL vs Triton: 96.45 vs 176.43 us at 16B gate/up (1.83x), 67.40
  vs 123.72 us at 16B down (1.84x), 805.70 vs 1595.22 us at 671B gate/up (1.98x), 235.59 vs
  457.76 us at 671B down (1.94x). The metric excludes the 2.02-2.43 us `Memcpy DtoD` of the
  CuteDSL impl's `logical_packed_length.clone()`, which the Triton op does not issue.
- The op gates of the port plan were set at the earlier record's shapes, which are no longer
  in the table: 16B gate/up `x (5632, 2048)` 21.17 us against <= 21 us -- over by 0.17 us
  (0.8%); 16B down `x (8192, 1408)` 20.68 us -- under by 0.32 us; 671B gate/up
  `x (8192, 7168)` 64.10 us against <= 55 us -- over by 9.10 us (16.5%); 671B down
  `x (28672, 2048)` 65.31 us -- over by 10.31 us (18.7%). At 16B the kernel alone (15.98 us)
  was under the gate and the 5.2 us `torch.mul` the miss; at 671B the kernel alone
  (57.2-58.9 us) missed the gate by 4-7% on its own. At the recipe token counts the glue is
  1-9% of the op and the kernel-only ratio, 1.83-1.98x, is the table's. Per tile, derived
  from the 671B kernel-only times over 57344 and 16384 tiles on 152 CTAs after the 7.64 us
  floor (the debug-model row less its glue): ~2.1 us, ~2530 cycles at 1200 MHz (2560-2600 at
  16B) against the design's 1.2-1.5k forecast -- the epilogues bound the kernel (eight col
  warps quantizing four blocks per lane from TMEM, eight row warps quantizing 4 x 16 elements
  per thread from shared memory), not the 8-UMMA chain or HBM; that is the next lever.
- At 671B gate/up the op moves 2936 MB (the bfloat16 read plus 1.125 bytes per element of
  codes and scales) at 3613 GB/s, 45.5% of the 7936 GB/s peak; the kernel alone runs at
  3644 GB/s, 45.9%.
- On the CUDA 13.4 / nvidia-cutlass-dsl 4.6.0 / Triton 3.6.0 toolchain the CuteDSL kernel
  compiles and all four outputs stay bitwise equal to Triton 3.6.0's at the eleven smoke
  shapes, exact and fast, with both backends fed the CuteDSL amaxes (Triton 3.6.0 does not
  compile the RHT-128 amax, as in the baseline table); the checked-in cross-backend tests
  compute their amaxes with the Triton op and therefore run on Triton 3.8.0 only.
- `cuobjdump -res-usage` of the compiled kernel: REG 62, STACK 0, SHARED 1024 (static),
  LOCAL 0, at 640 threads and no `setmaxnreg`. Its SASS (2280 lines) holds 8 `UTCHMMA`, 2
  `UTCBAR`, 4 `LDTM` + 12 `STTM` (a warp's four-block TMEM read and its zero fills: two
  stages up front, four chunks per tile), 2 `STG.E.128` (the col code quarters), 4
  `STG.E.64` (the row code pairs), 80 `F2FP`, 23 `MUFU.RCP` and no `HMMA`, `LDSM`, `STSM`,
  `LDL` or `STL`.

## Triton baseline for the nine grouped kernels

The Triton numbers the nine ports above are measured against: the seven V2 ops
(`row_cast_quantize` -- shared with V1_REQUANT --, `row_cast_col_rht_amax` / `_quantize`
at RHT-128, `row_rht_col_rht_amax` / `_quantize_ms_eden`, `col_rht_requant_amax` /
`_requantize`) and the two V1_REQUANT columnwise requantizers (`col_cast_requant_amax` /
`_requantize`), one launch per kernel over the local expert stack at `E = 4`, fed the
inputs the recipe builds: 128-aligned uniform token groups with cumulative row-end offsets
for the token-jagged ops, the `(E, M, N)` weight stack for the expert-uniform ones, and the
packed `row_fp4_w` / `row_sf_w` plus the matching `*_requant_amax` output for the
requantizers. The weight rows are the record of a script that is not checked in. The
activation rows (`x`, `dy`) were re-measured with the four checked-in activation benches
at the recipe token counts of the methodology (medians of three passes, GPU 3, 1200 MHz
local cap; Triton 3.6.0 in the C1 container of the same node); the earlier record for
those rows used tokens per expert equal to the weight row count and square `dy` blocks.

- Op time, not kernel-only time: the RHT-128 ops build their rotation matrix in torch
  (`get_dynamic_rht_matrix`, 6-9 us at this clock) and the amax ops zero-fill their output
  (about 2 us) inside the timed region. Compare a port of one of these kernels on
  kernel-only time, or make it build the RHT matrix the same way.
- Bytes are the tensors each op reads and writes.
- Triton 3.6.0 fails to compile the two RHT-128 amax ops
  (`TritonNvidiaGPUOptimizeTMemLayoutsPass`: `parent layout must have at least rank >= 2`),
  so those two rows are Triton 3.8 only, and in the 3.6 column the quantizers downstream
  of them were fed amaxes computed in torch with the same RHT-128 chunking.
- `row_cast_col_rht_quantize` runs with `use_fast_math=True`, the recipe default. The
  earlier record's synthetic `rs` rows for it (the recipe's stochastic-rounding path is the
  V1_REQUANT backward at RHT-16) were dropped with the shape change: the checked-in bench
  does not produce them and the CuteDSL op refuses stochastic rounding on this path.

`w` is the weight `(E, M, N)` -- `M` rows, `N` columns, the wrappers' vocabulary; the recipe
calls the same stack `(E, N, K)` -- `x` the packed activation `(E * tokens, N)` and `dy`
the packed gradient `(E * tokens, M)`, with the per-model `tokens` of the methodology.

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

`row_cast_quantize` is the only one of the nine near bandwidth (31-36% of peak; the
CuteDSL port above reaches 59%). The no-transform `col_cast_requantize` runs 3-4x slower
than its RHT twin `col_rht_requantize` at 2-3% of peak: the compiled kernel holds 254
registers and 64 KB of shared memory and materialises its 128x128 tile three times (once
per layout conversion), so it is neither memory- nor occupancy-bound; that is the largest
headroom of the nine. The RHT-128 activation and gradient kernels sit at 9-23% of peak,
with 6-9 us of each op spent building each rotation matrix in torch.
