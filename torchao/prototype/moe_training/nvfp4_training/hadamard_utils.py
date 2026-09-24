"""RHT utility functions: Hadamard matrix construction, sign vector helpers, and Triton JIT helpers.

Provides get_wgrad_sign_vector, get_hadamard_matrix, get_rht_matrix, cast_to_fp4x2,
_compute_pid, and the NVFP4 quantization / scale-factor store helpers (formerly fp4_triton_ops).
"""

import functools
import math

import torch
from torch.utils._triton import has_triton

_TMA_WORKSPACES: dict = {}

# The RHT sign vector every NVFP4 path defaults to, matching TransformerEngine's
# get_wgrad_sign_vector (transformer_engine/pytorch/tensor/nvfp4_tensor.py). Lives here
# rather than beside the kernels so the reference and the tests can import it without
# pulling in the Triton or CuteDSL runtime.
DEFAULT_SIGN_VECTOR = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)


def _device_key(device) -> str:
    """Normalize device to a canonical string key (e.g. 'cuda' and 'cuda:0' → 'cuda:0').

    torch.device('cuda') and torch.device('cuda:0') stringify differently but refer
    to the same physical device. Without normalization, callers using one form miss
    cache entries created by callers using the other, causing spurious re-allocations
    inside FakeTensor tracing mode (which produce FakeTensors instead of real tensors).
    """
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        return f"cuda:{torch.cuda.current_device()}"
    return str(d)


def _prewarm_rht_matrix(
    sign_vector: tuple[int, ...],
    device: torch.device,
) -> None:
    get_rht_matrix(sign_vector, _device_key(device), torch.bfloat16, len(sign_vector))


def prepare_for_cuda_graph(
    device,
    nbytes: int = 131072,
    *,
    sign_vectors: tuple[tuple[int, ...], ...] | None = None,
) -> torch.Tensor:
    """Pre-allocate per-device persistent state required for torch.compile CUDA graphs.

    Must be called once per device before torch.compile to ensure allocations
    happen outside the CUDA graph pool. Subsequent calls return the cached TMA
    workspace (no new allocation). Kernels run sequentially in the same CUDA stream
    and safely alias the TMA buffer.

    Also pre-warms get_rht_matrix (lru_cache) to prevent pool-allocation errors
    during graph capture. Pass any RHT sign vectors used by the graph through
    sign_vectors so those cache entries are allocated before capture.
    """
    key = _device_key(device)
    if key not in _TMA_WORKSPACES:
        _TMA_WORKSPACES[key] = torch.empty(nbytes, dtype=torch.uint8, device=device)
    # Pre-warm every call because tests and callers may clear get_rht_matrix's
    # cache after the workspace has already been initialized. Use the same
    # canonical device-key string as runtime callers so the lru_cache key
    # matches.
    for sign_vector in sign_vectors or ():
        _prewarm_rht_matrix(tuple(sign_vector), key)
    # Warm H128 unconditionally (32 KB of bf16). Recipes with dynamic signs form
    # diag(signs) @ H128 per launch and so have no sign vector to enumerate here,
    # but the Hadamard itself must still be allocated outside the graph pool.
    get_hadamard_matrix(128, key, torch.bfloat16)
    return _TMA_WORKSPACES[key]


def get_wgrad_sign_vector(
    shape, device, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Generate a random {-1, 1} sign vector for the Hadamard transform."""
    return torch.where(
        torch.rand(shape, device=device) >= 0.5,
        torch.ones(shape, dtype=dtype, device=device),
        -torch.ones(shape, dtype=dtype, device=device),
    )


@functools.lru_cache(maxsize=None)
def get_hadamard_matrix(
    hadamard_dimension: int, device, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Normalized Sylvester-ordered Hadamard matrix. Supports 16 and 128.

    128 is built by Kronecker-recursing the hardcoded 16x16 block:
    ``H_128 = H_16 (x) H_8`` where ``H_8 = H_16[:8, :8]`` is its Sylvester
    sub-block. That reproduces the popcount ordering the V2 reference uses, so a
    128-row RHT built here cancels against one built by the oracle.

    Cached because V2 forms ``diag(signs) @ H`` per launch from a resampled sign
    tensor: only the fixed Hadamard may be memoized, never the product.
    """
    if hadamard_dimension not in (16, 128):
        raise ValueError("Only hadamard dimension 16 or 128 is supported.")
    hadamard_scale = 1 / math.sqrt(hadamard_dimension)
    h16 = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
            [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
            [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
            [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1],
            [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1],
            [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1],
            [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1],
            [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1],
            [1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1],
            [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1],
            [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1],
            [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
            [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1],
        ],
        dtype=dtype,
        device=device,
    )
    if hadamard_dimension == 128:
        return torch.kron(h16, h16[:8, :8]) * hadamard_scale
    return h16 * hadamard_scale


def get_dynamic_rht_matrix(
    sign_vector: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Build ``diag(sign_vector) @ H`` from a live device buffer, caching only ``H``.

    The counterpart to ``get_rht_matrix`` for recipes whose signs are resampled
    during the run. ``get_rht_matrix`` is keyed on the sign *tuple*, which is sound
    only when the key set is a single element; keyed on a resampled vector its
    ``maxsize=None`` cache grows one entry per resample for the run's lifetime.
    Here only the fixed Hadamard is memoized and the product is formed per call.

    ``sign_vector`` is a device tensor rather than a tuple on purpose: it is a
    fixed-shape buffer updated in place by the cadence manager, so its address
    stays stable under CUDA-graph capture.
    """
    if sign_vector.ndim != 1:
        raise ValueError(f"sign_vector must be 1D, got {sign_vector.ndim}D")
    hadamard_dimension = sign_vector.shape[0]
    if hadamard_dimension not in (16, 128):
        raise ValueError(
            f"sign_vector length must be 16 or 128, got {hadamard_dimension}"
        )
    return sign_vector.to(dtype)[:, None] * get_hadamard_matrix(
        hadamard_dimension, device=sign_vector.device, dtype=dtype
    )


@functools.lru_cache(maxsize=None)
def get_rht_matrix(
    sign_vector: tuple[int, ...],
    device,
    dtype: torch.dtype,
    hadamard_dimension: int,
    /,
) -> torch.Tensor:
    """Construct an RHT matrix from an explicit sign vector. Avoid default arguments
    and require positional arguments to ensure lru_cache keys are unambiguous.

    ``sign_vector`` must be a hashable-by-value tuple. A ``torch.Tensor`` is rejected
    explicitly rather than left to fail on its own: tensors hash by *identity*, not by
    value, so a dynamic sign buffer would be accepted here and then -- because the
    cadence manager updates it in place, leaving its id unchanged -- keep returning
    the matrix built from its very first contents. The transform would stop cancelling
    and the gradients would be silently wrong, with no error anywhere. Recipes with
    resampled signs must use ``get_dynamic_rht_matrix``.
    """
    if isinstance(sign_vector, torch.Tensor):
        raise TypeError(
            "get_rht_matrix caches by value and cannot take a torch.Tensor: tensors "
            "hash by identity, so an in-place resample would silently return a stale "
            "RHT matrix. Use get_dynamic_rht_matrix for resampled sign vectors."
        )
    if len(sign_vector) != hadamard_dimension:
        raise ValueError(
            f"Expected sign_vector length {hadamard_dimension}, got {len(sign_vector)}"
        )
    signs = torch.tensor(sign_vector, dtype=dtype, device=device)
    sign_matrix = signs * torch.eye(hadamard_dimension, dtype=dtype, device=device)
    return sign_matrix @ get_hadamard_matrix(
        hadamard_dimension, device=device, dtype=dtype
    )


if has_triton():
    import triton
    import triton.language as tl

    # Elements in one swizzled scale tile (128 outer rows x 4 inner
    # scale-columns). Shared scope because the store helper and the grouped
    # wrapper that offsets into it must agree on this by construction. Must be a
    # tl.constexpr instance, not a bare int: @triton.jit refuses non-constexpr
    # module globals.
    TILE_ELEMS = tl.constexpr(32 * 16)

    @triton.jit
    def _compute_pid(tile_id, num_pid_in_group, num_pid_n, GROUP_SIZE_N: tl.constexpr):
        r"""Convert flat tile_id to (pid_n, pid_m) with L2-cache-friendly grouping."""
        group_id = tile_id // num_pid_in_group
        first_pid_n = group_id * GROUP_SIZE_N
        group_size_n = tl.minimum(num_pid_n - first_pid_n, GROUP_SIZE_N)
        pid_n = first_pid_n + (tile_id % group_size_n)
        pid_m = (tile_id % num_pid_in_group) // group_size_n
        return pid_n, pid_m

    @triton.jit
    def convert_8xfp32_to_4xfp4_packed(x_pairs):
        """Convert 8 FP32 values to 4 packed FP4 bytes using round-to-nearest.
        Calls four cvt.rn instructions, each packing two FP32 values into one packed int8."""
        x_fp4x2 = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b8 byte0, byte1, byte2, byte3;
            cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
            cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
            cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
            cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
            mov.b32 $0, {byte0, byte1, byte2, byte3};
            }
            """,
            constraints=("=r,r,r,r,r,r,r,r,r"),
            args=x_pairs,
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )
        return x_fp4x2

    @triton.jit
    def convert_4xfp4_packed_to_8xfp32(bytes4):
        """Inverse of ``convert_8xfp32_to_4xfp4_packed``.

        One uint8 tile in; two fp32 tiles out (low-nibble columns, high-nibble
        columns), interleaved back into the original column order. FP4 -> FP16 is a
        widening conversion and therefore exact, so this round-trips the packer
        bit-for-bit.
        """
        lo, hi = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b8  b0, b1, b2, b3;
            .reg .b32 p0, p1, p2, p3;
            .reg .b16 l0, h0, l1, h1, l2, h2, l3, h3;
            mov.b32 {b0, b1, b2, b3}, $8;          // 4 packed bytes -> 4 x b8
            cvt.rn.f16x2.e2m1x2 p0, b0;            // byte -> 2 f16 (widening = exact)
            cvt.rn.f16x2.e2m1x2 p1, b1;
            cvt.rn.f16x2.e2m1x2 p2, b2;
            cvt.rn.f16x2.e2m1x2 p3, b3;
            mov.b32 {l0, h0}, p0;                  // split f16x2 -> lo/hi halves
            mov.b32 {l1, h1}, p1;
            mov.b32 {l2, h2}, p2;
            mov.b32 {l3, h3}, p3;
            cvt.f32.f16 $0, l0;   cvt.f32.f16 $4, h0;
            cvt.f32.f16 $1, l1;   cvt.f32.f16 $5, h1;
            cvt.f32.f16 $2, l2;   cvt.f32.f16 $6, h2;
            cvt.f32.f16 $3, l3;   cvt.f32.f16 $7, h3;
            }
            """,
            constraints="=r,=r,=r,=r,=r,=r,=r,=r,r",
            args=[bytes4],
            dtype=(tl.float32, tl.float32),
            is_pure=True,
            pack=4,
        )
        return tl.interleave(lo, hi)

    @triton.jit
    def convert_8xfp32_to_4xfp4_packed_rs(x_pairs, rbits):
        """Convert 8 FP32 values to 4 packed FP4 bytes using stochastic rounding.
        Calls two cvt.rs instructions, each packing four FP32 values into one packed int8."""
        x_fp4x2 = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 half0, half1;
            cvt.rs.satfinite.e2m1x4.f32 half0, {$6, $2, $5, $1}, $9;
            cvt.rs.satfinite.e2m1x4.f32 half1, {$8, $4, $7, $3}, $10;
            mov.b32 $0, {half0, half1};
            }
            """,
            constraints=("=r,r,r,r,r,r,r,r,r,r,r,r,r"),
            args=[x_pairs[0], x_pairs[1], rbits],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )
        return x_fp4x2

    @triton.jit
    def _rcp_approx_ftz(x):
        """``rcp.approx.ftz.f32``: TE's fast-math reciprocal.

        No ``tl`` builtin reaches this instruction -- libdevice offers only the correctly
        rounded ``rcp_rn/rd/ru/rz``, and ``fast_dividef`` lowers to ``div.approx.f32``,
        which is a reciprocal *followed by a multiply* and so does not agree bit for bit.
        TransformerEngine gets here through CUTLASS ``reciprocal_approximate_ftz`` and the
        CuteDSL kernels through ``cute.arch.rcp_approx``; all three lower to one MUFU.RCP.
        """
        return tl.inline_asm_elementwise(
            asm="rcp.approx.ftz.f32 $0, $1;",
            constraints="=r,r",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _pack_fp4(
        scaled,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        STOCHASTIC_ROUNDING: tl.constexpr,
        seed_ptr,
        offset_base_ptr,
        tile_id,
    ):
        """Pack scaled float32 values to FP4 with optional stochastic rounding.

        seed_ptr: pointer to int64 per-call seed (unused when STOCHASTIC_ROUNDING=False).
        offset_base_ptr: pointer to int64 offset base; only the low 32 bits occupy the
            low 32 bits of the Philox counter (unused when STOCHASTIC_ROUNDING=False).
        tile_id: persistent-grid tile index; used to form the globally unique element index
            that occupies the high 32 bits of the Philox counter.
        """
        scaled_pairs = scaled.reshape(BLOCK_N, BLOCK_M // 2, 2).split()
        if STOCHASTIC_ROUNDING:
            seed = tl.load(seed_ptr)
            offset_base = tl.load(offset_base_ptr).to(tl.uint64) & 0xFFFFFFFF
            BLOCK_M_PACKED: tl.constexpr = BLOCK_M // 2
            local_n = tl.arange(0, BLOCK_N)[:, None]
            local_m = tl.arange(0, BLOCK_M_PACKED)[None, :]
            local_pos = local_n * BLOCK_M_PACKED + local_m
            linear_idx = tl.cast(tile_id, tl.int64) * (
                BLOCK_N * BLOCK_M_PACKED
            ) + tl.cast(local_pos, tl.int64)
            offset = (linear_idx.to(tl.uint64) << 32) | offset_base
            rbits = tl.randint(seed, offset)
            return convert_8xfp32_to_4xfp4_packed_rs(scaled_pairs, rbits)
        else:
            return convert_8xfp32_to_4xfp4_packed(scaled_pairs)

    @triton.jit
    def _nvfp4_global_scales(global_amax, FP8_E4M3_MAX: tl.constexpr):
        """Exact FP32 per-tensor ``(encode, decode)`` scales for a given FP8 ceiling.

        Factored out of ``_nvfp4_quantize`` because the requantization kernels must
        decode the saved forward weight with *exactly* the scale the forward encoded
        it with. If the two ever drift, the requantization amax stops bounding the
        tensor being quantized and both gradients come out biased low.

        ``FP8_E4M3_MAX`` is 448 for plain NVFP4 casts and 256 for MS-EDEN, which is
        what makes their per-tensor decode numerators 2688 and 1536 respectively.
        """
        FP4_E2M1_MAX: tl.constexpr = 6.0
        FP32_MAX: tl.constexpr = torch.finfo(torch.float32).max

        is_zero = global_amax == 0.0
        safe_global_amax = tl.where(is_zero, 1.0, global_amax)
        # TE's scale bytes follow correctly-rounded FP32 division; Triton's normal
        # "/" lowers through a reciprocal path that can flip FP8 midpoint ties.
        global_scale_num = tl.full(
            safe_global_amax.shape,
            FP8_E4M3_MAX * FP4_E2M1_MAX,
            safe_global_amax.dtype,
        )
        candidate = tl.div_rn(global_scale_num, safe_global_amax)
        candidate = tl.minimum(candidate, FP32_MAX)
        candidate = tl.where(candidate == 0.0, 1.0, candidate)
        global_encode_scale = tl.where(is_zero, 1.0, candidate)
        one = tl.full(
            safe_global_amax.shape, 1.0, safe_global_amax.dtype
        )  # div_rn needs a tensor numerator
        return global_encode_scale, tl.div_rn(one, global_encode_scale)

    @triton.jit
    def _nvfp4_dequantize(
        qa_fp4, sfa, gds, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        """Decode one packed NVFP4 tile back to FP32. The inverse of the encode half
        of ``_nvfp4_quantize``.

        Two steps, both of which the name has to carry: **unpack** the two FP4 codes
        per byte into FP32 lanes, then **rescale** each 16-element vector by its FP8
        block scale and the per-tensor global decode scale. It is not a rescale alone
        -- the unpack is why the output has 2x the elements of the input.

        Used to dequantize the quantize forward weight tile and in four or six 
        quantization in forward.

        Args:
            qa_fp4: ``(BLOCK_M, BLOCK_N//2)`` packed uint8 FP4 codes.
            sfa: ``(BLOCK_M, BLOCK_N//16)`` per-vector FP8 scale factors, already
                un-swizzled (see ``_load_scales_swizzle``).
            gds: per-tensor global *decode* scale from ``_nvfp4_global_scales``.

        Returns ``(BLOCK_M, BLOCK_N//16, 16)`` FP32; reshape to ``(BLOCK_M, BLOCK_N)``
        for a plain 2D view.

        Used by:
            ``_reconstruct_qdq_weight_tile`` below, its only caller -- which is to say
            every lazy requantization kernel, and nothing else. The forward quantize
            path never decodes.
        """
        qa_f32_unpacked = convert_4xfp4_packed_to_8xfp32(qa_fp4)
        qa_f32_blocked = qa_f32_unpacked.reshape(BLOCK_M, BLOCK_N // 16, 16)
        return qa_f32_blocked * sfa[:, :, None] * gds

    @triton.jit
    def _reconstruct_qdq_weight_tile(
        qw_ptr,
        sfw_ptr,
        global_amax_ptr,
        expert,
        pid_m,
        pid_n,
        M,
        N,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Reconstruct one ``W_qdq`` tile of expert ``expert`` on chip, in bf16.

        Reads the rowwise operand §11.1 saved -- ``(E, M, N//2)`` packed codes plus
        ``(E, M//128, N//64, 32, 16)`` swizzled scales -- and returns a
        ``(BLOCK_M, BLOCK_N)`` bf16 tile, un-transposed and un-rotated.

        Every lazy requantization path starts here: the cast one (§11.6/§11.7) then
        transposes, the RHT one (§11.4/§11.5) transposes and rotates. It is one
        function rather than one per file because the amax pass and the quantize pass
        must reconstruct bit-identically -- if they drift, the amax stops bounding the
        tensor being quantized and both gradients come out biased low. The two files
        select between each other by recipe off the same saved weight, so that
        invariant has to hold across them, not just within each.

        The bf16 round-through here is required, not incidental: the reference takes
        it, and dropping it makes the codes differ.

        Used by:
            ``_load_requant_weight_tile`` in ``group_col_cast_requantize_triton.py``,
            feeding ``_group_col_cast_requant_amax_kernel`` (§11.6) and
            ``_group_col_cast_requantize_kernel`` (§11.7).

            ``_load_rht_requant_weight_tile`` in ``group_col_rht_requantize_triton.py``,
            feeding ``_group_col_rht_requant_amax_kernel`` (§11.4) and
            ``_group_col_rht_requantize_kernel`` (§11.5).

            All four are lazy-requantization kernels: they rebuild the backward
            operand from the rowwise weight §11.1 saved, rather than re-reading the
            original BF16 weight. Any new kernel that decodes a saved forward weight
            belongs on this helper too.
        """
        # Load packed fp4 codes
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        packed_inner = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
        packed_offsets = offs_m[:, None] * (N // 2) + packed_inner[None, :]
        qw_expert_ptr = qw_ptr + expert * M * (N // 2)
        qw = tl.load(qw_expert_ptr + packed_offsets)

        # Load swizzled scales
        sfw_expert_stride = (M // 128) * (N // 64) * 32 * 16
        sfw_expert_ptr = sfw_ptr + expert * sfw_expert_stride
        sfw = _load_scales_swizzle(
            sfw_expert_ptr,
            pid_m,
            pid_n,
            M,
            N,
            BLOCK_M,
            BLOCK_N,
        )

        # Load global amax precomputed in forward pass. Per expert -- a reduction
        # across experts would decode every expert with expert 0's scale. This must be
        # the exact scale the forward encoded with, which is why it goes through
        # _nvfp4_global_scales rather than an open-coded divide.
        FP8_E4M3_MAX: tl.constexpr = 448.0
        amax_w = tl.load(global_amax_ptr + expert)
        _, global_decode_scale = _nvfp4_global_scales(amax_w, FP8_E4M3_MAX)
        dequant_w = _nvfp4_dequantize(qw, sfw, global_decode_scale, BLOCK_M, BLOCK_N)

        # A NaN or inf global_amax must reconstruct to zero, not propagate.
        valid_amax = (amax_w == amax_w) & (tl.abs(amax_w) != float("inf"))
        dequant_w = tl.where(valid_amax, dequant_w, 0.0).to(tl.bfloat16)
        return tl.reshape(dequant_w, [BLOCK_M, BLOCK_N])

    @triton.jit
    def _nvfp4_quantize(
        a_t_rht,
        global_amax,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        FAST_MATH: tl.constexpr = False,
        FP8_E4M3_MAX: tl.constexpr = 448.0,
        CLAMP: tl.constexpr = True,
    ):
        """Compute per-vector FP8 scale factors and scaled FP32 values ready for FP4 packing.

        FAST_MATH takes TE's approximate reciprocal for the per-vector encode scale
        instead of a correctly rounded divide, matching TransformerEngine under
        NVTE_USE_FAST_MATH=1. It applies to the columnwise and rowwise quantize alike;
        the other half of TE's fast math -- consuming the RHT accumulator without
        rounding it through bfloat16 -- belongs to the caller, because only the
        columnwise path has an accumulator.

        CLAMP=False returns the scaled values unclamped, for MS-EDEN's correction
        (``_ms_eden_correction_with_sr``), whose kernel clamps its own copy for the
        packer; every other caller packs the values, where ``cvt.rn.satfinite`` saturates
        anyway.
        """
        FP4_E2M1_MAX: tl.constexpr = 6.0
        FP32_MAX: tl.constexpr = torch.finfo(torch.float32).max

        a_vecs = tl.reshape(a_t_rht, [BLOCK_N, BLOCK_M // 16, 16])
        vec_max = tl.max(tl.abs(a_vecs), axis=-1, keep_dims=True)

        # Shared with the requantization kernels: they decode the saved forward
        # weight with this exact scale, so it must be one implementation.
        global_encode_scale, global_decode_scale = _nvfp4_global_scales(
            global_amax, FP8_E4M3_MAX
        )

        # Cap at FP8_E4M3_MAX only, no lower clamp: pvscale is non-negative and TE
        # emits a zero per-vector scale for zero/near-zero vectors, so pinning small
        # scales to a nonzero floor would diverge from the TE ground truth.
        global_encode_scale_over_fp4max = global_encode_scale * (1.0 / FP4_E2M1_MAX)
        pvscale = vec_max.to(tl.float32) * global_encode_scale_over_fp4max
        pvscale = tl.minimum(pvscale, FP8_E4M3_MAX)
        pvscale_fp8 = pvscale.to(tl.float8e4nv)
        scale_inv = tl.reshape(pvscale_fp8, [BLOCK_N, BLOCK_M // 16])

        denom = pvscale_fp8.to(tl.float32) * global_decode_scale
        if FAST_MATH:
            encode_scale = tl.minimum(_rcp_approx_ftz(denom), FP32_MAX)
        else:
            encode_num = tl.full(denom.shape, 1.0, tl.float32)
            encode_scale = tl.minimum(tl.div_rn(encode_num, denom), FP32_MAX)

        scaled = a_vecs * encode_scale
        if CLAMP:
            scaled = tl.clamp(scaled, -FP4_E2M1_MAX, FP4_E2M1_MAX)
        scaled = tl.reshape(scaled, [BLOCK_N, BLOCK_M])
        return scale_inv, scaled

    @triton.jit
    def _swizzle_scales(
        scale_inv, BLOCK_OUTER: tl.constexpr, BLOCK_INNER: tl.constexpr
    ):
        """Reshape (BLOCK_OUTER, BLOCK_INNER//16) → (BLOCK_OUTER//128, BLOCK_INNER//64, 32, 16).

        Columnwise: _swizzle_scales(scale_inv, BLOCK_N, BLOCK_M)
        Rowwise:    _swizzle_scales(scale_inv, BLOCK_M, BLOCK_N)
        """
        scale_inv = tl.reshape(
            scale_inv, [BLOCK_OUTER // 128, 4, 32, BLOCK_INNER // 64, 4]
        )
        scale_inv = tl.permute(scale_inv, [0, 3, 2, 1, 4])
        return tl.reshape(scale_inv, [BLOCK_OUTER // 128, BLOCK_INNER // 64, 32, 16])

    @triton.jit
    def _store_scales_swizzle(
        scale_inv,
        sf_ptr,
        pid_outer,
        pid_inner,
        OUTER,
        INNER,
        BLOCK_OUTER: tl.constexpr,
        BLOCK_INNER: tl.constexpr,
        base_elems=0,
    ):
        """Store pre-swizzled scale factors in tile-major layout (OUTER//128, INNER//64, 32, 16).

        Columnwise: _store_scales_swizzle(sf, ptr, pid_n, pid_m, N, M, BLOCK_N, BLOCK_M)
        Rowwise:    _store_scales_swizzle(sf, ptr, pid_m, pid_n, M, N, BLOCK_M, BLOCK_N)

        *INNER* is the extent this tiling runs over, which is the whole tensor
        unless the inner axis is grouped -- see _store_grouped_scales_swizzle,
        the only caller that passes a nonzero *base_elems*.
        """
        VEC_ELEMS_FP8: tl.constexpr = 16
        BLOCK_OUTER_TILES: tl.constexpr = BLOCK_OUTER // 128
        BLOCK_INNER_TILES: tl.constexpr = BLOCK_INNER // 64
        FLAT_TILE: tl.constexpr = BLOCK_OUTER_TILES * BLOCK_INNER_TILES * TILE_ELEMS

        INNER_TILES = tl.cdiv(INNER, 64)
        OUTER_TILES = tl.cdiv(OUTER, 128)

        rb_base = pid_outer * BLOCK_OUTER_TILES
        cb_base = pid_inner * BLOCK_INNER_TILES

        rb_idx = rb_base + tl.arange(0, BLOCK_OUTER_TILES)
        cb_idx = cb_base + tl.arange(0, BLOCK_INNER_TILES)
        elem_idx = tl.arange(0, TILE_ELEMS)

        offsets = (
            base_elems
            + rb_idx[:, None, None] * INNER_TILES * TILE_ELEMS
            + cb_idx[None, :, None] * TILE_ELEMS
            + elem_idx[None, None, :]
        )
        mask = (
            (rb_idx[:, None, None] < OUTER_TILES)
            & (cb_idx[None, :, None] < INNER_TILES)
            & (elem_idx[None, None, :] < TILE_ELEMS)
        )

        flat_ptrs = sf_ptr + tl.reshape(offsets, (FLAT_TILE,))
        flat_val = tl.reshape(scale_inv, (FLAT_TILE,))
        flat_msk = tl.reshape(mask, (FLAT_TILE,))

        tl.multiple_of(flat_ptrs, VEC_ELEMS_FP8)
        tl.max_contiguous(flat_ptrs, VEC_ELEMS_FP8)
        tl.store(flat_ptrs, flat_val, mask=flat_msk)

    @triton.jit
    def _load_scales_swizzle(
        sf_ptr,
        pid_outer,
        pid_inner,
        OUTER,
        INNER,
        BLOCK_OUTER: tl.constexpr,
        BLOCK_INNER: tl.constexpr,
    ):
        """Load one swizzled scale tile and return its plain ``(outer, inner//16)`` view.

        The read counterpart of ``_store_scales_swizzle``, needed by the
        requantization kernels: they consume the swizzled E4M3 scales the forward
        wrote and must undo the SWIZZLE_32_4_4 permutation on chip to reconstruct
        ``W_qdq``.
        """
        BLOCK_OUTER_TILES: tl.constexpr = BLOCK_OUTER // 128
        BLOCK_INNER_TILES: tl.constexpr = BLOCK_INNER // 64
        FLAT_TILE: tl.constexpr = BLOCK_OUTER_TILES * BLOCK_INNER_TILES * TILE_ELEMS

        INNER_TILES = tl.cdiv(INNER, 64)
        OUTER_TILES = tl.cdiv(OUTER, 128)
        rb_idx = pid_outer * BLOCK_OUTER_TILES + tl.arange(0, BLOCK_OUTER_TILES)
        cb_idx = pid_inner * BLOCK_INNER_TILES + tl.arange(0, BLOCK_INNER_TILES)
        elem_idx = tl.arange(0, TILE_ELEMS)
        offsets = (
            rb_idx[:, None, None].to(tl.int64) * INNER_TILES * TILE_ELEMS
            + cb_idx[None, :, None] * TILE_ELEMS
            + elem_idx[None, None, :]
        )
        mask = (
            (rb_idx[:, None, None] < OUTER_TILES)
            & (cb_idx[None, :, None] < INNER_TILES)
            & (elem_idx[None, None, :] < TILE_ELEMS)
        )
        swizzled = tl.load(
            sf_ptr + tl.reshape(offsets, (FLAT_TILE,)),
            mask=tl.reshape(mask, (FLAT_TILE,)),
            other=0.0,
        )
        swizzled = tl.reshape(
            swizzled,
            [BLOCK_OUTER_TILES, BLOCK_INNER_TILES, 32, 4, 4],
        )
        plain = tl.permute(swizzled, [0, 3, 2, 1, 4])
        return tl.reshape(plain, [BLOCK_OUTER, BLOCK_INNER // 16])

    @triton.jit
    def _store_grouped_scales_swizzle(
        scale_inv,
        sf_ptr,
        pid_outer,
        pid_inner,
        offsets_ptr,
        group_idx,
        OUTER,
        BLOCK_OUTER: tl.constexpr,
        BLOCK_INNER: tl.constexpr,
    ):
        """Store scales whose *inner* (64-blocked) axis is the grouped one.

        A grouped GEMM reads each group's block scales as an independently
        blocked buffer, the buffers concatenated flat -- what
        `_check_scales_blocked` calls `rounded_up_per_group(K/blocksize, 4)`
        (pytorch aten/src/ATen/native/cuda/GroupedBlas.cpp:401-403). So the
        tiling has to restart at every group boundary. Running it over the
        packed extent instead scatters each group's tiles through the buffer,
        and the GEMM then reads all of them from the wrong offset.

        The outer axis needs no equivalent: it is the slowest-varying term, so a
        group occupying whole 128-row tiles is already contiguous. That is why
        the rowwise store -- where the grouped axis is the outer one -- calls
        _store_scales_swizzle directly.

        Requires 128-aligned group boundaries, which is what makes the
        group-local tile index exact. The pad-128 token dispatcher guarantees
        it through the final logical group-end offset.
        """
        group_start = tl.load(
            offsets_ptr + group_idx - 1, mask=group_idx > 0, other=0
        ).to(tl.int32)
        group_end = tl.load(offsets_ptr + group_idx).to(tl.int32)
        _store_scales_swizzle(
            scale_inv,
            sf_ptr,
            pid_outer,
            pid_inner - group_start // BLOCK_INNER,
            OUTER,
            group_end - group_start,
            BLOCK_OUTER,
            BLOCK_INNER,
            base_elems=tl.cdiv(OUTER, 128) * (group_start // 64) * TILE_ELEMS,
        )

else:

    def _compute_pid(*args, **kwargs):
        raise RuntimeError("_compute_pid requires Triton")

    def convert_8xfp32_to_4xfp4_packed(*args, **kwargs):
        raise RuntimeError("convert_8xfp32_to_4xfp4_packed requires Triton")

    def convert_4xfp4_packed_to_8xfp32(*args, **kwargs):
        raise RuntimeError("convert_4xfp4_packed_to_8xfp32 requires Triton")

    def convert_8xfp32_to_4xfp4_packed_rs(*args, **kwargs):
        raise RuntimeError("convert_8xfp32_to_4xfp4_packed_rs requires Triton")

    def _pack_fp4(*args, **kwargs):
        raise RuntimeError("_pack_fp4 requires Triton")

    def _nvfp4_global_scales(*args, **kwargs):
        raise RuntimeError("_nvfp4_global_scales requires Triton")

    def _nvfp4_dequantize(*args, **kwargs):
        raise RuntimeError("_nvfp4_dequantize requires Triton")

    def _reconstruct_qdq_weight_tile(*args, **kwargs):
        raise RuntimeError("_reconstruct_qdq_weight_tile requires Triton")

    def _nvfp4_quantize(*args, **kwargs):
        raise RuntimeError("_nvfp4_quantize requires Triton")

    def _swizzle_scales(*args, **kwargs):
        raise RuntimeError("_swizzle_scales requires Triton")

    def _store_scales_swizzle(*args, **kwargs):
        raise RuntimeError("_store_scales_swizzle requires Triton")

    def _load_scales_swizzle(*args, **kwargs):
        raise RuntimeError("_load_scales_swizzle requires Triton")

    def _store_grouped_scales_swizzle(*args, **kwargs):
        raise RuntimeError("_store_grouped_scales_swizzle requires Triton")
