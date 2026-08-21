# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the improved torchao::_triton_permute_bwd scatter and its use as
the backward of the BF16 permute_and_pad path (forward stays eager)."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Test requires CUDA", allow_module_level=True)

pytest.importorskip("triton", reason="Triton required to run this test")

from torch._subclasses.fake_tensor import FakeTensorMode

from torchao.prototype.moe_training.ep.kernels import generate_permute_indices
from torchao.prototype.moe_training.ep.permute import (
    _triton_permute_bwd,
    permute_and_pad,
    permute_mxfp8_fwd_hp_bwd,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.utils import is_cuda_version_at_least, is_sm_at_least_100

_DEVICE = "cuda"


def _make_map(num_tokens, num_local_experts, ep_degree, alignment, seed=0):
    """Realistic dst-to-src map via generate_permute_indices; valid entries
    are a bijection onto [0, num_tokens)."""
    torch.manual_seed(seed)
    num_groups = num_local_experts * ep_degree
    weights = torch.rand(num_groups) + 0.05
    counts = (weights / weights.sum() * num_tokens).int()
    counts[-1] = num_tokens - counts[:-1].sum()
    counts = counts.to(device=_DEVICE, dtype=torch.int32)
    max_len = num_tokens + num_local_experts * alignment
    permuted_indices, _, _ = generate_permute_indices(
        counts, num_local_experts, ep_degree, max_len, alignment
    )
    return permuted_indices, num_tokens


def _eager_scatter(x, dst_to_src, original_rows):
    """The exact pre-improvement production inverse: sentinel-inclusive
    index_put, then slice the sentinel row off."""
    out_padded = x.new_zeros((original_rows + 1, x.shape[-1]))
    out_padded[dst_to_src, :] = x
    return out_padded[:-1]


@pytest.mark.parametrize("hidden", [128, 300, 2048, 4096, 7168])
def test_scatter_matches_eager(hidden):
    """Improved kernel bitwise vs the production eager scatter, including
    hidden sizes above one 256-column block (4096, 7168 -- the DSv3-671B
    width -- exercise the multi-column-program path and the masked final
    block)."""
    dst_to_src, num_tokens = _make_map(256, 4, 2, 32)
    x = torch.randn(dst_to_src.shape[0], hidden, device=_DEVICE, dtype=torch.bfloat16)
    out = _triton_permute_bwd(x, dst_to_src, num_tokens, hidden)
    torch.testing.assert_close(
        out, _eager_scatter(x, dst_to_src, num_tokens), rtol=0, atol=0
    )


def test_scatter_invalid_entries_write_nothing():
    """-1 and positive out-of-range map entries are both invalid: they write
    nothing (out-of-range cannot address memory outside the output)."""
    dst_to_src, num_tokens = _make_map(256, 4, 1, 32)
    x = torch.randn(dst_to_src.shape[0], 128, device=_DEVICE, dtype=torch.bfloat16)

    corrupt = dst_to_src.clone()
    valid_pos = (corrupt >= 0).nonzero()[0].item()
    corrupt[valid_pos] = num_tokens + 7  # out of range for the output
    as_invalid = dst_to_src.clone()
    as_invalid[valid_pos] = -1

    out = _triton_permute_bwd(x, corrupt, num_tokens, 128)
    ref = _eager_scatter(x, as_invalid, num_tokens)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "num_local_experts,ep_degree", [(4, 1), (8, 2)], ids=["ep1", "ep2"]
)
def test_permute_and_pad_backward_bitwise(num_local_experts, ep_degree):
    """permute_and_pad's Triton scatter backward is bitwise identical to the
    pure-eager autograd (aten indexing_backward) it replaces, under a random
    cotangent; forward outputs and the 5-tuple contract are unchanged."""
    counts = torch.tensor([40, 0, 17, 71], dtype=torch.int32, device=_DEVICE)
    counts = counts.repeat(num_local_experts * ep_degree // 4)
    num_tokens = int(counts.sum())
    x = torch.randn(
        num_tokens, 256, device=_DEVICE, dtype=torch.bfloat16, requires_grad=True
    )
    x_ref = x.detach().clone().requires_grad_(True)

    shape, permuted, indices, counts_padded, offsets = permute_and_pad(
        x, counts, ep_degree, num_local_experts, 32
    )

    # Pure-eager reference: the exact pre-improvement autograd path.
    x_ref_padded = torch.vstack((x_ref, x_ref.new_zeros((1, 256))))
    permuted_ref = x_ref_padded[indices, :]

    assert shape == torch.Size((num_tokens + 1, 256))
    torch.testing.assert_close(permuted, permuted_ref, rtol=0, atol=0)

    grad_seed = torch.randn_like(permuted)
    permuted.backward(grad_seed)
    permuted_ref.backward(grad_seed)
    assert x.grad.shape == (num_tokens, 256)
    torch.testing.assert_close(x.grad, x_ref.grad[:num_tokens], rtol=0, atol=0)
    # The eager reference accumulates all -1 grads into the sentinel row; the
    # scatter backward must simply drop them.
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)


def test_permute_and_pad_strided_input():
    """A strided x still works: the eager forward accepts arbitrary 2D
    strides, and the backward makes the cotangent contiguous."""
    counts = torch.tensor([40, 0, 17, 71], dtype=torch.int32, device=_DEVICE)
    num_tokens = int(counts.sum())
    x_big = torch.randn(
        num_tokens, 320, device=_DEVICE, dtype=torch.bfloat16, requires_grad=True
    )
    x = x_big[:, ::2]
    _, permuted, *_ = permute_and_pad(x, counts, 1, 4, 32)
    permuted.sum().backward()
    assert x_big.grad is not None


def test_permute_and_pad_compile_fullgraph():
    counts = torch.tensor([40, 0, 17, 71], dtype=torch.int32, device=_DEVICE)
    num_tokens = int(counts.sum())
    x = torch.randn(
        num_tokens, 256, device=_DEVICE, dtype=torch.bfloat16, requires_grad=True
    )
    x_ref = x.detach().clone().requires_grad_(True)

    def fn(t):
        return permute_and_pad(t, counts, 1, 4, 32)[1]

    eager_out = fn(x_ref)
    eager_out.sum().backward()
    compiled_out = torch.compile(fn, fullgraph=True)(x)
    compiled_out.sum().backward()
    torch.testing.assert_close(compiled_out, eager_out, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)


def test_scatter_cuda_graph_and_fake_tensor():
    dst_to_src, num_tokens = _make_map(256, 4, 1, 32)
    x = torch.randn(dst_to_src.shape[0], 128, device=_DEVICE, dtype=torch.bfloat16)
    ref = _triton_permute_bwd(x, dst_to_src, num_tokens, 128)

    graph = torch.cuda.CUDAGraph()
    out = None
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _triton_permute_bwd(x, dst_to_src, num_tokens, 128)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        out = _triton_permute_bwd(x, dst_to_src, num_tokens, 128)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)

    with FakeTensorMode():
        fx = torch.empty(320, 128, device=_DEVICE, dtype=torch.bfloat16)
        fmap = torch.empty(320, device=_DEVICE, dtype=torch.int32)
        fout = torch.ops.torchao._triton_permute_bwd(fx, fmap, 256, 128)
        assert fout.shape == (256, 128) and fout.dtype == torch.bfloat16


@pytest.mark.skipif(
    not (is_sm_at_least_100() and is_cuda_version_at_least(12, 8)),
    reason="MXFP8 test requires CUDA 12.8+ with SM >= 100",
)
def test_permute_mxfp8_eager_bwd_fallback_parity():
    """permute_mxfp8_fwd_hp_bwd(use_triton_for_bwd=False) matches the Triton
    backward bit-for-bit, and its gradient has exactly the input's row count
    (the eager slice must drop the sentinel row -- regression test for the
    [:-1] fix)."""
    block_size = 32
    tokens, dim = 64, 128
    num_experts, ep_degree = 8, 1
    counts = torch.full(
        (num_experts,), tokens // num_experts, dtype=torch.int32, device=_DEVICE
    )
    x = torch.randn(tokens, dim, device=_DEVICE, dtype=torch.bfloat16)

    grads = {}
    grad_seed = None
    for use_triton_for_bwd in (True, False):
        mx_input = MXTensor.to_mx(
            x, elem_dtype=torch.float8_e4m3fn, block_size=block_size
        )
        mx_input.requires_grad_(True)
        _, mx_output, _, _, _ = permute_mxfp8_fwd_hp_bwd(
            mx_input,
            counts,
            ep_degree,
            num_experts,
            block_size,
            use_triton_for_bwd=use_triton_for_bwd,
        )
        if grad_seed is None:
            grad_seed = torch.randn(
                mx_output.shape, device=_DEVICE, dtype=torch.bfloat16
            )
        mx_output.backward(grad_seed)
        assert mx_input.grad.shape == (tokens, dim)
        grads[use_triton_for_bwd] = mx_input.grad
    torch.testing.assert_close(grads[False], grads[True], rtol=0, atol=0)
