# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""PyTorch-only adapters with the public grouped kernel signatures.

These deliberately do not call any CuTeDSL/Triton operator. Keeping the layout
assembly separate from the quantization math makes per-expert addressing testable.
"""

import torch

from .nvfp4_reference import (
    reference_group_col_cast_requant_amax,
    reference_group_col_cast_requantize,
    reference_group_col_rht_requant_amax,
    reference_group_col_rht_requantize,
    reference_group_row_cast_quantize,
    reference_group_row_rht_col_rht_amax,
    reference_group_row_rht_col_rht_quantize_ms_eden,
    reference_software_e4m3_sr,
    to_blocked,
    to_blocked_grouped,
)


def reference_weight_amax(weight, num_tensors):
    assert weight.shape[0] == num_tensors
    return weight.float().abs().amax(dim=(-2, -1))


def _weight_outputs(refs, E, M, N):
    codes = torch.stack([r.codes for r in refs])
    scales = torch.stack([r.scales for r in refs]).view(E, M // 128, N // 64, 32, 16)
    return codes, scales


def reference_row_cast_op(weight, amax, num_tensors):
    return _weight_outputs(
        reference_group_row_cast_quantize(weight, amax), *weight.shape
    )


def reference_col_cast_amax_op(codes, scales, amax, num_tensors):
    return reference_group_col_cast_requant_amax(codes, scales, amax)


def reference_col_cast_op(codes, scales, amax, amax_t, num_tensors):
    E, M, half_N = codes.shape
    return _weight_outputs(
        reference_group_col_cast_requantize(codes, scales, amax, amax_t),
        E,
        half_N * 2,
        M,
    )


def reference_col_rht_amax_op(codes, scales, amax, signs, num_tensors):
    return reference_group_col_rht_requant_amax(codes, scales, amax, signs)


def reference_col_rht_op(codes, scales, amax, amax_t, signs, num_tensors):
    E, M, half_N = codes.shape
    return _weight_outputs(
        reference_group_col_rht_requantize(codes, scales, amax, amax_t, signs),
        E,
        half_N * 2,
        M,
    )


def reference_dual_amax_op(
    dy, d, w, offs, E, psl, hidden, shape_rep, logical_packed_length=None
):
    return reference_group_row_rht_col_rht_amax(dy, d, w, offs, E)


def reference_ms_eden_op(
    dy, ar, ac, d, w, offs, E, psl, hidden, shape_rep, rng, logical_packed_length=None
):
    row_codes, row_scale, col_codes, col_scale = (
        reference_group_row_rht_col_rht_quantize_ms_eden(dy, ar, ac, d, w, offs, E)
    )
    row_sf = to_blocked(reference_software_e4m3_sr(row_scale, rng[2], rng[3])).view(
        psl, hidden // 16
    )
    sizes = torch.diff(offs, prepend=offs.new_zeros(1)).tolist()
    col_plain = reference_software_e4m3_sr(col_scale, rng[0], rng[1])
    col_sf = torch.zeros_like(col_scale, dtype=torch.float8_e4m3fn)
    valid = sum(sizes) * hidden // 16
    col_sf.flatten()[:valid] = to_blocked_grouped(
        col_plain[:, : sum(sizes) // 16], sizes
    ).flatten()
    return row_codes, row_sf, col_codes, col_sf
