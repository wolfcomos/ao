# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Assertions for V2 recipe integration tests."""

import torch

from ._assertions import (
    assert_codes_bitwise,
    assert_scales_adjacent,
    assert_scales_bitwise,
)
from ._v2_reference_ops import reference_ms_eden_op


def assert_ms_eden_modes(inputs, software_outputs, hardware_outputs):
    """Compare a captured software call with PyTorch and both SR modes at fixed dy.

    These assertions apply to the raw MS-EDEN outputs. Activation quantization also
    changes under the recipe's unified flag, so they say nothing about equality of
    the full recipe's outputs or gradients.
    """
    expected = reference_ms_eden_op(*inputs)
    scale_bytes_changed = False
    for i, label in enumerate(("row codes", "row scales", "col codes", "col scales")):
        software, hardware, ref = software_outputs[i], hardware_outputs[i], expected[i]
        if i % 2 == 0:
            assert_codes_bitwise(software, ref, f"software {label}")
            assert_codes_bitwise(hardware, ref, f"hardware {label}")
        else:
            assert_scales_bitwise(software, ref, f"software {label}")
            assert_scales_adjacent(hardware, software, f"hardware vs software {label}")
            scale_bytes_changed |= not torch.equal(
                hardware.view(torch.uint8), software.view(torch.uint8)
            )
    assert scale_bytes_changed, "hardware SR must exercise a different Philox mapping"
