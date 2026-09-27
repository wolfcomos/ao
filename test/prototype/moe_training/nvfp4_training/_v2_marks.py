# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Hardware and runtime skip marks for the CuTeDSL V2 kernel tests."""

import pytest

from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.utils import torch_version_at_least

CUTEDSL_AVAILABLE = cutedsl_nvfp4_kernels_available()
requires_cutedsl = pytest.mark.skipif(
    not CUTEDSL_AVAILABLE or not torch_version_at_least("2.10.0"),
    reason="requires SM100+, PyTorch 2.10+, and CuTeDSL",
)


def maybe_sm100(fn):
    """Gate CUDA operator and wrapper tests on the supported runtime."""
    return requires_cutedsl(fn)
