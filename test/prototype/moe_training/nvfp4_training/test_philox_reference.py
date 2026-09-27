# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .nvfp4_reference import philox4x32


def test_philox4x32_zero_known_answer():
    # Random123 Philox4x32-10 zero counter/key known-answer vector.
    expected = torch.tensor([[0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]])
    actual = philox4x32(torch.tensor(0), torch.tensor(0), torch.tensor([0]))
    assert torch.equal(actual, expected)


def test_philox_offset_high_word_is_ignored():
    indexes = torch.arange(19)
    seed = torch.tensor(-7)
    a = philox4x32(seed, torch.tensor(5), indexes)
    b = philox4x32(seed, torch.tensor(2**32 + 5), indexes)
    assert torch.equal(a, b)
    assert not torch.equal(a, philox4x32(seed, torch.tensor(6), indexes))
