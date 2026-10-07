# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""mmcv.ops.masked_conv: stand-ins only (they raise when called or instantiated), see _unavailable.py."""
from ._unavailable import MaskedConv2d, masked_conv2d  # noqa: F401

__all__ = ['MaskedConv2d', 'masked_conv2d']
