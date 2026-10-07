# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""mmcv.ops.deform_conv: stand-ins only (they raise when called or instantiated), see _unavailable.py."""
from ._unavailable import DeformConv2d, DeformConv2dPack, deform_conv2d  # noqa: F401

__all__ = ['DeformConv2d', 'DeformConv2dPack', 'deform_conv2d']
