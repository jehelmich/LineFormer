# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""mmcv.ops.sync_bn: stand-ins only (they raise when called or instantiated), see _unavailable.py."""
from ._unavailable import SyncBatchNorm  # noqa: F401

__all__ = ['SyncBatchNorm']
