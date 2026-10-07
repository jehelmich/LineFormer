# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""mmcv.ops.merge_cells: stand-ins only (they raise when called or instantiated), see _unavailable.py."""
from ._unavailable import ConcatCell, GlobalPoolingCell, SumCell  # noqa: F401

__all__ = ['ConcatCell', 'GlobalPoolingCell', 'SumCell']
