# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""mmcv.ops.focal_loss: stand-ins only (they raise when called or instantiated), see _unavailable.py."""
from ._unavailable import SigmoidFocalLoss, SoftmaxFocalLoss, sigmoid_focal_loss, softmax_focal_loss  # noqa: F401

__all__ = ['SigmoidFocalLoss', 'SoftmaxFocalLoss', 'sigmoid_focal_loss', 'softmax_focal_loss']
