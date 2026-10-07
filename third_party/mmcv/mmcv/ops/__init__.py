# Copyright (c) OpenMMLab. All rights reserved.
# Modified by the LineFormer fork (2026): reduced to the names LineFormer and the vendored mmdetection 2.28.2
# import. Real code: MultiScaleDeformableAttention (pure-PyTorch path) and point_sample. Everything else here is a
# stand-in from _unavailable.py that raises OpUnavailableError when called or instantiated. Any other name of the
# full mmcv.ops is absent (AttributeError / ImportError).
from ._unavailable import OpUnavailableError
from .carafe import CARAFEPack, carafe
from .cc_attention import CrissCrossAttention
from .corner_pool import CornerPool
from .deform_conv import DeformConv2d, DeformConv2dPack, deform_conv2d
from .focal_loss import (SigmoidFocalLoss, SoftmaxFocalLoss,
                         sigmoid_focal_loss, softmax_focal_loss)
from .info import (get_compiler_version, get_compiling_cuda_version,
                   get_onnxruntime_op_path)
from .masked_conv import MaskedConv2d, masked_conv2d
from .modulated_deform_conv import (ModulatedDeformConv2d,
                                    ModulatedDeformConv2dPack,
                                    modulated_deform_conv2d)
from .multi_scale_deform_attn import MultiScaleDeformableAttention
from .nms import batched_nms, nms, nms_match, soft_nms
from .point_sample import (SimpleRoIAlign, point_sample,
                           rel_roi_point_to_rel_img_point)
from .roi_align import RoIAlign, roi_align
from .roi_pool import RoIPool, roi_pool
from .saconv import SAConv2d
from .sync_bn import SyncBatchNorm

__all__ = [
    'CARAFEPack', 'carafe', 'CornerPool', 'DeformConv2d', 'DeformConv2dPack',
    'deform_conv2d', 'SigmoidFocalLoss', 'SoftmaxFocalLoss',
    'sigmoid_focal_loss', 'softmax_focal_loss', 'get_compiler_version',
    'get_compiling_cuda_version', 'get_onnxruntime_op_path', 'MaskedConv2d',
    'masked_conv2d', 'ModulatedDeformConv2d', 'ModulatedDeformConv2dPack',
    'modulated_deform_conv2d', 'batched_nms', 'nms', 'soft_nms', 'nms_match',
    'RoIAlign', 'roi_align', 'RoIPool', 'roi_pool', 'SyncBatchNorm',
    'CrissCrossAttention', 'point_sample', 'rel_roi_point_to_rel_img_point',
    'SimpleRoIAlign', 'SAConv2d', 'MultiScaleDeformableAttention',
    'OpUnavailableError'
]
