# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""Explicit stand-ins for the mmcv 1.7.2 ops that need mmcv's compiled extension (``mmcv._ext``).

This subset of mmcv ships no compiled code. The vendored mmdetection imports a number of compiled ops at import
time (nms, RoIAlign, DeformConv2d, ...), so they must exist as names, but LineFormer inference never calls them
(measured: 0 calls into ``mmcv._ext`` on CPU and GPU). Rules:

* importing a stand-in always works;
* calling a function stand-in, instantiating a class stand-in, or calling anything on ``missing_ext(...)`` raises
  ``OpUnavailableError`` (a ``NotImplementedError``) naming the op;
* the registry names the full mmcv gives these layers ('DCN', 'DCNv2', 'SAC', 'carafe', 'CrissCrossAttention',
  'MMSyncBN') are registered with class stand-ins, so a config that asks for one fails on build with this error
  instead of an unknown-name KeyError.

The pure-PyTorch ops LineFormer needs are real code: ``multi_scale_deform_attn`` (with its pure-PyTorch function
``multi_scale_deformable_attn_pytorch``) and ``point_sample``.
"""
import torch.nn as nn

from mmcv.cnn.bricks.registry import (CONV_LAYERS, NORM_LAYERS, PLUGIN_LAYERS,
                                      UPSAMPLE_LAYERS)


class OpUnavailableError(NotImplementedError):
    """Raised when code calls an mmcv op that needs the compiled extension, which this subset does not have."""


def _message(qualname):
    return ('mmcv op %s needs mmcv\'s compiled extension (mmcv._ext), which the pure-Python mmcv 1.7.2 subset '
            'vendored by LineFormer (third_party/mmcv) does not include. LineFormer inference does not use it; '
            'install mmcv-full 1.7.x built for your torch if you need it.' % qualname)


def unavailable_function(name, module):
    qualname = '%s.%s' % (module, name)

    def stub(*args, **kwargs):
        raise OpUnavailableError(_message(qualname))

    stub.__name__ = stub.__qualname__ = name
    stub.__module__ = module
    stub.__doc__ = 'Not available in this mmcv subset: calling it raises OpUnavailableError.'
    stub.lineformer_unavailable = True
    return stub


def unavailable_class(name, module, base=nn.Module):
    qualname = '%s.%s' % (module, name)

    def __init__(self, *args, **kwargs):
        raise OpUnavailableError(_message(qualname))

    cls = type(name, (base,), {
        '__init__': __init__,
        '__module__': module,
        '__doc__': 'Not available in this mmcv subset: instantiating it raises OpUnavailableError.',
        'lineformer_unavailable': True,
    })
    return cls


class _MissingExt:
    """Stands in for ``ext_loader.load_ext('_ext', funcs)``: every listed function raises when called."""

    def __init__(self, name, funcs):
        self._name = name
        for fun in funcs:
            setattr(self, fun, unavailable_function(fun, 'mmcv.' + name))

    def __getattr__(self, fun):  # a function not listed: same failure, never an AttributeError at import
        return unavailable_function(fun, 'mmcv.' + self._name)


def missing_ext(name, funcs):
    return _MissingExt(name, funcs)


# ---------------------------------------------------------------- the names mmdetection 2.28.2 and mmcv import

_M = 'mmcv.ops'
# functions
batched_nms = unavailable_function('batched_nms', _M + '.nms')
nms = unavailable_function('nms', _M + '.nms')
nms_match = unavailable_function('nms_match', _M + '.nms')
soft_nms = unavailable_function('soft_nms', _M + '.nms')
roi_align = unavailable_function('roi_align', _M + '.roi_align')
roi_pool = unavailable_function('roi_pool', _M + '.roi_pool')
deform_conv2d = unavailable_function('deform_conv2d', _M + '.deform_conv')
modulated_deform_conv2d = unavailable_function('modulated_deform_conv2d', _M + '.modulated_deform_conv')
masked_conv2d = unavailable_function('masked_conv2d', _M + '.masked_conv')
sigmoid_focal_loss = unavailable_function('sigmoid_focal_loss', _M + '.focal_loss')
softmax_focal_loss = unavailable_function('softmax_focal_loss', _M + '.focal_loss')
carafe = unavailable_function('carafe', _M + '.carafe')
get_compiler_version = unavailable_function('get_compiler_version', _M + '.info')
get_compiling_cuda_version = unavailable_function('get_compiling_cuda_version', _M + '.info')

# classes
RoIAlign = unavailable_class('RoIAlign', _M + '.roi_align')
RoIPool = unavailable_class('RoIPool', _M + '.roi_pool')
DeformConv2d = unavailable_class('DeformConv2d', _M + '.deform_conv')
DeformConv2dPack = unavailable_class('DeformConv2dPack', _M + '.deform_conv')
ModulatedDeformConv2d = unavailable_class('ModulatedDeformConv2d', _M + '.modulated_deform_conv')
ModulatedDeformConv2dPack = unavailable_class('ModulatedDeformConv2dPack', _M + '.modulated_deform_conv')
MaskedConv2d = unavailable_class('MaskedConv2d', _M + '.masked_conv')
CornerPool = unavailable_class('CornerPool', _M + '.corner_pool')
SigmoidFocalLoss = unavailable_class('SigmoidFocalLoss', _M + '.focal_loss')
SoftmaxFocalLoss = unavailable_class('SoftmaxFocalLoss', _M + '.focal_loss')
CARAFEPack = unavailable_class('CARAFEPack', _M + '.carafe')
SAConv2d = unavailable_class('SAConv2d', _M + '.saconv')
CrissCrossAttention = unavailable_class('CrissCrossAttention', _M + '.cc_attention')
SyncBatchNorm = unavailable_class('SyncBatchNorm', _M + '.sync_bn')
# pure PyTorch in the full mmcv, but not used by LineFormer: not vendored
ConcatCell = unavailable_class('ConcatCell', _M + '.merge_cells')
GlobalPoolingCell = unavailable_class('GlobalPoolingCell', _M + '.merge_cells')
SumCell = unavailable_class('SumCell', _M + '.merge_cells')

# registry names of the full mmcv
CONV_LAYERS.register_module('DCN', module=DeformConv2dPack)
CONV_LAYERS.register_module('DCNv2', module=ModulatedDeformConv2dPack)
CONV_LAYERS.register_module('SAC', module=SAConv2d)
UPSAMPLE_LAYERS.register_module('carafe', module=CARAFEPack)
PLUGIN_LAYERS.register_module('CrissCrossAttention', module=CrissCrossAttention)
NORM_LAYERS.register_module('MMSyncBN', module=SyncBatchNorm)
