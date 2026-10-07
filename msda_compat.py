# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""Choose the MultiScaleDeformableAttention (MSDA) code path used on a GPU.

mmcv 1.7.x runs MSDA through its compiled extension (``MultiScaleDeformableAttnFunction``)
when the input is a CUDA/HIP tensor, and through the pure-PyTorch reference
``multi_scale_deformable_attn_pytorch`` (same algorithm, built on ``F.grid_sample``) otherwise.
An mmcv-full built with CPU ops only (e.g. on AMD ROCm, where the HIP kernels are not built)
has no GPU kernel, so inference on the GPU fails inside the pixel decoder.

Modes:
  ``compiled``  use mmcv's compiled GPU kernel; raise if it is not usable.
  ``pytorch``   send GPU inputs down mmcv's own pure-PyTorch branch.
  ``auto``      ``compiled`` if a tiny call through the compiled kernel works on the device,
                else ``pytorch``.

The ``pytorch`` route sets the module flag ``IS_CUDA_AVAILABLE`` in
``mmcv.ops.multi_scale_deform_attn`` to False; that flag is read only by
``MultiScaleDeformableAttention.forward`` to pick the compiled branch, so every input then takes
the branch that CPU inputs always take. It is process-wide. On ``cpu`` nothing is changed.
"""
import os

import torch

MODES = ('auto', 'compiled', 'pytorch')
ENV_VAR = 'LINEFORMER_MSDA'

_ORIG_FLAG = None
_active = None  # resolved mode last applied, for logging once


def _msda_module():
    import mmcv.ops.multi_scale_deform_attn as msda_mod
    return msda_mod


def _is_gpu(device):
    return torch.device(device).type == 'cuda'


def compiled_msda_usable(device):
    """Return (ok, reason): does mmcv's compiled MSDA kernel run on ``device``?"""
    msda_mod = _msda_module()
    try:
        from mmcv.ops import get_compiling_cuda_version
        built_with = get_compiling_cuda_version()
    except Exception as e:  # mmcv without the compiled extension at all
        return False, 'mmcv ops extension not available (%s)' % e
    if built_with == 'not available':
        return False, 'mmcv ops built without CUDA/HIP (get_compiling_cuda_version: not available)'
    # tiny call: 1 head, 1 level of 2x2, 1 query, 1 point
    try:
        dev = torch.device(device)
        value = torch.ones(1, 4, 1, 2, device=dev)
        shapes = torch.tensor([[2, 2]], dtype=torch.long, device=dev)
        start = torch.tensor([0], dtype=torch.long, device=dev)
        loc = torch.full((1, 1, 1, 1, 1, 2), 0.5, device=dev)
        attn = torch.ones(1, 1, 1, 1, 1, device=dev)
        out = msda_mod.MultiScaleDeformableAttnFunction.apply(value, shapes, start, loc, attn, 1)
        torch.cuda.synchronize(dev)
        if out.shape != (1, 1, 2):
            return False, 'compiled MSDA returned shape %s' % (tuple(out.shape),)
    except Exception as e:
        return False, 'compiled MSDA call failed on %s: %s' % (device, str(e).splitlines()[0])
    return True, 'compiled MSDA ran on %s (mmcv built with %s)' % (device, built_with)


def configure_msda(device, mode=None):
    """Select the MSDA path for ``device``; returns the resolved mode ('cpu', 'compiled' or 'pytorch').

    ``mode`` defaults to the environment variable LINEFORMER_MSDA, else 'auto'.
    Raises if a GPU is asked for and no usable path exists.
    """
    global _ORIG_FLAG, _active
    if mode is None:
        mode = os.environ.get(ENV_VAR, 'auto')
    if mode not in MODES:
        raise ValueError('MSDA mode must be one of %s, got %r' % (MODES, mode))
    msda_mod = _msda_module()
    if _ORIG_FLAG is None:
        _ORIG_FLAG = msda_mod.IS_CUDA_AVAILABLE

    if not _is_gpu(device):
        # CPU inference takes mmcv's pure-PyTorch branch already; leave mmcv untouched.
        msda_mod.IS_CUDA_AVAILABLE = _ORIG_FLAG
        resolved, reason = 'cpu', 'device %s: mmcv default (pure PyTorch on CPU tensors)' % device
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('device %r requested but torch.cuda.is_available() is False '
                               '(torch %s, hip %s)' % (device, torch.__version__, torch.version.hip))
        if mode == 'pytorch':
            resolved, reason = 'pytorch', 'requested'
        else:
            ok, why = compiled_msda_usable(device)
            if ok:
                resolved, reason = 'compiled', why
            elif mode == 'compiled':
                raise RuntimeError('MSDA mode "compiled" requested for %s, but: %s. '
                                   'Use mode "pytorch" or "auto".' % (device, why))
            else:
                resolved, reason = 'pytorch', why
        msda_mod.IS_CUDA_AVAILABLE = _ORIG_FLAG if resolved == 'compiled' else False

    if (resolved, reason) != _active:
        print('LineFormer MSDA path: %s (mode %s; %s)' % (resolved, mode, reason))
        _active = (resolved, reason)
    return resolved


def get_msda_path():
    """The MSDA path last configured ('cpu', 'compiled', 'pytorch'), or None."""
    return _active[0] if _active else None
