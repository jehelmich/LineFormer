# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""The vendored pure-Python mmcv subset (third_party/mmcv): no compiled extension anywhere in LineFormer's import
closure, the compiled ops are present as names but raise when used, and msda_compat falls back to the
pure-PyTorch MultiScaleDeformableAttention. CPU only, no checkpoint."""
import os
import subprocess
import sys
from pathlib import Path

import importlib.util
import unittest

_MISSING = [m for m in ('torch', 'mmcv', 'mmdet') if importlib.util.find_spec(m) is None]
if _MISSING:
    raise unittest.SkipTest('needs torch, mmcv (third_party/mmcv) and mmdet (rocm/install_rocm.sh); missing: %s'
                            % ', '.join(_MISSING))
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.cuda.is_available = lambda: False  # CPU only; under WSL+ROCm the probe would start the HSA runtime

import mmcv  # noqa: E402
import mmcv.ops  # noqa: E402

if not hasattr(mmcv.ops, 'OpUnavailableError'):
    raise unittest.SkipTest('the installed mmcv (%s) is not the vendored subset third_party/mmcv' % mmcv.__file__)

from mmcv.ops import OpUnavailableError  # noqa: E402
import mmcv.ops.multi_scale_deform_attn as msda_mod  # noqa: E402
import msda_compat  # noqa: E402

CONFIG = str(ROOT / 'lineformer_swin_t_config.py')

_CLOSURE = r'''
import importlib.util, json, sys
import torch
torch.cuda.is_available = lambda: False
import infer, kept_queries, msda_compat, scale_compat, tiling, lineformer_engine
import mmcv
from mmdet.apis import init_detector, inference_detector
model = init_detector(sys.argv[1], checkpoint=None, device='cpu')  # random weights
mods = {m: getattr(sys.modules[m], '__file__', None) for m in list(sys.modules) if m == 'mmcv' or m.startswith('mmcv.')}
print(json.dumps({'ext_spec': importlib.util.find_spec('mmcv._ext') is not None,
                  'ext_loaded': 'mmcv._ext' in sys.modules, 'modules': mods,
                  'compiled_files': [f for f in mods.values() if f and not f.endswith('.py')],
                  'n_params': sum(p.numel() for p in model.parameters())}))
'''


def test_import_closure_has_no_compiled_extension():
    """infer, the engine and mmdet import, and the LineFormer model builds, without any compiled mmcv code."""
    import json
    r = subprocess.run([sys.executable, '-c', _CLOSURE, CONFIG], cwd=str(ROOT), capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(ROOT), HIP_VISIBLE_DEVICES='-1',
                                CUDA_VISIBLE_DEVICES=''), timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert not out['ext_spec'] and not out['ext_loaded'], out
    assert out['compiled_files'] == [], out['compiled_files']
    pkg = os.path.dirname(mmcv.__file__)
    outside = [m for m, f in out['modules'].items() if f and not f.startswith(pkg)]
    assert outside == [], outside
    assert 'mmcv.ops.multi_scale_deform_attn' in out['modules']
    assert out['n_params'] > 1_000_000


def _stand_ins():
    out = []
    for name in mmcv.ops.__all__:
        obj = getattr(mmcv.ops, name)
        if getattr(obj, 'lineformer_unavailable', False):
            out.append((name, obj))
    return out


def test_every_stand_in_raises_when_used():
    found = _stand_ins()
    assert len(found) >= 25, [n for n, _ in found]
    for name, obj in found:
        try:
            obj(1, 2, 3)
        except OpUnavailableError as e:
            assert 'compiled extension' in str(e) and name in str(e), (name, str(e))
            assert isinstance(e, NotImplementedError)
        else:
            raise AssertionError('%s did not raise' % name)


def test_names_mmdetection_imports_exist():
    from mmcv.ops import (CornerPool, DeformConv2d, MaskedConv2d, RoIPool, batched_nms,  # noqa: F401
                          deform_conv2d, get_onnxruntime_op_path, nms, nms_match, point_sample,
                          rel_roi_point_to_rel_img_point, sigmoid_focal_loss)
    from mmcv.ops.carafe import CARAFEPack  # noqa: F401
    from mmcv.ops.merge_cells import ConcatCell, GlobalPoolingCell, SumCell  # noqa: F401
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d  # noqa: F401
    from mmcv.ops.nms import batched_nms as b2  # noqa: F401
    from mmcv.ops.roi_align import roi_align  # noqa: F401
    assert get_onnxruntime_op_path() == ''
    try:
        from mmcv.ops import voxelization  # noqa: F401  (a name of the full mmcv that is not vendored)
    except ImportError:
        pass
    else:
        raise AssertionError('mmcv.ops.voxelization should be absent')


def test_registered_names_fail_on_build():
    from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer, build_upsample_layer
    cases = [(build_conv_layer, dict(type='DCN'), (3, 3, 3)), (build_conv_layer, dict(type='DCNv2'), (3, 3, 3)),
             (build_conv_layer, dict(type='SAC'), (3, 3, 3)), (build_upsample_layer, dict(type='carafe'), ()),
             (build_norm_layer, dict(type='MMSyncBN'), (8,)),
             (build_plugin_layer, dict(type='CrissCrossAttention'), ())]
    for build, cfg, args in cases:
        try:
            build(cfg, *args)
        except OpUnavailableError:
            pass
        else:
            raise AssertionError('%s built' % cfg)


def _msda_inputs():
    g = torch.Generator().manual_seed(0)
    value = torch.randn(1, 4, 2, 4, generator=g)  # bs, keys (one 2x2 level), heads, dims per head
    shapes = torch.tensor([[2, 2]], dtype=torch.long)
    start = torch.tensor([0], dtype=torch.long)
    loc = torch.rand(1, 3, 2, 1, 2, 2, generator=g)  # bs, queries, heads, levels, points, xy
    attn = torch.rand(1, 3, 2, 1, 2, generator=g).softmax(-1)
    return value, shapes, start, loc, attn


def test_msda_compiled_function_raises_pytorch_function_runs():
    value, shapes, start, loc, attn = _msda_inputs()
    try:
        msda_mod.MultiScaleDeformableAttnFunction.apply(value, shapes, start, loc, attn, 64)
    except OpUnavailableError:
        pass
    else:
        raise AssertionError('the compiled MSDA function ran without a compiled extension')
    out = msda_mod.multi_scale_deformable_attn_pytorch(value, shapes, loc, attn)
    assert out.shape == (1, 3, 8) and torch.isfinite(out).all()
    m = msda_mod.MultiScaleDeformableAttention(embed_dims=8, num_heads=2, num_levels=1, num_points=2,
                                               batch_first=True)
    m.init_weights()
    q = torch.randn(1, 3, 8)
    ref = torch.rand(1, 3, 1, 2)
    y = m(q, value=torch.randn(1, 4, 8), reference_points=ref, spatial_shapes=shapes, level_start_index=start)
    assert y.shape == (1, 3, 8)


def test_msda_compat_without_compiled_kernel():
    ok, why = msda_compat.compiled_msda_usable('cuda:0')
    assert not ok and 'not available' in why, why
    orig_avail, orig_flag = torch.cuda.is_available, msda_mod.IS_CUDA_AVAILABLE
    torch.cuda.is_available = lambda: True  # pretend a GPU: the decision must not depend on running the kernel
    try:
        try:
            msda_compat.configure_msda('cuda:0', 'compiled')
        except RuntimeError as e:
            assert 'compiled' in str(e)
        else:
            raise AssertionError('mode compiled accepted without a compiled kernel')
        assert msda_compat.configure_msda('cuda:0', 'auto') == 'pytorch'
        assert msda_mod.IS_CUDA_AVAILABLE is False
        assert msda_compat.configure_msda('cuda:0', 'pytorch') == 'pytorch'
    finally:
        torch.cuda.is_available = orig_avail
        msda_compat.configure_msda('cpu')
        msda_mod.IS_CUDA_AVAILABLE = orig_flag
    assert msda_compat.get_msda_path() == 'cpu'
