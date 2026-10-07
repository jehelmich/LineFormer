# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""Unit tests for scale_compat.py: the overridden test pipelines and what they feed the network (no checkpoint).

pytest tests -q      or, without pytest:  python tests/test_scale_compat.py
"""
import sys
from pathlib import Path

import numpy as np
import importlib.util
import unittest

# These tests need the model stack; without it the module is skipped with the reason (pytest -rs prints it).
# find_spec only: importing mmcv here would probe the GPU before the CPU-only patch below.
_MISSING = [m for m in ('torch', 'mmcv', 'mmdet') if importlib.util.find_spec(m) is None]
if _MISSING:
    raise unittest.SkipTest('needs torch, mmcv-full and mmdet (rocm/install_rocm.sh); missing: %s'
                            % ', '.join(_MISSING))
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.cuda.is_available = lambda: False  # CPU only; under WSL+ROCm the probe would start the HSA runtime

import mmcv  # noqa: E402

import scale_compat as sc  # noqa: E402

CONFIG = str(ROOT / 'lineformer_swin_t_config.py')


class _M:
    """Stand-in for a model: input_meta only reads .cfg."""
    def __init__(self, size, pad=None):
        self.cfg = sc.build_config(CONFIG, size, pad)


def _img(h, w):
    return np.full((h, w, 3), 255, np.uint8)


def test_config_is_unchanged():
    cfg = mmcv.Config.fromfile(CONFIG)
    assert sc.test_pipeline_for(cfg, 'config') == cfg.data.test.pipeline


def test_config_size_matches_512():
    for h, w in ((1436, 1436), (1436, 2872), (300, 700)):
        a = sc.input_meta(_M('config'), _img(h, w))
        b = sc.input_meta(_M(512), _img(h, w))
        assert a == b
        assert max(a['img_shape'][:2]) == 512 or min(a['img_shape'][:2]) == 512


def test_fit_1024_keeps_ratio_no_pad():
    m = sc.input_meta(_M(1024), _img(1436, 2872))
    assert m['img_shape'][:2] == (512, 1024)
    assert m['pad_shape'][:2] == (512, 1024)
    assert m['input_shape'] == (3, 512, 1024)


def test_native_no_resize_pad_32():
    for h, w in ((1436, 1436), (1436, 2872), (512, 512), (1, 33)):
        m = sc.input_meta(_M('native'), _img(h, w))
        assert m['ori_shape'][:2] == (h, w)
        assert m['img_shape'][:2] == (h, w)
        ph, pw = m['pad_shape'][:2]
        assert ph % 32 == 0 and pw % 32 == 0 and 0 <= ph - h < 32 and 0 <= pw - w < 32
        assert m['input_shape'] == (3, ph, pw)
        assert np.allclose(m['scale_factor'], 1.0)


def test_native_pad_is_white():
    cfg = sc.build_config(CONFIG, 'native')
    norm = [t for t in cfg.data.test.pipeline[1]['transforms'] if t['type'] == 'Normalize'][0]
    from mmdet.apis.inference import replace_ImageToTensor
    from mmdet.datasets.pipelines import Compose
    cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    out = Compose(replace_ImageToTensor(cfg.data.test.pipeline))(dict(img=_img(40, 40)))
    x = out['img'][0].data.numpy()
    white = (255 - np.array(norm['mean'])) / np.array(norm['std'])
    assert x.shape == (3, 64, 64)
    assert np.allclose(x[:, 50, 50], white[::-1] if not norm.get('to_rgb') else white, atol=1e-4)
    assert np.allclose(x[:, 10, 10], x[:, 50, 50])


def test_bad_size_raises():
    for bad in (0, -3, 'fit', (512,), (0, 10), True):
        try:
            sc.build_config(CONFIG, bad)
        except ValueError:
            continue
        raise AssertionError('accepted %r' % (bad,))


if __name__ == '__main__':
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            n += 1
            print('ok', name)
    print('%d tests passed' % n)
