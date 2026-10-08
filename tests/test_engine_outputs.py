# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""The engine's output writer (lineformer_engine.write_outputs) with the real infer.get_dataseries on synthetic
detector results: the lines and instances it writes are in the geometric order whatever order the model returned
the instances in, and line i is instance i. Needs the model stack for `import infer` (no checkpoint, no GPU)."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

_MISSING = [m for m in ('torch', 'mmcv', 'mmdet', 'cv2', 'skimage') if importlib.util.find_spec(m) is None]
if _MISSING:
    raise unittest.SkipTest('needs torch, mmcv (third_party/mmcv), mmdet, OpenCV and scikit-image '
                            '(rocm/install_rocm.sh); missing: %s' % ', '.join(_MISSING))
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.cuda.is_available = lambda: False  # CPU only; under WSL+ROCm the probe would start the HSA runtime

import numpy as np  # noqa: E402

import lineformer_engine as engine  # noqa: E402
import lineformer_jobs as jobs  # noqa: E402

H, W = 120, 200


def _instances():
    """Four instances: three line masks and a low-score blob -> (boxes (4, 5), masks)."""
    masks = np.zeros((4, H, W), bool)
    masks[0, 79:82, 20:181] = True   # starts at x 20, y 80
    masks[1, 4:9, 150:160] = True    # score 0.2: no line
    masks[2, 29:32, 20:181] = True   # starts at x 20, y 30
    masks[3, 54:57, 5:101] = True    # starts at x 5
    scores = [0.95, 0.2, 0.6, 0.31]
    boxes = []
    for m, s in zip(masks, scores):
        ys, xs = np.nonzero(m)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, s])
    return np.array(boxes, np.float32), masks


def _write(out, order, outputs=('instances', 'masks')):
    import infer
    boxes, masks = _instances()
    rec = {'out': out, 'id': 'img', 'path': '/img/img.png', 'image_sha256': 'x', 'shape': [H, W, 3],
           'instances': 'instances' in outputs, 'masks': 'masks' in outputs, 'fingerprint': {}, 'job': 'j',
           't_pre_start': 0.0, 't_pre_end': 0.0, 't_gpu_start': 0.0, 't_gpu_end': 0.0}
    result = ([boxes[order]], [[masks[i] for i in order]])
    jpath, obj = engine.write_outputs(infer, rec, result)
    jobs.write_json_atomic(jpath, obj)
    return engine.load_outputs(out, 'img')


def test_output_order_does_not_depend_on_model_order():
    ref = None
    with tempfile.TemporaryDirectory() as d:
        for k, order in enumerate(([0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1])):
            out = str(Path(d) / str(k))
            Path(out).mkdir()
            res = _write(out, order)
            assert res['json']['order'] == jobs.ORDER == 'geometric'
            assert res['json']['n_lines'] == 3 and res['json']['n_instances'] == 4
            # lines by leftmost x, then mean y; the low-score instance last
            assert [min(p['x'] for p in ln) for ln in res['lines']] == [5, 20, 20]
            assert np.allclose(res['scores'], [0.31, 0.6, 0.95, 0.2])
            # line i is instance i: its points lie on that instance's mask
            for ln, m in zip(res['lines'], res['masks']):
                assert all(m[int(p['y']), int(p['x'])] for p in ln)
            if ref is None:
                ref = res
                continue
            assert res['lines'] == ref['lines']
            assert np.array_equal(res['boxes'], ref['boxes']) and np.array_equal(res['labels'], ref['labels'])
            assert np.array_equal(res['masks'], ref['masks'])


def test_lines_only_output_is_ordered_too():
    with tempfile.TemporaryDirectory() as d:
        a = _write(d, [3, 2, 1, 0], outputs=())
        b = _write(d, [0, 1, 2, 3], outputs=())
        assert a['lines'] == b['lines'] and a['json']['outputs'] == []
        assert [min(p['x'] for p in ln) for ln in a['lines']] == [5, 20, 20]
