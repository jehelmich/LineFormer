# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""The real inference path on CPU without a checkpoint: the model is built from lineformer_swin_t_config.py with
random weights (seeded), then one forward and infer.get_dataseries on the demo image, with the kept-queries mode
off (upstream path) and on. This exercises torch, the vendored mmcv subset and mmdet end to end; the numbers mean
nothing (random weights), only shapes, types and the agreement of the two modes are checked."""
import sys
from pathlib import Path

import importlib.util
import unittest

_MISSING = [m for m in ('torch', 'mmcv', 'mmdet', 'cv2') if importlib.util.find_spec(m) is None]
if _MISSING:
    raise unittest.SkipTest('needs torch, mmcv (third_party/mmcv), mmdet and OpenCV (rocm/install_rocm.sh); '
                            'missing: %s' % ', '.join(_MISSING))
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.cuda.is_available = lambda: False  # CPU only; under WSL+ROCm the probe would start the HSA runtime

import cv2  # noqa: E402
import numpy as np  # noqa: E402

CONFIG = str(ROOT / 'lineformer_swin_t_config.py')
DEMO = str(ROOT / 'demo' / 'PMC5959982___3_HTML.jpg')


def _run(kept_only):
    import infer
    from mmdet.apis import inference_detector
    torch.manual_seed(0)
    infer.load_model(CONFIG, None, 'cpu', kept_only=kept_only)  # checkpoint None: random weights
    img = cv2.imread(DEMO)
    with torch.no_grad():
        result = inference_detector(infer.model, img)
        lines, masks = infer.get_dataseries(img, to_clean=False, return_masks=True)
    return img, result, lines, masks


def test_forward_and_dataseries_on_cpu_random_weights():
    img, result, lines, masks = _run(kept_only=False)
    bboxes, segms = result[0][0], result[1][0]
    assert bboxes.shape == (100, 5) and len(segms) == 100  # all queries, as upstream
    assert all(m.shape == img.shape[:2] and m.dtype == bool for m in segms)
    n_line = int((bboxes[:, 4] > 0.3).sum())
    assert len(lines) == len(masks) == n_line
    for line in lines:
        assert isinstance(line, list)
        for pt in line:
            assert set(pt) == {'x', 'y'} and isinstance(pt['x'], int) and isinstance(pt['y'], int)
    for m in masks:
        assert m.shape == img.shape[:2] and m.dtype == np.uint8

    # kept-queries mode: the same instances above 0.3, the same lines
    _, result_k, lines_k, _ = _run(kept_only=True)
    bk = result_k[0][0]
    assert int((bk[:, 4] > 0.3).sum()) == n_line
    key = lambda ln: sorted((p['x'], p['y']) for p in ln)  # noqa: E731  (order of instances can differ)
    assert sorted(map(key, lines_k)) == sorted(map(key, lines))
