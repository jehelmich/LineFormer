# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""Unit tests for tiling.py on synthetic masks (no model, no torch).

pytest tests -q      or, without pytest:  python tests/test_tiling.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tiling  # noqa: E402


def _raises(fn, *a):
    try:
        fn(*a)
    except ValueError:
        return True
    return False


def test_crop_starts_cover_and_overlap():
    for length in (1, 100, 511, 512, 513, 896, 1436, 2872, 5000):
        for size, overlap in ((512, 128), (768, 192), (100, 0)):
            s = tiling.crop_starts(length, size, overlap)
            assert s[0] == 0
            assert s == sorted(set(s))
            end = min(length, s[-1] + size)
            assert end == length, (length, size, overlap, s)
            if length <= size:
                assert s == [0]
            else:
                assert s[-1] == length - size
                gaps = np.diff(s)
                assert (gaps <= size - overlap).all()          # neighbours share >= overlap px
                assert (gaps > 0).all()


def test_crop_starts_examples():
    assert tiling.crop_starts(1436, 512, 128) == [0, 384, 768, 924]
    assert tiling.crop_starts(512, 512, 128) == [0]
    assert tiling.crop_starts(300, 512, 128) == [0]
    assert tiling.crop_starts(896, 512, 128) == [0, 384]
    assert tiling.crop_starts(900, 512, 128) == [0, 384, 388]   # 2 crops cover at most 896 with overlap 128


def test_crop_starts_rejects_bad_args():
    assert _raises(tiling.crop_starts, 100, 512, 512)
    assert _raises(tiling.crop_starts, 100, 512, -1)
    assert _raises(tiling.crop_starts, 0, 512, 128)


def test_crop_boxes_cover_every_pixel():
    H, W = 1436, 2872
    cover = np.zeros((H, W), int)
    for y0, x0, y1, x1 in tiling.crop_boxes(H, W, 512, 128):
        assert y1 - y0 <= 512 and x1 - x0 <= 512
        cover[y0:y1, x0:x1] += 1
    assert cover.min() >= 1


def _fake_infer(full_masks, scores):
    """infer_fn that 'detects' the parts of ground-truth masks visible in a crop (by matching the crop to img)."""
    def fn(model, crop):
        y0, x0 = model['where'](crop)
        h, w = crop.shape[:2]
        ms = [m[y0:y0 + h, x0:x0 + w] for m in full_masks]
        keep = [k for k, m in enumerate(ms) if m.any()]
        bb = np.array([[0, 0, w, h, scores[k]] for k in keep], dtype=np.float32).reshape(-1, 5)
        return [bb], [[ms[k] for k in keep]]
    return fn


def _coord_image(H, W):
    """An image whose pixels encode their own coordinates, so a crop can be located."""
    img = np.zeros((H, W, 3), np.uint16)
    img[..., 0] = np.arange(H)[:, None]
    img[..., 1] = np.arange(W)[None, :]
    return img


def _model_for():
    return {'where': lambda crop: (int(crop[0, 0, 0]), int(crop[0, 0, 1]))}


def _line(H, W, y_of_x, width=5, x_range=None):
    m = np.zeros((H, W), bool)
    xa, xb = x_range or (0, W)
    for x in range(xa, xb):
        yc = int(round(y_of_x(x)))
        m[max(0, yc - width // 2):min(H, yc + width // 2 + 1), x] = True
    return m


def test_one_long_line_becomes_one_instance():
    H, W = 700, 2000
    gt = _line(H, W, lambda x: 300 + 100 * np.sin(x / 200))
    (bb,), (masks,) = tiling.run_tiled(_model_for(), _coord_image(H, W), 512, 128,
                                       infer_fn=_fake_infer([gt], [0.9]))
    assert len(masks) == 1
    assert (masks[0] == gt).all()
    ys, xs = np.nonzero(gt)
    assert list(bb[0, :4]) == [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    assert abs(bb[0, 4] - 0.9) < 1e-6


def test_parallel_lines_stay_apart():
    H, W = 600, 1500
    a = _line(H, W, lambda x: 200)
    b = _line(H, W, lambda x: 260)
    (bb,), (masks,) = tiling.run_tiled(_model_for(), _coord_image(H, W), 512, 128,
                                       infer_fn=_fake_infer([a, b], [0.8, 0.6]))
    assert len(masks) == 2
    assert (masks[0] == a).all() and (masks[1] == b).all()        # sorted by score
    assert np.allclose(bb[:, 4], [0.8, 0.6])


def test_crossing_lines_do_not_chain():
    H, W = 1000, 1000
    a = _line(H, W, lambda x: x * 0.9 + 20)
    b = _line(H, W, lambda x: 950 - x * 0.9)
    (bb,), (masks,) = tiling.run_tiled(_model_for(), _coord_image(H, W), 512, 128,
                                       infer_fn=_fake_infer([a, b], [0.9, 0.9]))
    assert len(masks) == 2
    got = sorted(int(m.sum()) for m in masks)
    assert got == sorted([int(a.sum()), int(b.sum())])
    assert all((m == a).all() or (m == b).all() for m in masks)


def test_low_scores_dropped_and_weighted_score():
    H, W = 400, 896   # two crops along x: [0, 512) and [384, 896)
    gt = _line(H, W, lambda x: 100)
    noise = _line(H, W, lambda x: 300, x_range=(10, 50))
    tiles = []
    for box in tiling.crop_boxes(H, W, 512, 128):
        y0, x0, y1, x1 = box
        s = 0.9 if x0 == 0 else 0.5
        tiles.append((box, np.array([s, 0.1]), np.stack([gt[y0:y1, x0:x1], noise[y0:y1, x0:x1]])))
    bb, masks, info = tiling.merge_instances(tiles, H, W, score_thr=0.3)
    assert len(masks) == 1 and (masks[0] == gt).all()
    a0 = gt[:, :512].sum()
    a1 = gt[:, 384:].sum()
    assert abs(bb[0, 4] - (0.9 * a0 + 0.5 * a1) / (a0 + a1)) < 1e-5
    assert len(info['links']) == 1 and info['members'][0] == [(0, 0), (1, 0)]


def test_split_instance_joins_only_its_best_partner():
    H, W = 300, 896
    gt = _line(H, W, lambda x: 100)
    # crop 0 sees the whole line as one instance; crop 1 sees it as two pieces that both reach the overlap
    boxes = tiling.crop_boxes(H, W, 512, 128)
    (b0, b1) = boxes
    p1 = gt.copy()
    p1[:, 600:] = False
    p2 = gt.copy()
    p2[:, :420] = False
    tiles = [(b0, np.array([0.9]), gt[None, b0[0]:b0[2], b0[1]:b0[3]]),
             (b1, np.array([0.9, 0.8]), np.stack([p1[b1[0]:b1[2], b1[1]:b1[3]], p2[b1[0]:b1[2], b1[1]:b1[3]]]))]
    bb, masks, info = tiling.merge_instances(tiles, H, W)
    assert len(masks) == 2          # documented limit: the second piece stays separate


def test_small_overlap_presence_not_linked():
    H, W = 300, 896
    a = _line(H, W, lambda x: 100, x_range=(0, 392))     # reaches only 8 columns into the overlap [384, 512)
    b = _line(H, W, lambda x: 100, x_range=(392, 896))
    boxes = tiling.crop_boxes(H, W, 512, 128)
    tiles = [(bx, np.array([0.9, 0.9]), np.stack([a[bx[0]:bx[2], bx[1]:bx[3]], b[bx[0]:bx[2], bx[1]:bx[3]]]))
             for bx in boxes]
    # crop 1 sees only an 8-column stub of a (40 px in the overlap): below min_px=50 it is not joined to a
    bb, masks, info = tiling.merge_instances(tiles, H, W, min_px=50)
    assert len(masks) == 3
    # with min_px=10 the stub joins a (b joins b either way; a and b never join: they touch, not overlap)
    bb, masks, info = tiling.merge_instances(tiles, H, W, min_px=10)
    assert len(masks) == 2
    assert sorted(int(m.sum()) for m in masks) == sorted([int(a.sum()), int(b.sum())])


def test_mask_shape_mismatch_raises():
    tiles = [((0, 0, 10, 10), np.array([0.9]), np.zeros((1, 9, 10), bool))]
    assert _raises(tiling.merge_instances, tiles, 10, 10)


def test_empty():
    H, W = 600, 600
    (bb,), (masks,) = tiling.run_tiled(_model_for(), _coord_image(H, W), 512, 128, infer_fn=_fake_infer([], []))
    assert bb.shape == (0, 5) and masks == []


if __name__ == '__main__':
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            n += 1
            print('ok', name)
    print('%d tests passed' % n)
