"""Run LineFormer on a large image as overlapping native-resolution crops and merge the instances.

Idea: instead of shrinking a large image to the network's training size, cut it into crops of ``size`` x ``size``
pixels at native resolution with ``overlap`` pixels shared between neighbours, run the detector on each crop
(use a model built with ``scale_compat.build_model(..., size='native')`` so the crop is not resized), put each
instance mask back at its place in the image, and join instances of neighbouring crops that are the same line.

Geometry (``crop_starts``): along each axis the starts are 0, step, 2*step, ... with ``step = size - overlap``,
and the last crop is moved to end exactly at the image edge (so it may overlap its neighbour by more than
``overlap``). An axis shorter than ``size`` gets one crop of the full axis length. Every pixel is covered; two
neighbouring crops share at least ``overlap`` pixels.

Merge rule (``merge_instances``):
  1. In each crop, instances with score < ``score_thr`` are dropped before merging (they are not returned).
  2. For every pair of crops whose rectangles intersect (side and diagonal neighbours), and every instance ``a`` of
     the first and ``b`` of the second, measure inside the shared rectangle R only:
         link(a, b) = |a & b & R| / min(|a & R|, |b & R|)
     (0 if either has fewer than ``min_px`` pixels in R). Both crops see R at the same native resolution, so the
     same line gives nearly the same pixels in both, while two different lines share only the pixels where they
     cross.
  3. ``a`` and ``b`` are joined if link >= ``link_thr`` and each is the other's best partner in this crop pair
     (mutual best): one instance is never joined to two instances of the same neighbouring crop, so crossing or
     touching lines do not chain together through one crop. Joins are closed transitively (union-find), so a line
     that runs through many crops becomes one instance.
  4. A merged instance's mask is the union (OR) of its members' masks; its score is the mean of the member scores
     weighted by member mask area; its box is the tight box of the merged mask.
  Known limits: a line that one crop splits into two instances joins only one of them to its neighbour (the other
  stays a separate instance); a line visible in R only as a few pixels (< ``min_px``) is not joined.

``run_tiled`` returns the same structure as ``mmdet.apis.inference_detector`` for a one-class model:
``([bboxes (N, 5) float32: x1, y1, x2, y2, score], [[mask (H, W) bool, ...]])``, sorted by score, descending.
"""
import numpy as np


def crop_starts(length, size, overlap):
    """Start offsets of the crops along one axis (see module docstring)."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError('need size > 0 and 0 <= overlap < size, got size=%r overlap=%r' % (size, overlap))
    if length <= 0:
        raise ValueError('axis length must be positive, got %r' % (length,))
    if length <= size:
        return [0]
    step = size - overlap
    starts = list(range(0, length - size, step))
    starts.append(length - size)
    return starts


def crop_boxes(height, width, size, overlap):
    """Crop rectangles (y0, x0, y1, x1), row-major."""
    ys, xs = crop_starts(height, size, overlap), crop_starts(width, size, overlap)
    return [(y, x, min(y + size, height), min(x + size, width)) for y in ys for x in xs]


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i, j):
        a, b = self.find(i), self.find(j)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def _intersect(b1, b2):
    y0, x0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    y1, x1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    return (y0, x0, y1, x1) if y1 > y0 and x1 > x0 else None


def merge_instances(tiles, height, width, score_thr=0.3, link_thr=0.5, min_px=20):
    """Merge per-crop instances into image instances.

    tiles: list of (box (y0, x0, y1, x1), scores (n,), masks (n, h, w) bool in crop coordinates).
    Returns (bboxes (N, 5) float32, list of N (height, width) bool masks, info dict with 'members' (list of lists of
    (tile index, instance index)) and 'links' (accepted joins with their link value)).
    """
    insts = []  # (tile, k, score, mask)
    for t, (box, scores, masks) in enumerate(tiles):
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if len(masks) != len(scores):
            raise ValueError('tile %d: %d masks but %d scores' % (t, len(masks), len(scores)))
        h, w = box[2] - box[0], box[3] - box[1]
        for k, (s, m) in enumerate(zip(scores, masks)):
            m = np.asarray(m, dtype=bool)
            if m.shape != (h, w):
                raise ValueError('tile %d instance %d: mask shape %s, crop is %s' % (t, k, m.shape, (h, w)))
            if s >= score_thr:
                insts.append((t, k, float(s), m))
    by_tile = {}
    for i, (t, _, _, _) in enumerate(insts):
        by_tile.setdefault(t, []).append(i)
    uf = _UnionFind(len(insts))
    links = []
    tids = sorted(by_tile)
    for ai, ta in enumerate(tids):
        for tb in tids[ai + 1:]:
            ba, bb = tiles[ta][0], tiles[tb][0]
            r = _intersect(ba, bb)
            if r is None:
                continue
            ia, ib = by_tile[ta], by_tile[tb]
            ra = (slice(r[0] - ba[0], r[2] - ba[0]), slice(r[1] - ba[1], r[3] - ba[1]))
            rb = (slice(r[0] - bb[0], r[2] - bb[0]), slice(r[1] - bb[1], r[3] - bb[1]))
            pa = [insts[i][3][ra] for i in ia]
            pb = [insts[j][3][rb] for j in ib]
            na = [int(p.sum()) for p in pa]
            nb = [int(p.sum()) for p in pb]
            L = np.zeros((len(ia), len(ib)))
            for x, (p, n1) in enumerate(zip(pa, na)):
                if n1 < min_px:
                    continue
                for y, (q, n2) in enumerate(zip(pb, nb)):
                    if n2 < min_px:
                        continue
                    L[x, y] = np.logical_and(p, q).sum() / min(n1, n2)
            for x in range(len(ia)):
                if not len(ib):
                    break
                y = int(np.argmax(L[x]))
                if L[x, y] >= link_thr and int(np.argmax(L[:, y])) == x:
                    uf.union(ia[x], ib[y])
                    links.append(((ta, insts[ia[x]][1]), (tb, insts[ib[y]][1]), round(float(L[x, y]), 4)))
    groups = {}
    for i in range(len(insts)):
        groups.setdefault(uf.find(i), []).append(i)
    out = []
    for members in groups.values():
        full = np.zeros((height, width), dtype=bool)
        wsum = ssum = 0.0
        for i in members:
            t, _, s, m = insts[i]
            y0, x0, y1, x1 = tiles[t][0]
            full[y0:y1, x0:x1] |= m
            a = float(m.sum())
            wsum += a
            ssum += a * s
        score = ssum / wsum if wsum > 0 else max(insts[i][2] for i in members)
        ys, xs = np.nonzero(full.any(axis=1))[0], np.nonzero(full.any(axis=0))[0]
        box = [xs[0], ys[0], xs[-1] + 1, ys[-1] + 1] if len(xs) else [0, 0, 0, 0]
        out.append((score, box, full, [(insts[i][0], insts[i][1]) for i in members]))
    out.sort(key=lambda o: -o[0])
    bboxes = np.array([b + [s] for s, b, _, _ in out], dtype=np.float32).reshape(-1, 5)
    return bboxes, [o[2] for o in out], dict(members=[o[3] for o in out], links=links)


def run_tiled(model, img, size=512, overlap=128, score_thr=0.3, link_thr=0.5, min_px=20, infer_fn=None,
              return_info=False):
    """Detector over native crops of ``img`` (HxWx3 ndarray), merged; result shaped like inference_detector's.

    ``model`` should keep crops at native size (scale_compat.build_model(..., size='native')). ``infer_fn(model,
    crop)`` defaults to mmdet's inference_detector and must return ([bboxes], [masks]) for one class.
    """
    if infer_fn is None:
        from mmdet.apis import inference_detector as infer_fn
    img = np.ascontiguousarray(img)
    H, W = img.shape[:2]
    tiles = []
    for box in crop_boxes(H, W, size, overlap):
        y0, x0, y1, x1 = box
        crop = np.ascontiguousarray(img[y0:y1, x0:x1])
        res = infer_fn(model, crop)
        bb, mm = res[0][0], res[1][0]
        keep = np.nonzero(np.asarray(bb)[:, 4] >= score_thr)[0] if len(bb) else np.zeros(0, int)
        masks = np.array([np.asarray(mm[k], dtype=bool) for k in keep]).reshape(-1, y1 - y0, x1 - x0)
        tiles.append((box, np.asarray(bb)[keep, 4] if len(keep) else np.zeros(0), masks))
    bboxes, masks, info = merge_instances(tiles, H, W, score_thr, link_thr, min_px)
    info['tiles'] = [t[0] for t in tiles]
    result = ([bboxes], [masks])
    return (result, info) if return_info else result
