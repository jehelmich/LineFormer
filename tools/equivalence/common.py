# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""Shared I/O for the equivalence harness (no torch / mmcv imports here).

One run directory holds, per image id:
  <id>.npz              boxes (N x 5 float, x1 y1 x2 y2 score), labels (N), masks packed with np.packbits
                        over the flattened (N, H, W) bool array, mask_shape = (N, H, W)
  <id>.dataseries.json  infer.get_dataseries(img, to_clean=False) output: list of lines, each a list of {x, y}
  <id>.meta.json        status ("ok" or "error" + traceback), timings, input hashes
and run_meta.json for the whole run (versions, device evidence, the list of image ids requested).

Image ids: a list file has one image per line, either "<path>" or "<id>\\t<path>". Without an explicit id the
id is "<parent dir name>__<file stem>", so that equally named files in different folders stay apart.
Python 3.8 compatible.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def image_id_for(path):
    p = Path(path)
    return "%s__%s" % (p.parent.name, p.stem)


def read_image_list(spec):
    """Return [(id, path)] from a list file or a directory (recursive, sorted). Duplicate ids raise."""
    spec = Path(spec)
    items = []
    if spec.is_dir():
        for p in sorted(spec.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                items.append((image_id_for(p), str(p)))
    else:
        for line in spec.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                iid, path = line.split("\t", 1)
                items.append((iid.strip(), path.strip()))
            else:
                items.append((image_id_for(line), line))
    ids = [i for i, _ in items]
    dup = sorted(set(i for i in ids if ids.count(i) > 1))
    if dup:
        raise ValueError("duplicate image ids in %s: %s" % (spec, dup))
    if not items:
        raise ValueError("no images found in %s" % spec)
    return items


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError("not JSON serialisable: %r" % type(o))


def write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, str(path))


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_instances(path, boxes, labels, masks):
    """boxes (N,5), labels (N,), masks (N,H,W) bool -> compressed npz with packed masks."""
    boxes = np.asarray(boxes)
    labels = np.asarray(labels, dtype=np.int64)
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or masks.shape[0] != boxes.shape[0] or labels.shape[0] != boxes.shape[0]:
        raise ValueError("inconsistent shapes boxes %s labels %s masks %s" % (boxes.shape, labels.shape, masks.shape))
    np.savez_compressed(
        str(path),
        boxes=boxes,
        labels=labels,
        masks_packed=np.packbits(masks.reshape(-1)),
        mask_shape=np.asarray(masks.shape, dtype=np.int64),
    )


def load_instances(path):
    """-> dict(boxes (N,5) float64, scores (N,), labels (N,), masks (N,H,W) bool)."""
    with np.load(str(path)) as z:
        shape = tuple(int(v) for v in z["mask_shape"])
        n = int(np.prod(shape))
        masks = np.unpackbits(z["masks_packed"], count=n).astype(bool).reshape(shape)
        boxes = z["boxes"].astype(np.float64).reshape(-1, 5)
        labels = z["labels"]
    return dict(boxes=boxes, scores=boxes[:, 4].copy(), labels=labels, masks=masks)


def run_paths(run_dir, iid):
    d = Path(run_dir)
    return dict(npz=d / (iid + ".npz"), ds=d / (iid + ".dataseries.json"), meta=d / (iid + ".meta.json"))
