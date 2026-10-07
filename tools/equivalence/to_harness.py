# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""Convert an engine output directory (`lineformer batch` / `serve` with instances and masks) into a run directory of
tools/equivalence, so that compare.py can judge the engine against reference runs.

    python tools/equivalence/to_harness.py --engine-out out/ --out runs/engine_kept [--tag engine_kept]

Per image of the manifest (<engine-out>/job.json): <id>.npz (boxes, labels from <id>.instances.npz; masks from
<id>.masks.npz - both are required), <id>.dataseries.json (the "lines" of <id>.json), <id>.meta.json (status,
file_sha256 = the engine's image_sha256, pixels_sha256 recomputed here from cv2.imread as run.py does, shape) and
run_meta.json (image ids, status, the engine's options and versions). An image that is not done/skipped in the
manifest, or whose outputs are missing, is written as status "error" (compare.py then fails it, never skips it).
Duplicate entries of the manifest are left out (they have no outputs of their own). Python 3.8 compatible.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import common  # noqa: E402
import lineformer_jobs as jobs  # noqa: E402


def convert(engine_out, out, tag=None):
    import cv2
    import numpy as np
    engine_out, out = Path(engine_out), Path(out)
    if engine_out.resolve() == out.resolve():
        raise SystemExit('--out must differ from --engine-out')
    man = common.read_json(engine_out / jobs.MANIFEST)
    out.mkdir(parents=True, exist_ok=True)
    ids, n_err = [], 0
    for r in man['images']:
        if r['status'] == 'duplicate':
            continue
        iid = r['id']
        ids.append(iid)
        paths = common.run_paths(out, iid)
        meta = {'id': iid, 'path': r['path'], 'status': 'error', 'engine_status': r['status']}
        try:
            if r['status'] not in ('done', 'skipped'):
                raise RuntimeError('engine status %s: %s' % (r['status'], r.get('error')))
            p = jobs.output_paths(engine_out, iid)
            obj = common.read_json(p['json'])
            for k in ('instances', 'masks'):
                if k not in obj.get('outputs', []) or not p[k].exists():
                    raise RuntimeError('%s output missing (run the engine with --instances --masks)' % k)
            with np.load(str(p['instances'])) as z:
                boxes, labels = z['boxes'], z['labels']
            with np.load(str(p['masks'])) as z:
                shape = tuple(int(v) for v in z['mask_shape'])
                masks = np.unpackbits(z['masks_packed'], count=int(np.prod(shape))).astype(bool).reshape(shape)
            if len(boxes) != shape[0]:
                raise RuntimeError('%d boxes but %d masks' % (len(boxes), shape[0]))
            common.save_instances(paths['npz'], boxes, labels, masks)
            common.write_json(paths['ds'], obj['lines'])
            img = cv2.imread(obj['image'])
            if img is None:
                raise RuntimeError('cannot read %s for its pixel hash' % obj['image'])
            meta.update(status='ok', file_sha256=obj['image_sha256'], pixels_sha256=common.sha256_array(img),
                        shape=list(img.shape), n_instances=int(len(boxes)),
                        **{'n_ge_0.3': int((boxes[:, 4] >= 0.3).sum())}, n_lines=len(obj['lines']),
                        timings=obj.get('timings'), job=obj.get('job'))
        except Exception as e:
            meta['error'] = '%s: %s' % (type(e).__name__, e)
            n_err += 1
        common.write_json(paths['meta'], meta)
    eng = man.get('engine') or {}
    run_meta = {'tag': tag or out.name, 'runner': 'lineformer engine (tools/equivalence/to_harness.py)',
                'engine_out': str(engine_out.resolve()), 'job': man.get('job'), 'job_status': man.get('status'),
                'image_ids': ids, 'device_requested': eng.get('device'), 'msda_handling': eng.get('msda_path'),
                'kept_queries_threshold': eng.get('kept_thr'), 'versions': eng.get('versions'),
                'engine': eng, 'timing': man.get('timing'), 'n_errors': n_err,
                'status': 'done' if n_err == 0 else 'done_with_errors', 'converted': time.strftime('%Y-%m-%d %H:%M:%S')}
    common.write_json(out / 'run_meta.json', run_meta)
    return run_meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--engine-out', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default=None)
    a = ap.parse_args(argv)
    m = convert(a.engine_out, a.out, a.tag)
    print('%s: %d images, %d errors -> %s' % (m['tag'], len(m['image_ids']), m['n_errors'], a.out))
    return 1 if m['n_errors'] else 0


if __name__ == '__main__':
    sys.exit(main())
