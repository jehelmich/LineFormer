"""Command line inference: chart images -> line data series (JSON), on CPU or a GPU (CUDA or ROCm).

    lineformer --ckpt iter_3000.pth --device cuda:0 --out out/ chart1.png chart2.png
    lineformer --ckpt iter_3000.pth --list images.txt --out out/ --masks

Per image <out>/<stem>.json: {"image": path, "lines": [[{"x":..,"y":..}, ...], ...]} from infer.get_dataseries
(score threshold 0.3, as upstream). --masks also writes <stem>.masks.npz (the kept instance masks, packed bits).
Existing outputs are skipped unless --force. Image reading runs ahead of the model in threads.
--kept-only (opt-in, kept_queries.py) post-processes only the queries whose class score reaches --kept-thr: same
lines, less GPU time and memory.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / 'lineformer_swin_t_config.py'


def _read(path):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f'cannot read image {path}')
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(prog='lineformer', description=__doc__.splitlines()[0])
    ap.add_argument('images', nargs='*', type=Path)
    ap.add_argument('--list', type=Path, help='text file with one image path per line')
    ap.add_argument('--ckpt', required=True, type=Path)
    ap.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    ap.add_argument('--device', default='cuda:0', help="'cpu', 'cuda' or 'cuda:N' (ROCm GPUs too)")
    ap.add_argument('--msda', choices=('auto', 'compiled', 'pytorch'), default=None)
    ap.add_argument('--kept-only', action='store_true',
                    help='speed-up: post-process only the queries whose class score reaches --kept-thr; the lines '
                         'are the same, instances below the threshold are not returned (kept_queries.py). '
                         'Without the flag: env LINEFORMER_KEPT_QUERIES, else off')
    ap.add_argument('--kept-thr', type=float, default=0.3, help='threshold of --kept-only (default 0.3)')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--masks', action='store_true', help='also save the kept instance masks')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)

    paths = list(a.images)
    if a.list:
        paths += [Path(s.strip()) for s in a.list.read_text().splitlines() if s.strip()]
    if not paths:
        ap.error('no images given')
    a.out.mkdir(parents=True, exist_ok=True)
    stems = [p.stem for p in paths]
    if len(set(stems)) != len(stems):
        raise SystemExit('two images share a file name stem; outputs would collide')
    todo = [p for p in paths if a.force or not (a.out / f'{p.stem}.json').exists()]
    print(f'{len(todo)} of {len(paths)} images to do on {a.device}', flush=True)
    if not todo:
        return 0

    import infer  # heavy imports after argument checks
    infer.load_model(str(a.config), str(a.ckpt), a.device, msda=a.msda,
                     kept_only=a.kept_thr if a.kept_only else None)

    t0 = time.time()
    ahead = 8
    with ThreadPoolExecutor(4) as pool:
        futs = [pool.submit(_read, p) for p in todo[:ahead]]
        for i, p in enumerate(todo):
            if i + ahead < len(todo):
                futs.append(pool.submit(_read, todo[i + ahead]))
            img = futs[i].result()
            futs[i] = None
            lines, masks = infer.get_dataseries(img, to_clean=False, return_masks=True)
            rec = {'image': str(p), 'lines': [[{'x': float(q['x']), 'y': float(q['y'])} for q in ln] for ln in lines]}
            tmp = a.out / f'{p.stem}.json.tmp'
            tmp.write_text(json.dumps(rec))
            tmp.replace(a.out / f'{p.stem}.json')
            if a.masks:
                np.savez_compressed(a.out / f'{p.stem}.masks.npz', shape=np.array(img.shape[:2]),
                                    masks=np.packbits(np.array([m > 0 for m in masks], dtype=bool).reshape(len(masks), -1), axis=1))
            if (i + 1) % 50 == 0:
                print(i + 1, 'done', round(time.time() - t0), 's', flush=True)
    print('finished', len(todo), round(time.time() - t0), 's', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())