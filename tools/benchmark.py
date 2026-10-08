# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""Throughput of the job engine on your own images: images/s with the warm-up excluded, per-stage times and the
device memory of each GPU worker.

    python tools/benchmark.py --list images.txt [--settings my.toml] [--repeat 2] [--json r.json]

Same engine and model options as `lineformer batch`, with its defaults (device auto, kept-queries mode at 0.3,
GPU workers sized automatically; --threshold, --cpu, --settings and the checkpoint lookup as there; set
gpu_workers / gpu_mem_budget in the settings file to compare configurations). The engine starts (model load and
worker start-up are not timed), a warm-up job runs the first --warmup images (not timed), then --repeat timed jobs
run over all images. Every job writes into a fresh temporary directory (lines only unless --instances / --masks),
deleted at the end unless --keep.

Reported per timed pass: images/s on the job clock (first image fed to last image done), median pre / GPU / post
seconds per image; per GPU worker the peak allocated and reserved device memory (torch.cuda max_memory_*). Numbers
depend on the machine, the image sizes and other load: compare configurations on the same images in one sitting.
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lineformer_cli as cli  # noqa: E402
import lineformer_jobs as jobs  # noqa: E402


def _median(stats):
    return None if not stats else round(stats['median'], 4)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('images', nargs='*', help='image paths (also: --list)')
    ap.add_argument('--list', type=Path, help='list file: "<path>" or "<id>\\t<path>" per line')
    ap.add_argument('--ids', choices=jobs.ID_SCHEMES, default='stem')
    ap.add_argument('--warmup', type=int, default=8, help='images of the untimed warm-up job (default 8; 0 = none)')
    ap.add_argument('--repeat', type=int, default=2, help='timed passes over all images (default 2)')
    ap.add_argument('--instances', action='store_true')
    ap.add_argument('--masks', action='store_true')
    ap.add_argument('--work-dir', type=Path, default=None, help='parent of the temporary output directory')
    ap.add_argument('--keep', action='store_true', help='keep the outputs')
    ap.add_argument('--json', type=Path, default=None, help='write the report here')
    cli._engine_args(ap)
    a = ap.parse_args(argv)
    items = list(a.images) + (jobs.read_list_file(a.list) if a.list else [])
    if not items:
        ap.error('no images given')
    if a.repeat < 1:
        ap.error('--repeat must be >= 1')
    jobs.normalize_items(items, a.ids)
    eng = cli._make_engine(a, ap, 'serve')  # engine options only (--ids, --instances, --masks are this tool's)
    work = Path(tempfile.mkdtemp(prefix='lineformer_bench_', dir=str(a.work_dir) if a.work_dir else None))
    outputs = {'instances': a.instances, 'masks': a.masks}

    def run(name, its):
        jid = eng.submit(its, str(work / name), outputs=outputs, force=True, ids=a.ids)
        s = eng.wait(jid)
        if s['status'] != 'done':
            raise SystemExit('%s: job %s %s %s' % (name, jid, s['status'], s['errors'][:3] or s['error']))
        return s

    report = {'images': len(items), 'engine': None, 'passes': [], 'gpu_workers': {}}
    try:
        eng.start()
        report['engine'] = {k: eng.engine_info.get(k) for k in (
            'device', 'msda_path', 'kept_thr', 'input_size', 'tile', 'gpu_workers', 'pre_workers', 'post_workers',
            'gpu_mem_budget', 'versions')}
        if a.warmup > 0:
            run('warmup', items[:a.warmup])
        for r in range(a.repeat):
            s = run('pass%d' % (r + 1), items)
            t = s['timing']
            p = {'pass': r + 1, 'done': s['counts']['done'], 'wall_s': round(t['wall_s'], 2),
                 'images_per_s': round(t['images_per_s'], 2), 'pre_s_median': _median(t['pre_s']),
                 'gpu_s_median': _median(t['gpu_s']), 'post_s_median': _median(t['post_s'])}
            report['passes'].append(p)
            print('pass %d: %d images, %.2f images/s (pre %s s, GPU %s s, post %s s per image, medians)' % (
                p['pass'], p['done'], p['images_per_s'], p['pre_s_median'], p['gpu_s_median'], p['post_s_median']),
                flush=True)
        t_end = time.time()
        deadline = t_end + 5.0  # the GPU workers report memory every ~2 s; wait for a report from after the passes
        while time.time() < deadline:
            with eng.cond:
                eng._drain_stats()
                fresh = len(eng.worker_stats) == eng.n_gpu and all(v.get('t', 0) > t_end
                                                                    for v in eng.worker_stats.values())
            if fresh:
                break
            time.sleep(0.25)
        with eng.cond:
            for wid, v in sorted(eng.worker_stats.items()):
                report['gpu_workers'][str(wid)] = {k: (round(v[k], 1) if isinstance(v.get(k), float) else v.get(k))
                                                   for k in ('max_allocated_MB', 'max_reserved_MB', 'n_images')}
    finally:
        eng.shutdown()
        if not a.keep:
            shutil.rmtree(str(work), ignore_errors=True)
    for wid, m in report['gpu_workers'].items():
        print('GPU worker %s: peak allocated %s MB, peak reserved %s MB, %s images' % (
            wid, m['max_allocated_MB'], m['max_reserved_MB'], m['n_images']))
    rates = [p['images_per_s'] for p in report['passes']]
    print('images/s over %d timed pass(es): %s (best %.2f)' % (len(rates), rates, max(rates)))
    if a.keep:
        print('outputs kept in %s' % work)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
