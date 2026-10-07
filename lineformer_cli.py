"""Command line inference: chart images -> line data series (JSON), on CPU or a GPU (CUDA or ROCm).

Three forms:

    lineformer --ckpt iter_3000.pth --out out/ chart1.png chart2.png              # single process
    lineformer batch --ckpt iter_3000.pth --list images.txt --out out/ --gpu-workers 2
    lineformer serve --ckpt iter_3000.pth --port 8775 --gpu-workers 2             # job server

Defaults of all three forms (since 0.2.0):
  * --device auto: 'cuda:0' if PyTorch sees a GPU (CUDA or ROCm), else 'cpu'; the choice is printed.
  * kept-queries mode ON at 0.3 (kept_queries.py): only the queries whose class score reaches --kept-thr are
    post-processed and returned. The lines are the same as with all queries (a line needs a final score > 0.3,
    and final score <= class score), at ~4x less GPU time and ~1 GB instead of ~12 GB device memory. The
    instances written with --instances / --masks are then only the kept queries. --all-queries switches it off
    (all 100 queries, as upstream). The environment variable LINEFORMER_KEPT_QUERIES is not read here.
  * batch / serve: 1 GPU worker (2 is the measured best), min(8, max(2, CPUs // 3)) pre-processing workers.

Single process: per image <out>/<stem>.json: {"image": path, "lines": [[{"x":.., "y":..}, ...], ...]} from
infer.get_dataseries (score threshold 0.3, as upstream). --masks also writes <stem>.masks.npz (the masks of the
instances behind the lines, packed bits per mask: masks, shape). Existing outputs are skipped unless --force.
Image reading runs ahead of the model in threads.

`lineformer batch` runs one job on the engine (lineformer_engine.py: pre-processing workers -> N GPU workers ->
post-processing workers) and `lineformer serve` keeps the engine up and takes jobs over HTTP (lineformer_serve.py;
client: lineformer_client.py). Their outputs, ids and skip rules are in lineformer_jobs.py: <out>/<id>.json (the
lines and provenance), optional <id>.instances.npz / <id>.masks.npz, and the manifest <out>/job.json.
`lineformer batch --help` lists the options. Exit codes of batch: 0 every image done or skipped, 1 some images
failed (or the job was cancelled), 2 the engine failed (e.g. out of GPU memory), 130 interrupted (rerun to resume).

--input-size native and --tile are EXPERIMENTAL: in an in-sample test on dense chart grids, native-resolution
input made the model segment grid lines as data lines (precision 0.97 -> ~0.2); results are best near the training
scale (~512 px per chart, the default).
"""
import argparse
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / 'lineformer_swin_t_config.py'
SUBCOMMANDS = ('batch', 'serve')
DEFAULT_KEPT_THR = 0.3
EXPERIMENTAL = ('EXPERIMENTAL: in an in-sample test on dense chart grids, native-resolution input made the model '
                'segment grid lines as data lines (precision 0.97 -> ~0.2); best results near the training scale '
                '(~512 px per chart)')


def default_pre_workers():
    """min(8, max(2, CPUs // 3)): enough readers to keep two GPU workers fed, without starving them of CPU."""
    return min(8, max(2, (os.cpu_count() or 1) // 3))


def _read(path):
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f'cannot read image {path}')
    return img


def _log(msg):
    print('[lineformer] %s' % msg, file=sys.stderr, flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in SUBCOMMANDS:
        return {'batch': main_batch, 'serve': main_serve}[argv[0]](argv[1:])
    return main_single(argv)


def _device_arg(g):
    g.add_argument('--device', default='auto',
                   help="'auto' (default: cuda:0 if PyTorch sees a GPU, else cpu), 'cpu', 'cuda' or 'cuda:N' "
                        "(ROCm GPUs too)")


def _kept_args(g):
    g.add_argument('--all-queries', action='store_true',
                   help='switch the kept-queries mode off: post-process and return all 100 queries, as upstream '
                        '(same lines; ~4x more GPU time and up to ~12 GB device memory per process). Needed only '
                        'for instances whose class score is below --kept-thr')
    g.add_argument('--kept-only', action='store_true',
                   help='kept-queries mode (kept_queries.py); ON by default, the flag is accepted for older '
                        'command lines')
    g.add_argument('--kept-thr', type=float, default=None,
                   help='class-score threshold of the kept-queries mode (default %.1f; e.g. 0.1 keeps lower-scoring '
                        'instances in .instances.npz / .masks.npz; lines still use score > 0.3)' % DEFAULT_KEPT_THR)


def _kept_thr(a, ap):
    """-> the kept-queries threshold, or None (mode off) with --all-queries."""
    if a.all_queries:
        if a.kept_only or a.kept_thr is not None:
            ap.error('--all-queries switches the kept-queries mode off; drop --kept-only / --kept-thr')
        return None
    return a.kept_thr if a.kept_thr is not None else DEFAULT_KEPT_THR


def _kept_text(kept):
    return 'off (all queries)' if kept is None else 'on, class-score threshold %g' % kept


# ------------------------------------------------------------------ single process (original form)

def main_single(argv=None):
    ap = argparse.ArgumentParser(prog='lineformer', description=__doc__.splitlines()[0],
                                 epilog='Subcommands: lineformer batch ... | lineformer serve ... (see --help of '
                                        'each). An image file named "batch" or "serve" needs a path: ./batch')
    ap.add_argument('images', nargs='*', type=Path)
    ap.add_argument('--list', type=Path, help='text file with one image path per line')
    ap.add_argument('--ckpt', required=True, type=Path)
    ap.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    _device_arg(ap)
    ap.add_argument('--msda', choices=('auto', 'compiled', 'pytorch'), default=None)
    _kept_args(ap)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--masks', action='store_true', help='also save the masks of the instances behind the lines')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    kept = _kept_thr(a, ap)

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
    if not todo:
        print(f'0 of {len(paths)} images to do', flush=True)
        return 0

    device = a.device
    if device == 'auto':
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'{len(todo)} of {len(paths)} images to do on {device}', flush=True)
    _log('device %s%s, kept-queries mode %s' % (device, ' (auto)' if a.device == 'auto' else '', _kept_text(kept)))
    import infer  # heavy imports after argument checks
    infer.load_model(str(a.config), str(a.ckpt), device, msda=a.msda, kept_only=False if kept is None else kept)

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


# ------------------------------------------------------------------ engine forms

def _engine_args(ap):
    g = ap.add_argument_group('model (one set per engine)')
    g.add_argument('--ckpt', required=True, type=Path)
    g.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    _device_arg(g)
    g.add_argument('--msda', choices=('auto', 'compiled', 'pytorch'), default=None,
                   help='MSDA path on a GPU (msda_compat.py; default env LINEFORMER_MSDA, else auto)')
    _kept_args(g)
    g.add_argument('--input-size', default='config',
                   help="network input: 'config' (fit 512, default), N (fit N x N) or 'native' (scale_compat.py). "
                        "'native' is " + EXPERIMENTAL)
    g.add_argument('--tile', type=int, default=None, metavar='CROP',
                   help='run native-resolution crops of CROP x CROP px and merge them (tiling.py); implies '
                        '--input-size native. ' + EXPERIMENTAL)
    g.add_argument('--tile-overlap', type=int, default=128, help='overlap of neighbouring crops in px (default 128)')
    g = ap.add_argument_group('workers')
    g.add_argument('--gpu-workers', type=int, default=1, help='model processes sharing the device (default 1; 2 '
                   'is the measured best)')
    g.add_argument('--gpu-mem-budget', default='0.85',
                   help='device memory all GPU workers together may use: a fraction (default 0.85) or a size such '
                        'as 4G; each worker gets budget / N. Out of memory ends the run')
    g.add_argument('--pre-workers', type=int, default=None,
                   help='image reading + resizing processes (default min(8, max(2, CPUs // 3)) = %d here)'
                        % default_pre_workers())
    g.add_argument('--post-workers', type=int, default=4)
    g.add_argument('--pre-threads', type=int, default=1)
    g.add_argument('--post-threads', type=int, default=1)
    g.add_argument('--gpu-threads', type=int, default=2)
    g.add_argument('--max-inflight', type=int, default=None, help='images between feeding and done (default '
                   '2 x all workers)')


def _make_engine(a, ap):
    from lineformer_engine import Engine, ModelOptions
    kept = _kept_thr(a, ap)
    size = a.input_size
    if a.tile is not None and size == 'config':
        size = 'native'
    mo = ModelOptions(ckpt=str(a.ckpt), config=str(a.config), device=a.device, msda=a.msda, kept_thr=kept,
                      input_size=size, tile=a.tile, tile_overlap=a.tile_overlap)
    try:
        mo = mo.resolved()
    except ValueError as e:
        ap.error(str(e))
    pre = a.pre_workers if a.pre_workers is not None else default_pre_workers()
    _log('device %s%s, kept-queries mode %s, %d GPU worker(s), %d pre-processing workers' % (
        mo.device, ' (auto)' if a.device == 'auto' else '', _kept_text(kept), a.gpu_workers, pre))
    if size == 'native':
        _log('WARNING: --input-size native / --tile is ' + EXPERIMENTAL)
    return Engine(mo, gpu_workers=a.gpu_workers, gpu_mem_budget=a.gpu_mem_budget, pre_workers=pre,
                  post_workers=a.post_workers, pre_threads=a.pre_threads, post_threads=a.post_threads,
                  gpu_threads=a.gpu_threads, max_inflight=a.max_inflight)


def main_batch(argv):
    import lineformer_jobs as jobs
    from lineformer_engine import EngineFailed
    ap = argparse.ArgumentParser(prog='lineformer batch', description='Run one job of images on the LineFormer '
                                 'engine (pre-processing workers -> GPU workers -> post-processing workers).')
    ap.add_argument('images', nargs='*', help='image paths (also: --list)')
    ap.add_argument('--list', type=Path, help='list file: "<path>" or "<id>\\t<path>" per line')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--ids', choices=jobs.ID_SCHEMES, default='stem',
                    help='output id of an image without an explicit id: file stem (default) or <parent>__<stem>')
    ap.add_argument('--instances', action='store_true',
                    help='also write <id>.instances.npz (boxes with scores; with the default kept-queries mode only '
                         'the kept queries, --all-queries for all 100)')
    ap.add_argument('--masks', action='store_true', help='also write <id>.masks.npz (masks of the same instances)')
    ap.add_argument('--force', action='store_true', help='recompute images that are done')
    ap.add_argument('--progress-s', type=float, default=10.0, help='progress line interval (s)')
    _engine_args(ap)
    a = ap.parse_args(argv)
    items = list(a.images)
    if a.list:
        items += jobs.read_list_file(a.list)
    if not items:
        ap.error('no images given')
    try:
        jobs.normalize_items(items, a.ids)  # bad input fails before the models load
    except jobs.JobError as e:
        ap.error(str(e))
    eng = _make_engine(a, ap)
    state = {'signals': 0}

    def on_signal(signum, frame):
        state['signals'] += 1
        if state['signals'] == 1:
            eng.log('signal %d: stopping (in-flight images finish; Ctrl-C again abandons them)' % signum)
            threading.Thread(target=eng.shutdown, daemon=True).start()
        else:
            eng._abandon.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        jid = eng.submit(items, str(a.out), outputs={'instances': a.instances, 'masks': a.masks}, force=a.force,
                         ids=a.ids)  # decides skip-if-done before any model is loaded
    except (jobs.JobError, EngineFailed) as e:
        eng.shutdown()
        print('job refused: %s' % e, file=sys.stderr)
        return 2
    if eng.status(jid)['status'] in jobs.FINAL_JOB_STATES:
        eng.log('nothing to do: no model loaded')
    else:
        try:
            eng.start()
        except EngineFailed as e:
            print('engine failed to start: %s' % e, file=sys.stderr)
            s = eng.status(jid)
            print(json.dumps({'job': jid, 'status': s['status'], 'counts': s['counts'], 'error': s['error']}))
            return 2
    while True:
        try:
            s = eng.wait(jid, timeout=a.progress_s)
            break
        except TimeoutError:
            s = eng.status(jid)
            c, t = s['counts'], s['timing']
            print('[lineformer batch] %s: done %d skipped %d failed %d pending %d running %d of %d, %.2f images/s'
                  % (s['status'], c['done'], c['skipped'], c['failed'], c['pending'], c['running'], c['total'],
                     t.get('images_per_s') or 0.0), file=sys.stderr, flush=True)
    eng.shutdown()
    s = eng.status(jid)
    c, t = s['counts'], s['timing']
    print(json.dumps({'job': jid, 'status': s['status'], 'counts': c, 'images_per_s': t.get('images_per_s'),
                      'wall_s': t.get('wall_s'), 'manifest': str(Path(s['out']) / jobs.MANIFEST),
                      'error': s['error']}), flush=True)
    for e in s['errors'][:10]:
        print('FAILED %s (%s): %s' % (e['id'], e['path'], (e['error'] or '').strip().splitlines()[-1:]),
              file=sys.stderr)
    return {'done': 0, 'done_with_errors': 1, 'cancelled': 1, 'failed': 2, 'interrupted': 130}.get(s['status'], 2)


def main_serve(argv):
    import lineformer_serve
    ap = argparse.ArgumentParser(prog='lineformer serve', description=lineformer_serve.__doc__.splitlines()[0],
                                 epilog='API: see lineformer_serve.py; client: lineformer_client.py')
    ap.add_argument('--host', default='127.0.0.1', help='bind address (default 127.0.0.1; no authentication)')
    ap.add_argument('--port', type=int, default=8775)
    ap.add_argument('--drain-timeout', type=float, default=600.0,
                    help='on SIGINT/SIGTERM, seconds to let in-flight images finish')
    ap.add_argument('--ready-file', type=Path, default=None, help='write {"url": ...} here once serving')
    ap.add_argument('--verbose', action='store_true', help='log every HTTP request')
    _engine_args(ap)
    a = ap.parse_args(argv)
    eng = _make_engine(a, ap)
    return lineformer_serve.serve(eng, host=a.host, port=a.port, drain_timeout=a.drain_timeout, verbose=a.verbose,
                                  ready_file=str(a.ready_file) if a.ready_file else None)


if __name__ == '__main__':
    sys.exit(main())
