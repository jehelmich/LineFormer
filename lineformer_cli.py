# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""Command line inference: chart images -> line data series (JSON), on CPU or a GPU (CUDA or ROCm).

Three forms, one small set of options (since 0.3.0):

    lineformer IMAGES... | --list FILE  --out DIR  [--threshold 0.3] [--masks] [--force] [--cpu] [--settings F.toml]
    lineformer batch     (the same options; the job engine, for many images)
    lineformer serve     [--port 8775] [--exit-on-failure] [--threshold] [--masks] [--cpu] [--settings F.toml]

Defaults:
  * device: the GPU ('cuda:0', CUDA or ROCm) if PyTorch sees one, else the CPU; --cpu forces the CPU.
  * --threshold T (default 0.3, as upstream) is the score a line needs to be reported: the lines are the instances
    whose final score (class score x mask score) is > T, and only the queries whose class score reaches T are
    post-processed (the kept-queries mode, kept_queries.py; exact, since final score <= class score). Lower values
    also return fainter, less certain lines; T is part of the output fingerprint. all_queries = true in the
    settings switches the kept-queries mode off (all 100 queries, as upstream; ~4x more GPU time and up to ~12 GB
    device memory per process); the line threshold is still T. LINEFORMER_KEPT_QUERIES is not read.
  * batch / serve: the GPU workers are sized from the free device memory at start (lineformer_engine.py,
    "Automatic sizing"); out of device memory on an image stops that worker and requeues the image once
    ("bounded back-off"); min(8, max(2, CPUs // 3)) pre-processing workers.
  * the checkpoint: --ckpt FILE, else the environment variable LINEFORMER_CKPT, else <fork root>/iter_3000.pth,
    else ~/.cache/lineformer/iter_3000.pth; none -> an error naming the paths searched and where to download it.

Settings file (--settings FILE.toml; lineformer.example.toml lists every key with its default): the model config,
input size, tiling (EXPERIMENTAL), all_queries, ids, worker counts, threads, memory budget, MSDA path. Unknown keys
fail. The flags of v0.2.0 that set these values (--gpu-workers, --gpu-mem-budget, --kept-thr, --input-size,
--instances, --device, ...) are still accepted in this version, with their values applied and a one-line
deprecation warning; a flag and the settings file that disagree fail. The effective settings (after the automatic
sizing) are in each job manifest (<out>/job.json, engine.settings).

Single process: per image <out>/<id>.json: {"image": path, "lines": [[{"x":.., "y":..}, ...], ...]} from
infer.get_dataseries (the model's order, as upstream). --masks also writes <id>.masks.npz (the masks of the
instances behind the lines, packed bits per mask: masks, shape). Existing outputs are skipped unless --force.

`lineformer batch` runs one job on the engine (lineformer_engine.py: pre-processing workers -> GPU workers ->
post-processing workers) and `lineformer serve` keeps the engine up and takes jobs over HTTP (lineformer_serve.py;
client: lineformer_client.py). Their outputs, ids and skip rules are in lineformer_jobs.py: <out>/<id>.json (the
lines and provenance), with --masks <id>.masks.npz and <id>.instances.npz, and the manifest <out>/job.json. Ids
are file stems (or "<id><TAB><path>" lines in the list file); two images with one id stop batch before it runs.
Exit codes of batch: 0 every image done or skipped, 1 some images failed (or the job was cancelled), 2 the engine
failed (e.g. out of GPU memory after the back-off) or bad input, 130 interrupted (rerun to resume).

input_size 'native' and tile are EXPERIMENTAL: in an in-sample test on dense chart grids, native-resolution input
made the model segment grid lines as data lines (precision 0.97 -> ~0.2); results are best near the training scale
(~512 px per chart, the default).
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

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / 'lineformer_swin_t_config.py'
SUBCOMMANDS = ('batch', 'serve')
DEFAULT_KEPT_THR = 0.3  # --threshold default: the line threshold and the kept-queries threshold
THRESHOLD_HELP = ('score a line needs to be reported (default 0.3, as upstream); lower values also return fainter, '
                  'less certain lines')
EXPERIMENTAL = ('EXPERIMENTAL: in an in-sample test on dense chart grids, native-resolution input made the model '
                'segment grid lines as data lines (precision 0.97 -> ~0.2); best results near the training scale '
                '(~512 px per chart)')
CKPT_NAME = 'iter_3000.pth'
CKPT_ENV = 'LINEFORMER_CKPT'
CKPT_URL = 'https://drive.google.com/drive/folders/1K_zLZwgoUIAJtfjwfCU5Nv33k17R0O5T?usp=sharing'
EXAMPLE_SETTINGS = 'lineformer.example.toml'
FORMS = ('single', 'batch', 'serve')


def default_pre_workers():
    """min(8, max(2, CPUs // 3)): enough readers to keep two GPU workers fed, without starving them of CPU."""
    return min(8, max(2, (os.cpu_count() or 1) // 3))


def _read(path):
    import cv2
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


# ================================================================== checkpoint

class SetupError(ValueError):
    """Bad options, settings or setup found before any model is loaded (exit code 2)."""


def resolve_ckpt(flag=None, environ=None, root=HERE, home=None):
    """-> (absolute checkpoint path, source). Order: --ckpt, $LINEFORMER_CKPT, <fork root>/iter_3000.pth,
    ~/.cache/lineformer/iter_3000.pth. A path given by the flag or the variable must exist (no fall-through)."""
    environ = os.environ if environ is None else environ
    if flag:
        p = Path(flag).expanduser().resolve()
        if not p.is_file():
            raise SetupError('checkpoint not found: --ckpt %s' % p)
        return str(p), '--ckpt'
    env = (environ.get(CKPT_ENV) or '').strip()
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_file():
            raise SetupError('checkpoint not found: %s=%s' % (CKPT_ENV, p))
        return str(p), CKPT_ENV
    home = Path(home) if home is not None else Path.home()
    cands = [(Path(root) / CKPT_NAME, 'fork root'), (home / '.cache' / 'lineformer' / CKPT_NAME, 'user cache')]
    for p, src in cands:
        if p.is_file():
            return str(p.resolve()), src
    raise SetupError('no checkpoint found. Searched: --ckpt (not given), %s (not set), %s. Download %s from the '
                     'authors\' link (%s; README, "Inference") and put it at one of these paths, set %s, or pass '
                     '--ckpt FILE' % (CKPT_ENV, ', '.join(str(p) for p, _ in cands), CKPT_NAME, CKPT_URL,
                                      CKPT_ENV))


# ================================================================== settings

def _int_at_least(lo):
    def check(v):
        if isinstance(v, bool) or not isinstance(v, int) or v < lo:
            raise ValueError('an integer >= %d, got %r' % (lo, v))
        return v
    return check


def _bool(v):
    if not isinstance(v, bool):
        raise ValueError('true or false, got %r' % (v,))
    return v


def _choice(*choices):
    def check(v):
        if v not in choices:
            raise ValueError('one of %s, got %r' % (', '.join(repr(c) for c in choices), v))
        return v
    return check


def _positive_number(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        raise ValueError('a number > 0, got %r' % (v,))
    return float(v)


def _input_size(v):
    from lineformer_engine import parse_input_size
    if isinstance(v, bool):
        raise ValueError("'config', 'native' or a positive integer, got %r" % (v,))
    return parse_input_size(v)


def _budget(v):
    from lineformer_engine import parse_mem_budget
    if isinstance(v, bool) or not isinstance(v, (str, int, float)):
        raise ValueError('a fraction in (0, 1] or a size such as "4G", got %r' % (v,))
    parse_mem_budget(v)
    return v


def _path(v):
    if not isinstance(v, str) or not v:
        raise ValueError('a file path, got %r' % (v,))
    return v


# key: (default, check, forms that use it, help). The order is the order of lineformer.example.toml.
SETTINGS = {
    'config': (str(DEFAULT_CONFIG), _path, FORMS, 'model config (a path relative to the settings file)'),
    'input_size': ('config', _input_size, ('batch', 'serve'),
                   "network input: 'config' (fit 512), N (fit N x N) or 'native' (EXPERIMENTAL)"),
    'tile': (None, _int_at_least(1), ('batch', 'serve'), 'native crops of N x N px, merged (EXPERIMENTAL)'),
    'tile_overlap': (128, _int_at_least(0), ('batch', 'serve'), 'overlap of neighbouring crops (px)'),
    'all_queries': (False, _bool, FORMS, 'kept-queries mode off: all 100 queries (validation only)'),
    'msda': (None, _choice('auto', 'compiled', 'pytorch'), FORMS, 'MSDA path on a GPU (debugging only)'),
    'ids': ('stem', _choice('stem', 'parent_stem'), ('single', 'batch'), 'output id of an image'),
    'gpu_workers': (None, _int_at_least(1), ('batch', 'serve'), 'GPU workers (default: automatic sizing)'),
    'gpu_mem_budget': (None, _budget, ('batch', 'serve'),
                       'device memory of all GPU workers together (default: automatic sizing)'),
    'pre_workers': (None, _int_at_least(1), ('batch', 'serve'), 'pre-processing workers'),
    'post_workers': (4, _int_at_least(1), ('batch', 'serve'), 'post-processing workers'),
    'pre_threads': (1, _int_at_least(1), ('batch', 'serve'), 'threads per pre-processing worker'),
    'post_threads': (1, _int_at_least(1), ('batch', 'serve'), 'threads per post-processing worker'),
    'gpu_threads': (2, _int_at_least(1), ('batch', 'serve'), 'CPU threads per GPU worker'),
    'max_inflight': (None, _int_at_least(1), ('batch', 'serve'), 'images between feeding and done'),
    'progress_s': (10.0, _positive_number, ('batch',), 'progress line interval (s)'),
}
# result-changing keys: a form that cannot honour them fails instead of ignoring them
RESULT_KEYS = ('input_size', 'tile', 'tile_overlap', 'ids')

# deprecated flags (argparse dest -> settings key); still applied in this version, with a warning
FLAG_KEYS = {'config': 'config', 'input_size': 'input_size', 'tile': 'tile', 'tile_overlap': 'tile_overlap',
             'all_queries': 'all_queries', 'ids': 'ids', 'gpu_workers': 'gpu_workers',
             'gpu_mem_budget': 'gpu_mem_budget', 'pre_workers': 'pre_workers', 'post_workers': 'post_workers',
             'pre_threads': 'pre_threads', 'post_threads': 'post_threads', 'gpu_threads': 'gpu_threads',
             'max_inflight': 'max_inflight', 'progress_s': 'progress_s', 'msda': 'msda'}


def _flag(dest):
    return '--' + dest.replace('_', '-')


def load_settings(path):
    """Read and check a settings file -> {key: value} (only the keys it sets). Unknown keys or bad values raise
    SetupError. A relative config path is taken relative to the settings file."""
    try:
        import tomllib
    except ImportError:  # Python < 3.11
        try:
            import tomli as tomllib
        except ImportError:
            raise SetupError('--settings needs Python >= 3.11 (tomllib) or the tomli package') from None
    path = Path(path)
    try:
        with open(path, 'rb') as f:
            raw = tomllib.load(f)
    except OSError as e:
        raise SetupError('cannot read the settings file %s: %s' % (path, e)) from None
    except tomllib.TOMLDecodeError as e:
        raise SetupError('settings file %s is not valid TOML: %s' % (path, e)) from None
    unknown = sorted(set(raw) - set(SETTINGS))
    if unknown:
        raise SetupError('settings file %s: unknown key(s) %s (known: %s; see %s)' % (
            path, ', '.join(unknown), ', '.join(SETTINGS), EXAMPLE_SETTINGS))
    out = {}
    for k, v in raw.items():
        try:
            out[k] = SETTINGS[k][1](v)
        except ValueError as e:
            raise SetupError('settings file %s: %s must be %s' % (path, k, e)) from None
    if 'config' in out:
        out['config'] = str((path.resolve().parent / Path(out['config']).expanduser()).resolve())
    return out


def _same(key, a, b):
    """Do two values of a settings key mean the same?"""
    if key == 'gpu_mem_budget':
        from lineformer_engine import parse_mem_budget
        return parse_mem_budget(a) == parse_mem_budget(b)
    if key == 'input_size':
        return _input_size(a) == _input_size(b)
    if key == 'config':
        return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()
    return a == b


def effective_settings(a, form, warn=_log):
    """Settings file + deprecated flags + defaults -> (values {key: value}, sources {key: 'default' | 'settings' |
    flag}). Raises SetupError when a flag and the settings file disagree or a form cannot honour a key."""
    from_file = load_settings(a.settings) if getattr(a, 'settings', None) else {}
    values, sources = {}, {}
    for k, (default, check, forms, _) in SETTINGS.items():
        values[k], sources[k] = default, 'default'
        if k in from_file:
            values[k], sources[k] = from_file[k], 'settings'
    for dest, key in FLAG_KEYS.items():
        v = getattr(a, dest, None)
        if v is None or (form == 'serve' and dest in ('ids', 'progress_s')):  # (tools/benchmark.py has its own)
            continue
        warn('DEPRECATED %s: set %s in the settings file instead (--settings FILE.toml, see %s); the flag still '
             'works in this version' % (_flag(dest), key, EXAMPLE_SETTINGS))
        try:
            v = SETTINGS[key][1](v)
        except ValueError as e:
            raise SetupError('%s must be %s' % (_flag(dest), e)) from None
        if key in from_file and not _same(key, v, from_file[key]):
            raise SetupError('%s %r and %s = %r in the settings file %s disagree' % (
                _flag(dest), v, key, from_file[key], a.settings))
        values[key], sources[key] = v, _flag(dest)
    for k in from_file:
        if form not in SETTINGS[k][2]:
            if k in RESULT_KEYS and from_file[k] != SETTINGS[k][0]:
                raise SetupError('settings key %s is not supported by the %s form (use lineformer batch)'
                                 % (k, {'single': 'single-process', 'serve': 'serve'}.get(form, form)))
            warn('note: settings key %s is not used by the %s form' % (k, form))
    return values, sources


def _threshold(a, values):
    """-> (kept-queries threshold or None with all_queries, line threshold). Both are --threshold T."""
    kept_thr = getattr(a, 'kept_thr', None)
    if getattr(a, 'kept_only', None):
        _log('DEPRECATED --kept-only: no effect (the kept-queries mode is the default)')
    if kept_thr is not None:
        _log('DEPRECATED --kept-thr: use --threshold; the flag still works in this version')
        if a.threshold is not None and a.threshold != kept_thr:
            raise SetupError('--threshold %g and --kept-thr %g disagree' % (a.threshold, kept_thr))
    thr = a.threshold if a.threshold is not None else kept_thr
    thr = DEFAULT_KEPT_THR if thr is None else float(thr)
    if not 0.0 < thr < 1.0:
        raise SetupError('--threshold must lie in (0, 1), got %g' % thr)
    return (None if values['all_queries'] else thr), thr


def _device(a):
    dev = getattr(a, 'device', None)
    if dev is not None:
        _log('DEPRECATED --device: use --cpu for the CPU, or nothing for the automatic choice; the flag still works '
             'in this version')
        if a.cpu and dev != 'cpu':
            raise SetupError('--cpu and --device %s disagree' % dev)
        return dev
    return 'cpu' if a.cpu else 'auto'


def _thr_text(kept, line):
    return 'line threshold %g (score > %g), kept-queries mode %s' % (
        line, line, 'off (all queries)' if kept is None else 'on')


# ================================================================== argument parsers

H = argparse.SUPPRESS


def _public_args(ap, form):
    if form != 'serve':
        ap.add_argument('images', nargs='*', metavar='IMAGE', help='image files (or --list)')
        ap.add_argument('--list', type=Path, metavar='FILE',
                        help='text file with one image per line: "<path>" or "<id><TAB><path>"')
        ap.add_argument('--out', required=True, type=Path, metavar='DIR', help='output directory')
    else:
        ap.add_argument('--port', type=int, default=8775, help='port on 127.0.0.1 (default 8775)')
        ap.add_argument('--exit-on-failure', action='store_true',
                        help='exit with code 2 as soon as the engine fails (models do not load, a GPU worker dies, '
                             'out of memory after the back-off; running jobs are marked failed first) instead of '
                             'staying up and answering 503')
    ap.add_argument('--threshold', type=float, default=None, metavar='T', help=THRESHOLD_HELP)
    if form == 'single':
        ap.add_argument('--masks', action='store_true',
                        help='also write <id>.masks.npz (the masks of the instances behind the lines)')
    else:
        ap.add_argument('--masks', action='store_true',
                        help='also write <id>.masks.npz and <id>.instances.npz (masks, boxes, scores and labels of '
                             'the returned instances)%s' % (' for every job' if form == 'serve' else ''))
    if form != 'serve':
        ap.add_argument('--force', action='store_true', help='recompute images whose outputs exist')
    ap.add_argument('--cpu', action='store_true',
                    help='run on the CPU (default: the GPU if PyTorch sees one, else the CPU)')
    ap.add_argument('--settings', type=Path, metavar='FILE.toml',
                    help='advanced settings (model config, workers, threads, ...): see %s' % EXAMPLE_SETTINGS)
    ap.add_argument('--ckpt', type=Path, default=None, help=H)


def _hidden_model_args(ap, engine=True):
    """Deprecated flags of v0.2.0, still accepted (applied, with a warning); not in --help."""
    ap.add_argument('--config', type=str, default=None, help=H)
    ap.add_argument('--device', default=None, help=H)
    ap.add_argument('--msda', choices=('auto', 'compiled', 'pytorch'), default=None, help=H)
    ap.add_argument('--all-queries', action='store_true', default=None, help=H)
    ap.add_argument('--kept-only', action='store_true', default=None, help=H)
    ap.add_argument('--kept-thr', type=float, default=None, help=H)
    if engine:
        ap.add_argument('--input-size', default=None, help=H)
        ap.add_argument('--tile', type=int, default=None, help=H)
        ap.add_argument('--tile-overlap', type=int, default=None, help=H)
        ap.add_argument('--gpu-workers', type=int, default=None, help=H)
        ap.add_argument('--gpu-mem-budget', default=None, help=H)
        ap.add_argument('--pre-workers', type=int, default=None, help=H)
        ap.add_argument('--post-workers', type=int, default=None, help=H)
        ap.add_argument('--pre-threads', type=int, default=None, help=H)
        ap.add_argument('--post-threads', type=int, default=None, help=H)
        ap.add_argument('--gpu-threads', type=int, default=None, help=H)
        ap.add_argument('--max-inflight', type=int, default=None, help=H)


def _engine_args(ap):
    """The model and worker options of the engine forms (also used by tools/benchmark.py): --threshold, --cpu,
    --settings, and the hidden --ckpt and deprecated flags."""
    ap.add_argument('--threshold', type=float, default=None, metavar='T', help=THRESHOLD_HELP)
    ap.add_argument('--cpu', action='store_true', help='run on the CPU')
    ap.add_argument('--settings', type=Path, metavar='FILE.toml', help='advanced settings (%s)' % EXAMPLE_SETTINGS)
    ap.add_argument('--ckpt', type=Path, default=None, help=H)
    _hidden_model_args(ap)


EPILOG = ('Advanced settings (model config, workers, threads, memory budget, ...) go into a TOML file given with '
          '--settings; %s in the fork root lists every key with its default. The checkpoint is found at --ckpt FILE, '
          'else $%s, else <fork root>/%s, else ~/.cache/lineformer/%s.' % (EXAMPLE_SETTINGS, CKPT_ENV, CKPT_NAME,
                                                                           CKPT_NAME))


def _parser(form):
    prog = {'single': 'lineformer', 'batch': 'lineformer batch', 'serve': 'lineformer serve'}[form]
    usage = {'single': '%(prog)s IMAGE... | --list FILE  --out DIR [--threshold T] [--masks] [--force] [--cpu] '
                       '[--settings FILE.toml]',
             'batch': '%(prog)s IMAGE... | --list FILE  --out DIR [--threshold T] [--masks] [--force] [--cpu] '
                      '[--settings FILE.toml]',
             'serve': '%(prog)s [--port 8775] [--exit-on-failure] [--threshold T] [--masks] [--cpu] '
                      '[--settings FILE.toml]'}[form]
    desc = {'single': 'Chart images -> line data series (JSON), one process. Many images: lineformer batch; '
                      'a server: lineformer serve (see --help of each).',
            'batch': 'Run one job of images on the LineFormer engine (pre-processing workers -> GPU workers -> '
                     'post-processing workers; the GPU workers sized from the free device memory). Exit code 0 '
                     'all done, 1 some images failed, 2 engine failure or bad input, 130 interrupted (rerun to '
                     'resume).',
            'serve': 'A long-lived owner of the GPU that takes LineFormer jobs over HTTP on 127.0.0.1 (no '
                     'authentication). API: lineformer_serve.py; client: lineformer_client.py.'}[form]
    epilog = EPILOG
    if form == 'single':
        epilog += ' An image file named "batch" or "serve" needs a path: ./batch'
    if form == 'serve':
        epilog += (' Also accepted: --host ADDR, --drain-timeout S, --ready-file FILE, --verbose '
                   '(lineformer_serve.py).')
    ap = argparse.ArgumentParser(prog=prog, usage=usage, description=desc, epilog=epilog)
    _public_args(ap, form)
    _hidden_model_args(ap, engine=form != 'single')
    if form == 'batch':
        ap.add_argument('--ids', choices=('stem', 'parent_stem'), default=None, help=H)
        ap.add_argument('--instances', action='store_true', default=None, help=H)
        ap.add_argument('--progress-s', type=float, default=None, help=H)
    if form == 'serve':
        ap.add_argument('--host', default='127.0.0.1', help=H)
        ap.add_argument('--drain-timeout', type=float, default=600.0, help=H)
        ap.add_argument('--ready-file', type=Path, default=None, help=H)
        ap.add_argument('--verbose', action='store_true', help=H)
    return ap


def _items(a, ap, ids):
    """Images of the command line -> (items, checked records); two images with one id stop here (before any
    model)."""
    import lineformer_jobs as jobs
    items = [str(p) for p in a.images]
    if a.list:
        items += jobs.read_list_file(a.list)
    if not items:
        ap.error('no images given')
    try:
        recs = jobs.normalize_items(items, ids)
    except jobs.JobError as e:
        ap.error(str(e))
    clash = [r for r in recs if r['status'] == 'failed']
    if clash:
        first = {}
        for r in recs:
            first.setdefault(r['id'].casefold(), r)
        shown = '; '.join('id %r: %s and %s' % (r['id'], first[r['id'].casefold()]['path'], r['path'])
                          for r in clash[:5])
        hint = ('set ids = "parent_stem" in the settings file (--settings) for "<parent dir>__<stem>" ids, or give '
                'explicit ids ("<id><TAB><path>" lines in --list)' if ids == 'stem' else
                'give explicit ids ("<id><TAB><path>" lines in --list)')
        ap.error('%d image(s) map to an id another image already uses (%s%s); %s'
                 % (len(clash), shown, ', ...' if len(clash) > 5 else '', hint))
    return items, recs


# ================================================================== single process (original form)

def main_single(argv=None):
    import numpy as np
    ap = _parser('single')
    a = ap.parse_args(argv)
    try:
        values, _ = effective_settings(a, 'single')
        kept, line = _threshold(a, values)
        device = _device(a)
    except SetupError as e:
        ap.error(str(e))
    recs = [r for r in _items(a, ap, values['ids'])[1] if r['status'] == 'pending']
    try:
        ckpt, ckpt_src = resolve_ckpt(a.ckpt)
    except SetupError as e:
        ap.error(str(e))
    a.out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in recs if a.force or not (a.out / ('%s.json' % r['id'])).exists()]
    if not todo:
        print(f'0 of {len(recs)} images to do', flush=True)
        return 0
    if device == 'auto':
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'{len(todo)} of {len(recs)} images to do on {device}', flush=True)
    _log('device %s%s, %s, checkpoint %s (%s)' % (
        device, ' (auto)' if not a.cpu and getattr(a, 'device', None) is None else '', _thr_text(kept, line), ckpt,
        ckpt_src))
    import infer  # heavy imports after argument checks
    infer.load_model(str(values['config']), ckpt, device, msda=values['msda'],
                     kept_only=False if kept is None else kept)
    if line != DEFAULT_KEPT_THR:  # get_dataseries passes 0.3 to do_instance; the line threshold replaces it here
        do_instance = infer.do_instance
        infer.do_instance = lambda model, img, score_thr=0.3: do_instance(model, img, score_thr=line)

    t0 = time.time()
    ahead = 8
    with ThreadPoolExecutor(4) as pool:
        futs = [pool.submit(_read, r['path']) for r in todo[:ahead]]
        for i, r in enumerate(todo):
            if i + ahead < len(todo):
                futs.append(pool.submit(_read, todo[i + ahead]['path']))
            img = futs[i].result()
            futs[i] = None
            lines, masks = infer.get_dataseries(img, to_clean=False, return_masks=True)
            rec = {'image': r['path'],
                   'lines': [[{'x': float(q['x']), 'y': float(q['y'])} for q in ln] for ln in lines]}
            tmp = a.out / ('%s.json.tmp' % r['id'])
            tmp.write_text(json.dumps(rec))
            tmp.replace(a.out / ('%s.json' % r['id']))
            if a.masks:
                np.savez_compressed(a.out / ('%s.masks.npz' % r['id']), shape=np.array(img.shape[:2]),
                                    masks=np.packbits(np.array([m > 0 for m in masks], dtype=bool)
                                                      .reshape(len(masks), -1), axis=1))
            if (i + 1) % 50 == 0:
                print(i + 1, 'done', round(time.time() - t0), 's', flush=True)
    print('finished', len(todo), round(time.time() - t0), 's', flush=True)
    return 0


# ================================================================== engine forms

def _make_engine(a, ap, form='batch', values=None, sources=None):
    """The engine of batch / serve (also tools/benchmark.py) from the parsed arguments. Bad options -> ap.error."""
    from lineformer_engine import Engine, ModelOptions
    try:
        if values is None:
            values, sources = effective_settings(a, form)
        kept, line = _threshold(a, values)
        device = _device(a)
        ckpt, ckpt_src = resolve_ckpt(getattr(a, 'ckpt', None))
    except SetupError as e:
        ap.error(str(e))
    size = values['input_size']
    if values['tile'] is not None and size == 'config':
        size = 'native'
    mo = ModelOptions(ckpt=ckpt, config=str(values['config']), device=device, msda=values['msda'], kept_thr=kept,
                      line_thr=line, input_size=size, tile=values['tile'], tile_overlap=values['tile_overlap'])
    try:
        mo = mo.resolved()
    except (ValueError, RuntimeError) as e:
        ap.error(str(e))
    pre = values['pre_workers'] if values['pre_workers'] is not None else default_pre_workers()
    settings = dict(values, pre_workers=pre, input_size=mo.input_size, config=mo.config, threshold=line,
                    device=mo.device, checkpoint={'path': mo.ckpt, 'source': ckpt_src},
                    settings_file=str(Path(a.settings).resolve()) if getattr(a, 'settings', None) else None,
                    sources=dict(sources or {}))
    if values['gpu_workers'] is None or values['gpu_mem_budget'] is None:
        for k in ('gpu_workers', 'gpu_mem_budget'):
            if values[k] is None:
                settings['sources'][k] = 'auto'
    if values['gpu_workers'] is not None:
        nw = '%d GPU worker(s)' % values['gpu_workers']
    elif mo.device.startswith('cuda'):
        nw = 'GPU workers sized at start'
    else:
        nw = '1 model worker (CPU)'
    _log('device %s%s, %s, %s, %d pre-processing workers, checkpoint %s (%s)' % (
        mo.device, ' (auto)' if device == 'auto' else '', _thr_text(kept, line), nw, pre, mo.ckpt, ckpt_src))
    if mo.input_size == 'native':
        _log('WARNING: input_size native / tile is ' + EXPERIMENTAL)
    return Engine(mo, gpu_workers=values['gpu_workers'], gpu_mem_budget=values['gpu_mem_budget'], pre_workers=pre,
                  post_workers=values['post_workers'], pre_threads=values['pre_threads'],
                  post_threads=values['post_threads'], gpu_threads=values['gpu_threads'],
                  max_inflight=values['max_inflight'], settings=settings)


def main_batch(argv):
    import lineformer_jobs as jobs
    from lineformer_engine import EngineFailed
    ap = _parser('batch')
    a = ap.parse_args(argv)
    try:
        values, sources = effective_settings(a, 'batch')
    except SetupError as e:
        ap.error(str(e))
    if a.instances:
        _log('DEPRECATED --instances: use --masks (it writes the instances and the masks); the flag still works in '
             'this version')
    items, _ = _items(a, ap, values['ids'])  # bad input fails before the models load
    eng =_make_engine(a, ap, 'batch', values, sources)
    outputs = {'instances': bool(a.masks or a.instances), 'masks': bool(a.masks)}
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
        jid = eng.submit(items, str(a.out), outputs=outputs, force=a.force,
                         ids=values['ids'])  # decides skip-if-done before any model is loaded
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
            s = eng.wait(jid, timeout=values['progress_s'])
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
    ap = _parser('serve')
    a = ap.parse_args(argv)
    try:
        values, sources = effective_settings(a, 'serve')
    except SetupError as e:
        ap.error(str(e))
    eng = _make_engine(a, ap, 'serve', values, sources)
    return lineformer_serve.serve(eng, host=a.host, port=a.port, drain_timeout=a.drain_timeout, verbose=a.verbose,
                                  ready_file=str(a.ready_file) if a.ready_file else None,
                                  exit_on_failure=a.exit_on_failure,
                                  always_outputs=('instances', 'masks') if a.masks else ())


if __name__ == '__main__':
    sys.exit(main())
