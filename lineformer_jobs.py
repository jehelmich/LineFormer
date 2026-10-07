"""Jobs of the LineFormer engine: image ids, output files, skip-if-done, the scheduler and the job manifest.

No torch / mmcv / numpy imports: the engine's main process, the HTTP server and the unit tests use this module.

A job is a list of images, an output directory and output options. Model options (device, input size,
kept-queries threshold, tiling) belong to the engine, not to the job.

Image ids (the output file names):
  * an item is a path, or {"id": ..., "path": ...} (list files: "<path>" or "<id>\\t<path>" per line);
  * without an explicit id the scheme names it: "stem" (default, the file name without its extension) or
    "parent_stem" ("<parent dir name>__<stem>", the equivalence harness's ids);
  * ids are compared case-insensitively (an output directory on a Windows drive is case-insensitive);
  * a repeated id in one job: the same image listed twice -> the repeat gets status "duplicate" (no work, not an
    error); a different image -> the repeat FAILS ("id collision"), the first one is processed. Give explicit
    ids or use "parent_stem" to keep both;
  * an id must be a usable file name: no "/" "\\" or control characters, not "." / "..", no leading ".", at most
    200 characters, not "job" (the manifest is job.json).

Outputs per image in the output directory (each written atomically: temp file + rename, the JSON last):
  <id>.json            {"id", "image", "image_sha256", "shape", "lines": [[{"x", "y"}, ...], ...], "n_lines",
                       "n_instances", "outputs", "fingerprint", "job", "timings"}; lines as infer.get_dataseries
                       (instances with score > 0.3); line order is not meaningful
  <id>.instances.npz   (option "instances") boxes (N, 5) float32 x1 y1 x2 y2 score, labels (N,) int64: every
                       instance the model returned (all 100 queries, or only the kept ones in kept-queries mode)
  <id>.masks.npz       (option "masks") masks_packed = np.packbits(masks.reshape(-1)), mask_shape = (N, H, W):
                       the masks of the same N instances, in the same order
  job.json             the manifest of the last job that wrote into the directory

Skip-if-done (unless force): an image is skipped when <id>.json exists, names the same image path, has the same
fingerprint (engine version, checkpoint and config sha256, input size, kept threshold, tiling) and the requested
optional outputs exist. The same id from ANOTHER image path fails the image, with or without force (outputs are
never overwritten by a different image); a different fingerprint fails the image unless force.

Image states: pending -> running -> done | failed; skipped, duplicate (decided at submit); cancelled (cancel);
pending/running left when the engine stops are reported as such in an "interrupted" or "failed" job.
Job states: queued, running, cancelling, done, done_with_errors, cancelled, interrupted, failed.
Python 3.8 compatible.
"""
from __future__ import annotations

import collections
import json
import os
import threading
import time
from pathlib import Path

ENGINE_VERSION = '1'
MANIFEST = 'job.json'
ID_SCHEMES = ('stem', 'parent_stem')
OUTPUT_KINDS = ('instances', 'masks')
FINAL_JOB_STATES = ('done', 'done_with_errors', 'cancelled', 'interrupted', 'failed')
FINAL_IMAGE_STATES = ('done', 'failed', 'skipped', 'duplicate', 'cancelled')
MAX_ID_LEN = 200


class JobError(ValueError):
    """A job request that cannot be accepted (bad input)."""


class OutDirBusy(JobError):
    """Another active job writes into the same output directory."""


# ------------------------------------------------------------------ ids and paths

def default_id(path, scheme='stem'):
    p = Path(path)
    if scheme == 'stem':
        return p.stem
    if scheme == 'parent_stem':
        return '%s__%s' % (p.parent.name, p.stem)
    raise JobError('unknown id scheme %r (known: %s)' % (scheme, ', '.join(ID_SCHEMES)))


def check_id(iid):
    """Raise JobError unless iid is usable as an output file name."""
    if not isinstance(iid, str) or not iid:
        raise JobError('image id must be a non-empty string, got %r' % (iid,))
    if len(iid) > MAX_ID_LEN:
        raise JobError('image id longer than %d characters: %r' % (MAX_ID_LEN, iid[:50] + '...'))
    if iid in ('.', '..') or iid.startswith('.'):
        raise JobError('image id must not start with ".": %r' % iid)
    bad = [c for c in iid if c in '/\\' or ord(c) < 32 or ord(c) == 127]
    if bad:
        raise JobError('image id %r contains %r (not allowed in a file name)' % (iid, bad[0]))
    if iid.casefold() == Path(MANIFEST).stem:
        raise JobError('image id %r is reserved (the manifest is %s)' % (iid, MANIFEST))
    return iid


def output_paths(out, iid):
    out = Path(out)
    return {'json': out / (iid + '.json'), 'instances': out / (iid + '.instances.npz'),
            'masks': out / (iid + '.masks.npz')}


def read_list_file(path):
    """Image list file -> items: "<path>" or "<id>\\t<path>" per line; blank lines and "#" comments skipped."""
    items = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        if '\t' in s:
            iid, p = s.split('\t', 1)
            items.append({'id': iid.strip(), 'path': p.strip()})
        else:
            items.append(s)
    return items


def _resolve(path):
    return str(Path(path).expanduser().resolve())


def normalize_items(images, scheme='stem', require_absolute=False):
    """images: list of paths or {"id", "path"} dicts -> list of image records (dicts with idx, id, path, status).

    Id collisions are decided here (see module docstring). Raises JobError on malformed input."""
    if scheme not in ID_SCHEMES:
        raise JobError('unknown id scheme %r (known: %s)' % (scheme, ', '.join(ID_SCHEMES)))
    if not isinstance(images, (list, tuple)) or not images:
        raise JobError('images must be a non-empty list')
    recs, first = [], {}
    for idx, item in enumerate(images):
        if isinstance(item, dict):
            extra = set(item) - {'id', 'path'}
            if extra or 'path' not in item:
                raise JobError('image %d: expected {"path": ..., "id": ...(optional)}, got keys %s'
                               % (idx, sorted(item)))
            path, iid = item['path'], item.get('id')
        else:
            path, iid = item, None
        if isinstance(path, Path):
            path = str(path)
        if not isinstance(path, str) or not path:
            raise JobError('image %d: path must be a non-empty string, got %r' % (idx, path))
        if require_absolute and not os.path.isabs(path):
            raise JobError('image %d: path must be absolute (the server resolves no relative paths): %r'
                           % (idx, path))
        path = _resolve(path)
        iid = check_id(default_id(path, scheme) if iid is None else iid)
        rec = {'idx': idx, 'id': iid, 'path': path, 'status': 'pending'}
        key = iid.casefold()
        if key in first:
            other = recs[first[key]]
            if other['path'] == path:
                rec['status'] = 'duplicate'
                rec['note'] = 'same image as index %d' % other['idx']
            else:
                rec['status'] = 'failed'
                rec['error'] = ('id collision: id %r is also used by %s (index %d); give explicit ids '
                                '("<id>\\t<path>" lines or {"id", "path"}) or the parent_stem id scheme'
                                % (iid, other['path'], other['idx']))
        else:
            first[key] = idx
        recs.append(rec)
    return recs


# ------------------------------------------------------------------ atomic writes and skip-if-done

def write_json_atomic(path, obj, indent=None):
    path = str(path)
    tmp = '%s.tmp%d' % (path, os.getpid())
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def fingerprint_diff(a, b):
    keys = sorted(set(a or {}) | set(b or {}))
    return [k for k in keys if (a or {}).get(k) != (b or {}).get(k)]


def done_state(rec, out, fingerprint, outputs, force=False):
    """Decide skip-if-done for one pending record; updates rec (status / error / note) and returns it."""
    p = output_paths(out, rec['id'])
    if not p['json'].exists():
        return rec
    try:
        with open(str(p['json']), encoding='utf-8') as f:
            old = json.load(f)
    except Exception as e:  # a JSON is written atomically, so this is not a half-written output
        rec['note'] = 'existing %s unreadable (%s): recomputed' % (p['json'].name, e)
        return rec
    if old.get('image') != rec['path']:
        rec['status'] = 'failed'
        rec['error'] = ('output %s exists for another image (%s); use another output directory or an explicit id'
                        % (p['json'], old.get('image')))
        return rec
    if force:
        rec['note'] = 'recomputed (force)'
        return rec
    diff = fingerprint_diff(old.get('fingerprint'), fingerprint)
    if diff:
        rec['status'] = 'failed'
        rec['error'] = ('output %s was made with other model options (%s differ); force recomputes it, or use '
                        'another output directory' % (p['json'], ', '.join(diff)))
        return rec
    missing = [k for k in OUTPUT_KINDS if outputs.get(k) and not (k in old.get('outputs', []) and p[k].exists())]
    if missing:
        rec['note'] = 'done earlier without %s: recomputed' % ', '.join(missing)
        return rec
    rec['status'] = 'skipped'
    return rec


# ------------------------------------------------------------------ jobs

def _stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return {'n': n, 'median': xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2]),
            'mean': sum(xs) / n, 'p90': xs[min(n - 1, int(round(0.9 * (n - 1))))], 'max': xs[-1]}


class Job:
    """One job: its image records and state. Not thread-safe by itself (the engine holds a lock)."""

    def __init__(self, job_id, out, records, outputs=None, force=False, priority=0, name=None, seq=0):
        self.id = job_id
        self.out = _resolve(out)
        self.records = records
        self.outputs = {k: bool((outputs or {}).get(k)) for k in OUTPUT_KINDS}
        unknown = set(outputs or {}) - set(OUTPUT_KINDS)
        if unknown:
            raise JobError('unknown outputs %s (known: %s)' % (sorted(unknown), ', '.join(OUTPUT_KINDS)))
        self.force = bool(force)
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise JobError('priority must be an integer, got %r' % (priority,))
        self.priority = priority
        self.name = name
        self.seq = seq
        self.status = 'queued'
        self.error = None
        self.created = time.time()
        self.started = None
        self.finished = None
        self.pending = collections.deque(r['idx'] for r in records if r['status'] == 'pending')
        self.running = set()
        self.event = threading.Event()
        self.dirty = True

    # --- scheduling
    def next_record(self):
        if self.status not in ('queued', 'running') or not self.pending:
            return None
        rec = self.records[self.pending.popleft()]
        rec['status'] = 'running'
        rec['t_fed'] = time.time()
        self.running.add(rec['idx'])
        if self.started is None:
            self.started = rec['t_fed']
        self.status = 'running'
        self.dirty = True
        return rec

    def finish_record(self, result):
        """result: the record coming back from the post worker (dict with idx, status, ...)."""
        idx = result['idx']
        if idx not in self.running:
            raise RuntimeError('job %s: image %d came back but was not running (duplicate result?)' % (self.id, idx))
        self.running.discard(idx)
        rec = self.records[idx]
        rec.update(result)
        rec['t_done'] = time.time()
        if rec['status'] not in ('done', 'failed'):
            raise RuntimeError('job %s image %d: unexpected status %r from a worker' % (self.id, idx, rec['status']))
        self.dirty = True
        self.maybe_finish()

    def cancel(self):
        if self.status in FINAL_JOB_STATES:
            return False
        while self.pending:
            self.records[self.pending.popleft()]['status'] = 'cancelled'
        self.status = 'cancelling'
        self.dirty = True
        self.maybe_finish()
        return True

    def maybe_finish(self):
        if self.status in FINAL_JOB_STATES or self.pending or self.running:
            return False
        if self.status == 'cancelling':
            self.status = 'cancelled'
        else:
            self.status = 'done_with_errors' if self.counts()['failed'] else 'done'
        self.finished = time.time()
        self.dirty = True
        self.event.set()
        return True

    def stop(self, status, error=None):
        """Engine stops (interrupted) or fails: the job ends where it is; pending/running records stay so."""
        if self.status in FINAL_JOB_STATES:
            return
        for idx in self.running:
            r = self.records[idx]
            r['note'] = 'in flight when the engine %s' % ('failed' if status == 'failed' else 'stopped')
        self.status = status
        self.error = error
        self.finished = time.time()
        self.dirty = True
        self.event.set()

    @property
    def final(self):
        return self.status in FINAL_JOB_STATES

    # --- reporting
    def counts(self):
        c = collections.Counter(r['status'] for r in self.records)
        out = {k: c.get(k, 0) for k in ('pending', 'running', 'done', 'failed', 'skipped', 'duplicate',
                                         'cancelled')}
        out['total'] = len(self.records)
        return out

    def timing(self):
        done = [r for r in self.records if r['status'] == 'done']
        t = {'created': self.created, 'started': self.started, 'finished': self.finished}
        if self.started is not None:
            end = self.finished or time.time()
            t['wall_s'] = end - self.started
            t['images_per_s'] = len(done) / t['wall_s'] if t['wall_s'] > 0 and done else None
        tim = [r.get('timings') or {} for r in done]
        t['latency_s'] = _stats([r['t_done'] - r['t_fed'] for r in done if r.get('t_fed') and r.get('t_done')])
        for k in ('pre_s', 'gpu_s', 'post_s'):
            t[k] = _stats([x.get(k) for x in tim])
        return t

    def summary(self):
        return {'job': self.id, 'name': self.name, 'status': self.status, 'error': self.error, 'out': self.out,
                'outputs': self.outputs, 'force': self.force, 'priority': self.priority, 'counts': self.counts(),
                'timing': self.timing(),
                'errors': [{'idx': r['idx'], 'id': r['id'], 'path': r['path'], 'error': _short(r.get('error'))}
                           for r in self.records if r['status'] == 'failed'][:20]}

    def manifest(self, engine_info):
        m = self.summary()
        m['engine'] = engine_info
        m['images'] = [_public(r) for r in self.records]
        return m

    def write_manifest(self, engine_info):
        Path(self.out).mkdir(parents=True, exist_ok=True)
        write_json_atomic(Path(self.out) / MANIFEST, self.manifest(engine_info), indent=1)
        self.dirty = False


_PUBLIC = ('idx', 'id', 'path', 'status', 'error', 'note', 'image_sha256', 'shape', 'n_lines', 'n_instances',
           'gpu_worker', 'timings', 't_fed', 't_done')


def _public(r):
    return {k: r[k] for k in _PUBLIC if k in r}


def _short(err, n=2000):
    if not err:
        return err
    return err if len(err) <= n else err[:300] + ' ... ' + err[-(n - 300):]


class Scheduler:
    """Jobs in submission order; the next image comes from the highest-priority job, FIFO among equal priority."""

    def __init__(self):
        self.jobs = collections.OrderedDict()
        self._seq = 0

    def add(self, job):
        if job.id in self.jobs:
            raise JobError('job id %s exists' % job.id)
        self._seq += 1
        job.seq = self._seq
        self.jobs[job.id] = job

    def active(self):
        return [j for j in self.jobs.values() if not j.final]

    def check_out_dir(self, out):
        out = _resolve(out)
        for j in self.active():
            if j.out == out or os.path.normcase(j.out) == os.path.normcase(out):
                raise OutDirBusy('output directory %s is in use by active job %s' % (out, j.id))

    def next_task(self):
        cands = [j for j in self.jobs.values() if j.status in ('queued', 'running') and j.pending]
        if not cands:
            return None, None
        job = min(cands, key=lambda j: (-j.priority, j.seq))
        return job, job.next_record()

    def n_pending(self):
        return sum(len(j.pending) for j in self.jobs.values())

    def n_running(self):
        return sum(len(j.running) for j in self.jobs.values())


def new_job_id(seq):
    return '%s-%04d' % (time.strftime('%Y%m%d-%H%M%S'), seq)
