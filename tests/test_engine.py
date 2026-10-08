# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""Unit tests of the engine pieces that need no GPU and no checkpoint: ids, skip-if-done, scheduler, job states,
manifest, option parsing, shared-memory transfer, the tiled split path, and the HTTP API + client against a stub
engine.

pytest tests -q      or, without pytest:  python tests/test_engine.py
"""
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lineformer_client as client  # noqa: E402
import lineformer_engine as engine  # noqa: E402
import lineformer_jobs as jobs  # noqa: E402
import tiling  # noqa: E402


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError('%s not raised' % exc.__name__)


# ------------------------------------------------------------------ ids

def test_default_ids_and_checks():
    assert jobs.default_id('/x/a/chart.1.png') == 'chart.1'
    assert jobs.default_id('/x/a/chart.png', 'parent_stem') == 'a__chart'
    _raises(jobs.JobError, jobs.default_id, '/x/a.png', 'hash')
    for bad in ('', '.', '..', '.hidden', 'a/b', 'a\\b', 'a\nb', 'job', 'JOB', 'x' * 201):
        _raises(jobs.JobError, jobs.check_id, bad)
    assert jobs.check_id('job2') == 'job2'


def test_normalize_collisions():
    recs = jobs.normalize_items(['/d1/a.png', '/d2/a.png', '/d1/a.png', '/d1/B.png', '/d3/b.png',
                                 {'id': 'a2', 'path': '/d2/a.png'}])
    st = [(r['id'], r['status']) for r in recs]
    assert st[0] == ('a', 'pending')
    assert st[1] == ('a', 'failed') and 'id collision' in recs[1]['error']
    assert st[2] == ('a', 'duplicate')
    assert st[3] == ('B', 'pending')
    assert st[4] == ('b', 'failed')  # case-insensitive: B.png and b.png would share a file on Windows drives
    assert st[5] == ('a2', 'pending')
    ps = jobs.normalize_items(['/d1/a.png', '/d2/a.png'], 'parent_stem')
    assert [r['status'] for r in ps] == ['pending', 'pending']
    _raises(jobs.JobError, jobs.normalize_items, [])
    _raises(jobs.JobError, jobs.normalize_items, [{'file': 'x'}])
    _raises(jobs.JobError, jobs.normalize_items, ['rel.png'], require_absolute=True)


def test_list_file():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'l.txt'
        p.write_text('# comment\n/a/x.png\n\nid1\t/b/y.png\n', encoding='utf-8')
        assert jobs.read_list_file(p) == ['/a/x.png', {'id': 'id1', 'path': '/b/y.png'}]


# ------------------------------------------------------------------ skip-if-done

def _write_done(out, iid, path, fp, outputs=()):
    obj = {'id': iid, 'image': path, 'lines': [], 'fingerprint': fp, 'outputs': list(outputs)}
    jobs.write_json_atomic(jobs.output_paths(out, iid)['json'], obj)
    for k in outputs:
        jobs.output_paths(out, iid)[k].write_bytes(b'x')


def test_done_state():
    fp = {'engine': '1', 'kept_thr': 0.3}
    with tempfile.TemporaryDirectory() as out:
        rec = lambda iid, path: {'idx': 0, 'id': iid, 'path': path, 'status': 'pending'}  # noqa: E731
        _write_done(out, 'a', '/img/a.png', fp, ['instances'])
        assert jobs.done_state(rec('a', '/img/a.png'), out, fp, {'instances': True})['status'] == 'skipped'
        r = jobs.done_state(rec('a', '/img/a.png'), out, fp, {'masks': True})
        assert r['status'] == 'pending' and 'without masks' in r['note']
        r = jobs.done_state(rec('a', '/img/a.png'), out, fp, {}, force=True)
        assert r['status'] == 'pending'
        r = jobs.done_state(rec('a', '/other/a.png'), out, fp, {}, force=True)
        assert r['status'] == 'failed' and 'another image' in r['error']
        r = jobs.done_state(rec('a', '/img/a.png'), out, dict(fp, kept_thr=None), {})
        assert r['status'] == 'failed' and 'kept_thr' in r['error']
        assert jobs.done_state(rec('new', '/img/n.png'), out, fp, {})['status'] == 'pending'
        (Path(out) / 'bad.json').write_text('{', encoding='utf-8')
        r = jobs.done_state(rec('bad', '/img/bad.png'), out, fp, {})
        assert r['status'] == 'pending' and 'unreadable' in r['note']


# ------------------------------------------------------------------ scheduler and job states

def _job(n, out='/tmp/o', priority=0, jid=None):
    recs = jobs.normalize_items(['/img/%d.png' % i for i in range(n)])
    j = jobs.Job(jid or 'j', out, recs, priority=priority)
    return j


def test_scheduler_fifo_and_priority():
    s = jobs.Scheduler()
    a, b, c = _job(2, '/o/a', jid='a'), _job(2, '/o/b', jid='b'), _job(1, '/o/c', priority=5, jid='c')
    for j in (a, b):
        s.add(j)
    assert [s.next_task()[0].id for _ in range(2)] == ['a', 'a']
    s.add(c)
    order = [s.next_task()[0].id for _ in range(3)]
    assert order == ['c', 'b', 'b'], order
    assert s.next_task() == (None, None)
    _raises(jobs.OutDirBusy, s.check_out_dir, '/o/a')
    s.check_out_dir('/o/new')


def test_job_lifecycle_and_manifest():
    with tempfile.TemporaryDirectory() as out:
        recs = jobs.normalize_items(['/img/0.png', '/img/1.png', '/img/2.png', '/img/0.png'])
        j = jobs.Job('j1', out, recs, outputs={'instances': True})
        _raises(jobs.JobError, jobs.Job, 'x', out, recs, outputs={'pdf': True})
        assert j.counts()['duplicate'] == 1 and len(j.pending) == 3
        r0, r1 = j.next_record(), j.next_record()
        assert j.status == 'running' and j.counts()['running'] == 2
        j.finish_record({'idx': r0['idx'], 'status': 'done', 'timings': {'pre_s': 0.1, 'gpu_s': 0.2, 'post_s': 0.3}})
        _raises(RuntimeError, j.finish_record, {'idx': r0['idx'], 'status': 'done'})  # duplicate result
        assert j.cancel() and j.status == 'cancelling'
        assert j.counts()['cancelled'] == 1
        j.finish_record({'idx': r1['idx'], 'status': 'failed', 'error': 'boom'})
        assert j.status == 'cancelled' and j.event.is_set()
        j.write_manifest({'engine': 'stub'})
        m = json.loads((Path(out) / 'job.json').read_text())
        assert m['status'] == 'cancelled' and m['engine'] == {'engine': 'stub'}
        assert [i['status'] for i in m['images']] == ['done', 'failed', 'cancelled', 'duplicate']
        assert m['counts']['total'] == 4 and m['errors'][0]['error'] == 'boom'
        j2 = _job(2, out)
        r = j2.next_record()
        j2.finish_record({'idx': r['idx'], 'status': 'done'})
        j2.stop('interrupted', 'stopped')
        assert j2.status == 'interrupted' and j2.counts()['pending'] == 1
        j3 = _job(1, out)
        r = j3.next_record()
        j3.finish_record({'idx': r['idx'], 'status': 'failed', 'error': 'x'})
        assert j3.status == 'done_with_errors'
        j4 = jobs.Job('j4', out, [dict(r, status='skipped') for r in jobs.normalize_items(['/img/0.png'])])
        assert j4.maybe_finish() and j4.status == 'done'


def test_manifest_contents():
    import hashlib
    import re
    with tempfile.TemporaryDirectory() as out:
        recs = jobs.normalize_items(['/img/0.png', '/img/1.png'])
        j = jobs.Job('j1', out, recs, outputs={'instances': True})
        r0, r1 = j.next_record(), j.next_record()
        j.finish_record({'idx': r0['idx'], 'status': 'done', 'outputs_sha256': {'json': 'a', 'instances': 'b'}})
        j.finish_record({'idx': r1['idx'], 'status': 'failed', 'error': 'Traceback ...\n' + 'x' * 5000})
        info = {'versions': {'lineformer': '0.2.0', 'lineformer_git': 'abc', 'lineformer_git_dirty': False},
                'model_options': {'device': 'cuda:0'}}
        j.write_manifest(info)
        m = json.loads((Path(out) / 'job.json').read_text())
        assert m['manifest_version'] == jobs.MANIFEST_VERSION == 2
        assert m['fork'] == {'version': '0.2.0', 'git_commit': 'abc', 'git_dirty': False}
        assert m['engine'] == info and m['status'] == 'done_with_errors'
        iso = re.compile(r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$')
        assert all(iso.match(m['times_utc'][k]) for k in ('created', 'started', 'finished', 'written'))
        assert m['images'][0]['outputs_sha256'] == {'json': 'a', 'instances': 'b'}
        assert m['failures'] == [{'idx': 1, 'id': '1', 'path': r1['path'], 'error': 'Traceback ...\n' + 'x' * 5000}]
        assert len(m['errors'][0]['error']) < 5000  # the summary's short form
        # atomic: an object that cannot be written leaves the old manifest and no temp file
        _raises(TypeError, j.write_manifest, {'versions': {}, 'bad': object()})
        assert json.loads((Path(out) / 'job.json').read_text()) == m
        assert sorted(os.listdir(out)) == ['job.json']
        # skip-if-done records the sha256 of the existing <id>.json
        _write_done(out, 'a', '/img/a.png', {'engine': '1'})
        rec = jobs.done_state({'idx': 0, 'id': 'a', 'path': '/img/a.png', 'status': 'pending'}, out,
                              {'engine': '1'}, {})
        want = hashlib.sha256((Path(out) / 'a.json').read_bytes()).hexdigest()
        assert rec['status'] == 'skipped' and rec['outputs_sha256'] == {'json': want}
        (Path(out) / 'a.instances.npz').write_bytes(b'npz')
        assert jobs.outputs_sha256(out, 'a', ['instances']) == {
            'json': want, 'instances': hashlib.sha256(b'npz').hexdigest()}


def test_versions_and_git_state():
    import subprocess
    v = engine.versions()
    for k in engine.PACKAGES + ('python', 'platform', 'engine', 'lineformer_git', 'lineformer_git_dirty'):
        assert k in v, k
    head = subprocess.run(['git', '-C', str(engine.HERE), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    if head.returncode == 0:  # running from a checkout
        assert v['lineformer_git'] == head.stdout.strip() and v['lineformer_git_dirty'] in (True, False)
    with tempfile.TemporaryDirectory() as d:
        assert engine.git_state(d) == {'commit': None, 'dirty': None}


def test_engine_job_manifest_without_workers():
    """A job submitted to the engine (as batch and serve do) writes its manifest at once; a job whose images are
    all done ends 'done' without any worker started, its manifest holding the fork and the model options."""
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / 'm.pth'
        ck.write_bytes(b'0')
        out = Path(d) / 'out'
        eng = engine.Engine(engine.ModelOptions(ckpt=str(ck), device='cpu', kept_thr=0.3), log=lambda *a: None)
        eng.prepare()
        out.mkdir()
        _write_done(out, 'a', str(Path(d) / 'a.png'), eng.fingerprint)
        jid = eng.submit([str(Path(d) / 'a.png'), str(Path(d) / 'b.png')], str(out))
        m = json.loads((out / 'job.json').read_text())
        assert m['job'] == jid and m['status'] == 'queued' and m['counts']['skipped'] == 1
        assert m['engine']['model_options']['kept_thr'] == 0.3 and m['engine']['device'] == 'cpu'
        assert m['engine']['order'] == 'geometric' and m['engine']['gpu_workers'] == 1
        assert m['fork']['git_commit'] == eng.versions['lineformer_git']
        assert m['images'][0]['outputs_sha256']['json']
        _raises(jobs.OutDirBusy, eng.submit, [str(Path(d) / 'a.png')], str(out))  # the first job is active
        out2 = Path(d) / 'out2'
        out2.mkdir()
        _write_done(out2, 'a', str(Path(d) / 'a.png'), eng.fingerprint)
        jid2 = eng.submit([str(Path(d) / 'a.png')], str(out2))
        m2 = json.loads((out2 / 'job.json').read_text())
        assert m2['job'] == jid2 and m2['status'] == 'done' and m2['times_utc']['finished']
        assert eng.state == 'prepared'  # no worker was started


# ------------------------------------------------------------------ options

def test_option_parsing():
    assert engine.parse_input_size('config') == 'config'
    assert engine.parse_input_size('NATIVE') == 'native'
    assert engine.parse_input_size('1024') == 1024 and engine.parse_input_size(768) == 768
    for bad in ('0', '-5', 'big', True, 0, 1.5):
        _raises(ValueError, engine.parse_input_size, bad)
    assert engine.parse_mem_budget('0.85') == ('frac', 0.85)
    assert engine.parse_mem_budget(0.5) == ('frac', 0.5)
    assert engine.parse_mem_budget('4G') == ('bytes', 4 * 2 ** 30)
    assert engine.parse_mem_budget('512MB') == ('bytes', 512 * 2 ** 20)
    assert engine.parse_mem_budget('2GiB') == ('bytes', 2 * 2 ** 30)
    for bad in ('0', '1.5', '-1G', 'x'):
        _raises(ValueError, engine.parse_mem_budget, bad)


def test_model_options():
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / 'm.pth'
        ck.write_bytes(b'0')
        mo = engine.ModelOptions(ckpt=str(ck)).resolved()
        assert mo.kept_thr is None and mo.input_size == 'config' and mo.tile is None
        assert engine.ModelOptions(ckpt=str(ck), kept_thr=0.1).resolved().kept_thr == 0.1
        for bad in (0.0, 1.0, True, -0.2):
            _raises(ValueError, engine.ModelOptions(ckpt=str(ck), kept_thr=bad).resolved)
        t = engine.ModelOptions(ckpt=str(ck), tile=512, tile_overlap=128).resolved()
        assert t.input_size == 'native' and t.tile_score_thr == 0.3
        assert engine.ModelOptions(ckpt=str(ck), tile=512, kept_thr=0.1).resolved().tile_score_thr == 0.1
        _raises(ValueError, engine.ModelOptions(ckpt=str(ck), tile=512, input_size=1024).resolved)
        _raises(ValueError, engine.ModelOptions(ckpt=str(ck), tile=512, tile_overlap=512).resolved)
        _raises(ValueError, engine.ModelOptions(ckpt=str(Path(d) / 'missing.pth')).resolved)
        fa = engine.fingerprint(mo, 'c', 'f')
        fb = engine.fingerprint(t, 'c', 'f')
        assert jobs.fingerprint_diff(fa, fb) == ['input_size', 'tile']


# ------------------------------------------------------------------ shared-memory transfer

def test_pack_result_roundtrip():
    rng = np.random.default_rng(0)
    boxes = np.array([[0, 0, 5, 5, 0.9], [1, 1, 4, 4, 0.2], [0, 0, 2, 2, 0.31]], np.float32)
    masks = [rng.random((7, 9)) > 0.5 for _ in range(3)]
    for transfer, want in (('all', [0, 1, 2]), ('kept', [0, 2])):
        b, m = engine.unpack_result(engine.pack_result(([boxes], [masks]), transfer))
        assert np.array_equal(b[0], boxes)
        for i in range(3):
            if i in want:
                assert np.array_equal(m[0][i], masks[i])
            else:
                assert m[0][i] is None
    b, m = engine.unpack_result(engine.pack_result(([np.zeros((0, 5), np.float32)], [[]]), 'all'))
    assert m == [[]]


def test_pack_tiles_roundtrip():
    rng = np.random.default_rng(1)
    tiles = [((0, 0, 4, 5), np.array([0.5, 0.7]), rng.random((2, 4, 5)) > 0.5),
             ((0, 3, 4, 8), np.zeros(0), np.zeros((0, 4, 5), bool)),
             ((2, 0, 6, 5), np.array([0.4]), rng.random((1, 4, 5)) > 0.5)]
    got = engine.unpack_tiles(engine.pack_tiles(tiles))
    for (b, s, m), (b2, s2, m2) in zip(tiles, got):
        assert tuple(b) == b2 and np.array_equal(s, s2) and np.array_equal(m, m2) and m2.dtype == bool


def test_tiled_split_equals_run_tiled():
    """The engine's split (crop -> per-crop selection -> merge in another process) equals tiling.run_tiled."""
    H, W = 300, 420
    img = np.zeros((H, W, 3), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    lines = [(np.abs(yy - (0.3 * xx + 40)) < 2), (np.abs(yy - (250 - 0.4 * xx)) < 2)]

    def fake(model, crop_box):
        y0, x0, y1, x1 = crop_box
        ms, bs = [], []
        for k, L in enumerate(lines):
            m = L[y0:y1, x0:x1]
            if not m.any():
                continue
            ms.append(m)
            bs.append([0, 0, 1, 1, 0.9 - 0.25 * k])
        ms.append(np.zeros((y1 - y0, x1 - x0), bool))
        bs.append([0, 0, 0, 0, 0.1])
        return ([np.array(bs, np.float32)], [ms])

    boxes_seen = []

    def infer_fn(model, crop):
        box = boxes_seen.pop(0)
        return fake(model, box)

    size, ov, thr = 160, 40, 0.3
    boxes = tiling.crop_boxes(H, W, size, ov)
    boxes_seen[:] = list(boxes)
    (rb,), (rm,) = tiling.run_tiled(None, img, size, ov, score_thr=thr, infer_fn=infer_fn)
    tiles = []
    for box in boxes:
        s, m = engine.tile_crop_result(fake(None, box), box[2] - box[0], box[3] - box[1], thr)
        tiles.append((box, s, m))
    tiles = engine.unpack_tiles(engine.pack_tiles(tiles))
    eb, em, _ = tiling.merge_instances(tiles, H, W, thr, 0.5, 20)
    assert np.array_equal(rb, eb) and len(rm) == len(em) == 2
    assert all(np.array_equal(a, b) for a, b in zip(rm, em))


def test_instance_order():
    # instances in model order: a line starting at x 20 (mean y 80), a low-score instance, a line starting at x 20
    # (mean y 30), a line starting at x 5; scores differ from the geometric order on purpose
    boxes = np.array([[20, 78, 180, 82, 0.95], [0, 0, 10, 10, 0.2], [20, 28, 180, 32, 0.6],
                      [5, 50, 100, 60, 0.31]], np.float32)
    labels = np.zeros(4, np.int64)
    line = lambda x0, x1, y: [{'x': x, 'y': y} for x in range(x0, x1)]  # noqa: E731
    lines = [line(20, 181, 80), line(20, 181, 30), line(5, 101, 55)]
    perm, lperm = engine.instance_order(boxes, labels, lines)
    assert perm == [3, 2, 0, 1] and lperm == [2, 1, 0]
    # the same instances in any other model order give the same output order
    rng = np.random.default_rng(0)
    for _ in range(10):
        p = rng.permutation(4)
        lines_p = [lines[[0, None, 1, 2][i]] for i in p if i != 1]
        perm_p, _ = engine.instance_order(boxes[p], labels[p], lines_p)
        assert p[perm_p].tolist() == perm
    # exact geometric ties: the score decides (higher first), not the model order
    tied = np.array([[0, 0, 9, 9, 0.5], [0, 0, 9, 9, 0.7]], np.float32)
    assert engine.instance_order(tied, np.zeros(2, np.int64), [line(0, 9, 4), line(0, 9, 4)])[0] == [1, 0]
    assert engine.instance_order(tied, np.zeros(2, np.int64), [], line_thr=0.9)[0] == [1, 0]
    # an empty line sorts last among the lines; a line count that does not fit the instances fails loud
    assert engine.instance_order(tied, np.zeros(2, np.int64), [[], line(3, 9, 4)])[0] == [1, 0]
    _raises(RuntimeError, engine.instance_order, boxes, labels, lines[:2])
    assert engine.instance_order(np.zeros((0, 5), np.float32), np.zeros(0, np.int64), []) == ([], [])


def test_split_result():
    b = np.array([[0, 0, 1, 1, 0.5]], np.float32)
    boxes, labels, masks = engine.split_result(([b], [[None]]))
    assert boxes.shape == (1, 5) and labels.tolist() == [0] and masks == [None]


# ------------------------------------------------------------------ HTTP API + client against a stub engine

class StubEngine:
    """Implements the engine methods the server calls; jobs finish when finish() is called."""

    def __init__(self):
        self.engine_info = {'model_options': {'kept_thr': 0.3, 'input_size': 'config', 'device': 'cuda:0'}}
        self.sched = jobs.Scheduler()
        self.lock = threading.Lock()
        self.state = 'running'
        self.n = 0

    def submit(self, images, out, outputs=None, force=False, priority=0, ids='stem', name=None,
               require_absolute=False):
        if not os.path.isabs(out):
            raise jobs.JobError('out must be absolute')
        recs = jobs.normalize_items(images, ids, require_absolute=require_absolute)
        with self.lock:
            self.sched.check_out_dir(out)
            self.n += 1
            j = jobs.Job('s%d' % self.n, out, recs, outputs=outputs, force=force, priority=priority, name=name)
            self.sched.add(j)
            return j.id

    def status(self, job_id, images=False):
        j = self.sched.jobs[job_id]
        s = j.summary()
        if images:
            s['images'] = [jobs._public(r) for r in j.records]
        return s

    def list_jobs(self):
        return [{'job': j.id, 'status': j.status} for j in self.sched.jobs.values()]

    def cancel(self, job_id):
        j = self.sched.jobs[job_id]
        return j.cancel(), j.summary()

    def health(self):
        return {'state': self.state, 'engine': self.engine_info}

    def finish(self, job_id, fail=False):
        j = self.sched.jobs[job_id]
        while True:
            _, r = self.sched.next_task()
            if r is None:
                break
            j.finish_record({'idx': r['idx'], 'status': 'failed' if fail else 'done', 'error': 'x' if fail else None})


def _stub_server():
    import lineformer_serve
    eng = StubEngine()
    srv = lineformer_serve.Server(('127.0.0.1', 0), eng)
    t = threading.Thread(target=srv.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
    t.start()
    return eng, srv, client.LineFormerClient('http://127.0.0.1:%d' % srv.server_address[1], timeout=5)


def test_client_against_stub_server():
    eng, srv, lf = _stub_server()
    try:
        assert lf.health()['state'] == 'running'
        jid = lf.submit(['/img/a.png', {'id': 'b', 'path': '/img/b.png'}], '/o/1', instances=True,
                        require={'kept_thr': 0.3})
        s = lf.status(jid, images=True)
        assert s['status'] == 'queued' and s['counts']['pending'] == 2 and s['outputs']['instances']
        assert [i['id'] for i in s['images']] == ['a', 'b']
        e = _raises(client.LineFormerError, lf.submit, ['/img/c.png'], '/o/1')
        assert e.status == 409 and 'in use' in str(e)
        e = _raises(client.LineFormerError, lf.submit, ['/img/c.png'], '/o/2', require={'kept_thr': 0.1})
        assert e.status == 409 and 'kept_thr' in str(e)
        e = _raises(client.LineFormerError, lf._call, 'POST', '/jobs', {'images': ['rel.png'], 'out': '/o/3'})
        assert e.status == 400 and 'absolute' in str(e)
        e = _raises(client.LineFormerError, lf._call, 'POST', '/jobs', {'images': [], 'out': '/o/3', 'x': 1})
        assert e.status == 400 and 'unknown keys' in str(e)
        e = _raises(client.LineFormerError, lf.status, 'nope')
        assert e.status == 404
        e = _raises(client.LineFormerError, lf._call, 'GET', '/what')
        assert e.status == 404
        threading.Timer(0.3, eng.finish, (jid,)).start()
        final = lf.wait(jid, poll=0.05, timeout=5)
        assert final['status'] == 'done' and final['counts']['done'] == 2
        j2 = lf.submit(['/img/d.png'], '/o/1')  # the directory is free again
        eng.finish(j2, fail=True)
        e = _raises(client.JobFailed, lf.wait, j2, poll=0.05)
        assert e.summary['status'] == 'done_with_errors'
        assert lf.wait(j2, accept=client.FINAL)['counts']['failed'] == 1
        j3 = lf.submit(['/img/e.png', '/img/f.png'], '/o/4')
        r = lf.cancel(j3)
        assert r['cancelled'] and r['job']['status'] == 'cancelled'
        assert {j['job'] for j in lf.jobs()} == {jid, j2, j3}
        _raises(TimeoutError, lf.run, ['/img/g.png'], '/o/5', timeout=0.2, poll=0.05)  # stub never finishes it
    finally:
        srv.shutdown()
        srv.server_close()
    _raises(client.LineFormerError, lf.health)  # server gone: a clear error, not a hang


if __name__ == '__main__':
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            t = time.time()
            fn()
            n += 1
            print('ok', name, '%.2fs' % (time.time() - t))
    print('%d tests passed' % n)
