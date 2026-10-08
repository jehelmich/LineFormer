# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""Automatic sizing of the GPU workers (arithmetic with mocked free memory; no GPU) and the bounded back-off on
out of device memory (fault injection through Engine(_test_hooks=tests/oom_hooks.FakeGPU) in real worker
processes on the CPU, no checkpoint; these need the model stack for the pre- and post-processing workers and are
skipped without it).

pytest tests/test_autosize_backoff.py -q      or      python tests/run_all.py
"""
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))  # oom_hooks, importable by the spawned workers too

import lineformer_engine as engine  # noqa: E402

GiB, MiB = 2 ** 30, 2 ** 20
_MISSING = [m for m in ('torch', 'mmcv', 'mmdet', 'cv2', 'skimage') if importlib.util.find_spec(m) is None]


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError('%s not raised' % exc.__name__)


# ------------------------------------------------------------------ automatic sizing: the rule

def test_plan_fits_two_one_zero():
    need = engine.auto_need_mb(0.3)
    assert need == engine.AUTO_SAFETY * (engine.AUTO_KEPT_PEAK_MB + engine.AUTO_CONTEXT_MB)
    total = 24 * GiB
    head = 0.1 * total  # > 2 GiB
    p = engine.plan_gpu_workers(20 * GiB, total, 0.3)
    assert p['gpu_workers'] == 2 and p['headroom_MB'] == round(head / MiB)
    assert p['budget'] == ('bytes', int(20 * GiB - head)) and p['usable_MB'] == round((20 * GiB - head) / MiB)
    p = engine.plan_gpu_workers(int(head + 1.5 * need * MiB), total, 0.3)
    assert p['gpu_workers'] == 1
    e = _raises(engine.InsufficientMemory, engine.plan_gpu_workers, int(head + 0.5 * need * MiB), total, 0.3)
    for s in ('device free', 'headroom', 'need per GPU worker', '--cpu'):
        assert s in str(e), (s, str(e))
    _raises(engine.InsufficientMemory, engine.plan_gpu_workers, 1 * GiB, total, 0.3)  # less than the headroom
    # a small device: the headroom is 2 GiB, not 10 %
    p = engine.plan_gpu_workers(8 * GiB, 8 * GiB, 0.3)
    assert p['headroom_MB'] == 2048 and p['gpu_workers'] == 2


def test_plan_all_queries_and_given_values():
    total = 24 * GiB
    assert engine.plan_gpu_workers(23 * GiB, total, None)['gpu_workers'] == 1  # all queries: one per 24 GB card
    _raises(engine.InsufficientMemory, engine.plan_gpu_workers, 12 * GiB, total, None)
    # gpu_workers given, budget decided: must fit
    assert engine.plan_gpu_workers(20 * GiB, total, 0.3, gpu_workers=2)['gpu_workers'] == 2
    need = engine.auto_need_mb(0.3) * MiB
    _raises(engine.InsufficientMemory, engine.plan_gpu_workers, int(0.1 * total + 1.5 * need), total, 0.3,
            gpu_workers=2)
    # budget given, workers decided from it (not from the free memory)
    p = engine.plan_gpu_workers(20 * GiB, total, 0.3, gpu_mem_budget='4G')
    assert p['gpu_workers'] == min(2, int(4 * GiB // need)) and p['budget'] == ('bytes', 4 * GiB)
    assert engine.plan_gpu_workers(20 * GiB, total, 0.3, gpu_mem_budget=0.5)['budget'] == ('bytes', 12 * GiB)
    _raises(engine.InsufficientMemory, engine.plan_gpu_workers, 20 * GiB, total, 0.3, gpu_mem_budget='1G')


def _engine(d, device, **kw):
    ck = Path(d) / 'm.pth'
    ck.write_bytes(b'0')
    return engine.Engine(engine.ModelOptions(ckpt=str(ck), device=device, kept_thr=0.3), log=lambda *a: None,
                         **kw)


def test_engine_sizing_with_mocked_probe():
    orig = engine.probe_device_memory
    try:
        with tempfile.TemporaryDirectory() as d:
            engine.probe_device_memory = lambda dev: (20 * GiB, 24 * GiB)
            eng = _engine(d, 'cuda:0', gpu_workers=None, gpu_mem_budget=None, settings={'gpu_workers': None})
            eng.prepare()
            assert eng.engine_info['gpu_workers'] == 'auto'
            eng._size_workers()
            assert eng.n_gpu == 2 and eng.budget[0] == 'bytes' and eng.max_inflight == 2 * (4 + 2 + 4)
            info = eng.engine_info
            assert info['gpu_workers'] == 2 and info['auto_sizing']['gpu_workers'] == 2
            assert info['auto_sizing']['free_MB'] == 20480 and info['settings']['gpu_workers'] == 2
            assert info['gpu_mem_budget'] == '%dM' % (eng.budget[1] // MiB)
            # nothing fits: the engine fails with the numbers, the job manifest says so
            engine.probe_device_memory = lambda dev: (2 * GiB, 24 * GiB)
            eng = _engine(d, 'cuda:0', gpu_workers=None, gpu_mem_budget=None)
            out = Path(d) / 'out'
            jid = eng.submit([str(Path(d) / 'a.png')], str(out))
            e = _raises(engine.EngineFailed, eng.start)
            assert 'not enough free GPU memory' in str(e) and 'need per GPU worker' in str(e)
            m = json.loads((out / 'job.json').read_text())
            assert m['job'] == jid and m['status'] == 'failed' and 'free GPU memory' in m['error']
            # measured configurations only
            eng = engine.Engine(engine.ModelOptions(ckpt=str(Path(d) / 'm.pth'), device='cuda:0', input_size=1024),
                                gpu_workers=None, gpu_mem_budget=None, log=lambda *a: None)
            assert 'input size' in str(_raises(engine.EngineFailed, eng.start))
            # explicit values: no probe at all
            engine.probe_device_memory = None
            eng = _engine(d, 'cuda:0', gpu_workers=2, gpu_mem_budget='3G')
            eng.prepare()
            eng._size_workers()
            assert eng.n_gpu == 2 and eng.budget == ('bytes', 3 * GiB) and eng.auto_sizing is None
            # CPU: one worker, no budget, no probe
            eng = _engine(d, 'cpu', gpu_workers=None, gpu_mem_budget=None)
            eng.prepare()
            eng._size_workers()
            assert eng.n_gpu == 1 and eng.budget is None
    finally:
        engine.probe_device_memory = orig


# ------------------------------------------------------------------ bounded back-off (worker processes, CPU)

def _need_stack():
    if _MISSING:
        raise unittest.SkipTest('needs torch, mmcv (third_party/mmcv), mmdet, OpenCV and scikit-image '
                                '(rocm/install_rocm.sh); missing: %s' % ', '.join(_MISSING))


def _images(d, n=4):
    import cv2
    import numpy as np
    paths = []
    for i in range(n):
        p = Path(d) / ('img%d.png' % i)
        cv2.imwrite(str(p), np.full((120, 200, 3), 255, np.uint8))
        paths.append(str(p))
    return paths


def _run(d, gpu_workers, oom):
    """One job of 4 images on a CPU engine with fake GPU workers -> (engine, job summary, manifest, log lines)."""
    from oom_hooks import FakeGPU
    marker = Path(d) / 'markers'
    marker.mkdir(exist_ok=True)
    logs = []
    ck = Path(d) / 'm.pth'
    ck.write_bytes(b'0')
    eng = engine.Engine(engine.ModelOptions(ckpt=str(ck), device='cpu', kept_thr=0.3), gpu_workers=gpu_workers,
                        pre_workers=1, post_workers=1, gpu_threads=1, log=lambda *a: logs.append(' '.join(map(str, a))),
                        _test_hooks=FakeGPU(oom, str(marker)))
    out = Path(d) / 'out'
    try:
        eng.start()
        jid = eng.submit(_images(d), str(out))
        s = eng.wait(jid, timeout=300)
    finally:
        eng.shutdown(timeout=60)
    return eng, s, json.loads((out / 'job.json').read_text()), logs


def test_backoff_one_oom_requeued_job_succeeds():
    _need_stack()
    with tempfile.TemporaryDirectory() as d:
        eng, s, m, logs = _run(d, 2, {'img2': 'once'})
        assert s['status'] == 'done' and s['counts']['done'] == 4, s
        ev = m['engine']['backoff']
        assert len(ev) == 1 and ev[0]['image'] == 'img2' and ev[0]['action'] == 'worker stopped, image requeued'
        assert ev[0]['gpu_workers_left'] == 1 and ev[0]['attempt'] == 1 and 'memory' in ev[0]
        assert m['engine']['gpu_workers_active'] == 1 and m['engine']['gpu_workers'] == 2
        img2 = [i for i in m['images'] if i['id'] == 'img2'][0]
        assert img2['status'] == 'done' and 'requeued' in img2['note']
        assert img2['gpu_worker'] != ev[0]['gpu_worker']  # done by the worker that is left
        assert any('BACK-OFF' in line and 'img2' in line for line in logs)
        # the same image content gives the same lines, whichever worker ran it and after a requeue
        lines = {i: json.loads((Path(d) / 'out' / ('img%d.json' % i)).read_text())['lines'] for i in range(4)}
        assert lines[0] and all(lines[i] == lines[0] for i in range(4))


def test_backoff_second_oom_of_an_image_fails_loud():
    _need_stack()
    with tempfile.TemporaryDirectory() as d:
        eng, s, m, logs = _run(d, 2, {'img1': 'always'})
        assert s['status'] == 'failed' and eng.state == 'failed'
        assert 'out of device memory' in s['error'] and 'second out of memory' in s['error']
        assert m['status'] == 'failed' and 'second out of memory' in m['error']
        ev = m['engine']['backoff']
        assert [e['action'] for e in ev] == ['worker stopped, image requeued',
                                             'engine failed (its second out of memory (requeued once already))']
        assert [e['image'] for e in ev] == ['img1', 'img1'] and ev[0]['gpu_worker'] != ev[1]['gpu_worker']


def test_backoff_last_worker_oom_fails_loud():
    _need_stack()
    with tempfile.TemporaryDirectory() as d:
        eng, s, m, logs = _run(d, 1, {'img0': 'once'})  # 'once' would pass on another worker; there is none
        assert s['status'] == 'failed' and 'no GPU worker left' in s['error']
        assert m['status'] == 'failed' and len(m['engine']['backoff']) == 1
        assert m['engine']['backoff'][0]['action'] == 'engine failed (no GPU worker left)'


def test_backoff_serve_exit_on_failure():
    """serve --exit-on-failure: the back-off's final failure still stops the server with code 2."""
    _need_stack()
    import signal
    import lineformer_client as client
    import lineformer_serve
    from oom_hooks import FakeGPU
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / 'm.pth'
        ck.write_bytes(b'0')
        eng = engine.Engine(engine.ModelOptions(ckpt=str(ck), device='cpu', kept_thr=0.3), gpu_workers=1,
                            pre_workers=1, post_workers=1, gpu_threads=1, log=lambda *a: None,
                            _test_hooks=FakeGPU({'img0': 'always'}, d))
        ready = Path(d) / 'ready.json'
        seen = {}

        def helper():
            for _ in range(1200):
                if ready.exists():
                    break
                time.sleep(0.1)
            lf = client.LineFormerClient(json.loads(ready.read_text())['url'], timeout=10)
            seen['job'] = lf.submit(_images(d), os.path.join(d, 'out'))

        watchdog = threading.Timer(240.0, os.kill, (os.getpid(), signal.SIGTERM))
        watchdog.start()
        threading.Thread(target=helper, daemon=True).start()
        try:
            code = lineformer_serve.serve(eng, port=0, ready_file=str(ready), exit_on_failure=True)
        finally:
            watchdog.cancel()
        assert code == 2 and eng.state == 'failed' and 'no GPU worker left' in eng.error
        m = json.loads((Path(d) / 'out' / 'job.json').read_text())
        assert m['job'] == seen['job'] and m['status'] == 'failed'


def test_backoff_hook_is_not_reachable_from_the_environment():
    """The fault injection exists only as an Engine argument; no environment variable or CLI option sets it."""
    import inspect
    import lineformer_cli
    src = inspect.getsource(engine) + inspect.getsource(lineformer_cli)
    assert '_test_hooks' not in inspect.getsource(lineformer_cli)
    assert 'os.environ' not in inspect.getsource(engine.gpu_worker) and 'getenv' not in src
    assert src.count('_test_hooks') >= 2
