# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""Client of `lineformer serve` - standard library only (no torch, no numpy), Python >= 3.8.

Copy this one file into another project (or put the LineFormer checkout on sys.path) and:

    from lineformer_client import LineFormerClient
    lf = LineFormerClient('http://127.0.0.1:8775')
    lf.health()['state']                                   # 'running'
    job = lf.submit(['/abs/a.png', {'id': 'b2', 'path': '/abs/b.png'}], out='/abs/out',
                    instances=True, require={'kept_thr': 0.3})
    final = lf.wait(job)                                   # polls; raises JobFailed unless the job ends 'done'
    lines = read_lines('/abs/out', 'a')                    # [[{"x":..,"y":..}, ...], ...]

    lf.status(job)  lf.jobs()  lf.cancel(job)  lf.run(images, out)  (= submit + wait)

Command line:
    python lineformer_client.py [--url URL] health | jobs | status JOB | cancel JOB | wait JOB
    python lineformer_client.py submit --out DIR [--list FILE] [IMAGES...] [--instances] [--masks] [--force]
                                       [--priority N] [--wait]

Paths are server-side and must be absolute (relative ones are made absolute here, against this process's working
directory). Output files and the manifest <out>/job.json are described in lineformer_jobs.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = 'http://127.0.0.1:8775'
FINAL = ('done', 'done_with_errors', 'cancelled', 'interrupted', 'failed')


class LineFormerError(RuntimeError):
    """The server answered with an error (status code and message), or could not be reached."""

    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status = status


class JobFailed(LineFormerError):
    """wait(): the job ended in a state other than the accepted ones. .summary holds the job summary."""

    def __init__(self, summary):
        c = summary.get('counts', {})
        msg = 'job %s ended %s (done %s, skipped %s, failed %s, of %s)%s' % (
            summary.get('job'), summary.get('status'), c.get('done'), c.get('skipped'), c.get('failed'),
            c.get('total'), ': ' + summary['error'] if summary.get('error') else '')
        errs = summary.get('errors') or []
        if errs:
            last = (errs[0].get('error') or '').strip().splitlines()
            msg += '; first failure %s: %s' % (errs[0].get('id'), last[-1] if last else '?')
        super().__init__(msg)
        self.summary = summary


class LineFormerClient:
    def __init__(self, url=DEFAULT_URL, timeout=60.0):
        self.url = url.rstrip('/')
        self.timeout = timeout

    def _call(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={'Content-Type': 'application/json'} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read().decode()).get('error')
            except Exception:
                msg = e.reason
            raise LineFormerError('%s %s -> %d: %s' % (method, path, e.code, msg), status=e.code) from None
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            raise LineFormerError('cannot reach the LineFormer server at %s (%s); is `lineformer serve` running?'
                                  % (self.url, getattr(e, 'reason', e))) from None

    def health(self):
        return self._call('GET', '/health')

    def submit(self, images, out, instances=False, masks=False, force=False, priority=0, ids=None, name=None,
               require=None):
        """Queue a job; returns the job id. images: paths or {"id", "path"} dicts. require: model options the
        caller relies on (e.g. {"kept_thr": 0.3}); the server refuses the job (409) if it runs with others."""
        imgs = []
        for it in images:
            if isinstance(it, dict):
                it = dict(it, path=os.path.abspath(str(it['path'])))
            else:
                it = os.path.abspath(str(it))
            imgs.append(it)
        body = {'images': imgs, 'out': os.path.abspath(str(out)),
                'outputs': {'instances': bool(instances), 'masks': bool(masks)}, 'force': bool(force),
                'priority': int(priority)}
        if ids:
            body['ids'] = ids
        if name:
            body['name'] = name
        if require:
            body['model'] = dict(require)
        return self._call('POST', '/jobs', body)['job']

    def status(self, job, images=False):
        return self._call('GET', '/jobs/%s%s' % (job, '?images=1' if images else ''))

    def jobs(self):
        return self._call('GET', '/jobs')['jobs']

    def cancel(self, job):
        return self._call('POST', '/jobs/%s/cancel' % job, {})

    def wait(self, job, poll=1.0, timeout=None, accept=('done',), progress=None):
        """Poll until the job is final; returns its summary. Raises JobFailed if the final state is not in
        `accept` (default: only 'done'; pass accept=FINAL to never raise), TimeoutError after `timeout` s."""
        t0 = time.time()
        while True:
            s = self.status(job)
            if progress:
                progress(s)
            if s['status'] in FINAL:
                if s['status'] not in accept:
                    raise JobFailed(s)
                return s
            if timeout is not None and time.time() - t0 > timeout:
                raise TimeoutError('job %s still %s after %s s' % (job, s['status'], timeout))
            time.sleep(poll)

    def run(self, images, out, **kw):
        """submit + wait; keyword arguments of both."""
        wait_kw = {k: kw.pop(k) for k in ('poll', 'timeout', 'accept', 'progress') if k in kw}
        return self.wait(self.submit(images, out, **kw), **wait_kw)


def read_lines(out, image_id):
    """The lines of one image from <out>/<id>.json."""
    with open(os.path.join(str(out), image_id + '.json'), encoding='utf-8') as f:
        return json.load(f)['lines']


def read_manifest(out):
    with open(os.path.join(str(out), 'job.json'), encoding='utf-8') as f:
        return json.load(f)


def _read_list(path):
    items = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if '\t' in s:
                iid, p = s.split('\t', 1)
                items.append({'id': iid.strip(), 'path': p.strip()})
            else:
                items.append(s)
    return items


def main(argv=None):
    ap = argparse.ArgumentParser(prog='lineformer_client', description=__doc__.splitlines()[0])
    ap.add_argument('--url', default=os.environ.get('LINEFORMER_URL', DEFAULT_URL))
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('health')
    sub.add_parser('jobs')
    for c in ('status', 'cancel', 'wait'):
        p = sub.add_parser(c)
        p.add_argument('job')
        if c == 'status':
            p.add_argument('--images', action='store_true')
    p = sub.add_parser('submit')
    p.add_argument('images', nargs='*')
    p.add_argument('--list')
    p.add_argument('--out', required=True)
    p.add_argument('--instances', action='store_true')
    p.add_argument('--masks', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--priority', type=int, default=0)
    p.add_argument('--ids', choices=('stem', 'parent_stem'), default=None)
    p.add_argument('--name', default=None)
    p.add_argument('--wait', action='store_true')
    a = ap.parse_args(argv)
    lf = LineFormerClient(a.url)
    try:
        if a.cmd == 'health':
            out = lf.health()
        elif a.cmd == 'jobs':
            out = lf.jobs()
        elif a.cmd == 'status':
            out = lf.status(a.job, images=a.images)
        elif a.cmd == 'cancel':
            out = lf.cancel(a.job)
        elif a.cmd == 'wait':
            out = lf.wait(a.job, accept=FINAL)
        else:
            items = list(a.images) + (_read_list(a.list) if a.list else [])
            if not items:
                ap.error('no images given')
            job = lf.submit(items, a.out, instances=a.instances, masks=a.masks, force=a.force, priority=a.priority,
                            ids=a.ids, name=a.name)
            out = lf.wait(job, accept=FINAL) if a.wait else {'job': job}
    except LineFormerError as e:
        print('error: %s' % e, file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1))
    if isinstance(out, dict) and out.get('status') in FINAL:
        return 0 if out['status'] == 'done' else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
