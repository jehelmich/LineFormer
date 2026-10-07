"""Load check of a running `lineformer serve`: concurrent jobs + one cancelled job, then consistency checks.

    python tools/engine/serve_check.py --url http://127.0.0.1:8775 --images list.txt --out-base /tmp/sc \
        [--jobs 2] [--cancel-after 3] [--instances --masks] [--report report.json]

Submits --jobs jobs over the same image list (out dirs <out-base>/job<k>) and one more job (<out-base>/cancel,
priority --cancel-priority, default 1 so that it runs first) that is cancelled --cancel-after seconds later, while
images of it are in flight, and waits for all. Checks (exit 1 if any fails):
  * every normal job ends "done" with every image done (none failed, skipped or duplicate);
  * the cancelled job ends "cancelled": done + cancelled = all, none failed, and only done images have outputs;
  * per job: the manifest's counts equal its per-image statuses; every done image has exactly one <id>.json whose
    "job" is this job and whose "image" is the listed path; no other <id>.json, no leftover temp files;
  * the lines of every image are identical across the normal jobs (and the cancelled job's done images).
Throughput: images done by all jobs / (first job start .. last job end), from the manifests.
The out dirs must not exist (a fresh check never skips work). Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lineformer_client import FINAL, LineFormerClient, _read_list  # noqa: E402


def _load(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def check_job(out, job_id, expect, items):
    """-> (problems, lines by id, manifest)."""
    probs = []
    man = _load(Path(out) / 'job.json')
    if man['job'] != job_id:
        probs.append('%s: manifest is of job %s, expected %s' % (out, man['job'], job_id))
    st = [r['status'] for r in man['images']]
    counts = {k: st.count(k) for k in set(st)}
    for k, v in man['counts'].items():
        if k != 'total' and counts.get(k, 0) != v:
            probs.append('%s: manifest count %s=%d but %d images have that status' % (out, k, v, counts.get(k, 0)))
    if man['counts']['total'] != len(items):
        probs.append('%s: %d images in the manifest, %d submitted' % (out, man['counts']['total'], len(items)))
    if man['status'] != expect:
        probs.append('%s: job status %s, expected %s' % (out, man['status'], expect))
    if expect == 'done' and counts.get('done', 0) != len(items):
        probs.append('%s: %d of %d images done (%s)' % (out, counts.get('done', 0), len(items), counts))
    if expect == 'cancelled' and (counts.get('failed') or counts.get('done', 0) + counts.get('cancelled', 0)
                                  != len(items)):
        probs.append('%s: cancelled job has %s' % (out, counts))
    done = {r['id']: r for r in man['images'] if r['status'] == 'done'}
    lines = {}
    files = sorted(os.listdir(out))
    jsons = [f[:-5] for f in files if f.endswith('.json') and f != 'job.json']
    tmps = [f for f in files if '.tmp' in f]
    if tmps:
        probs.append('%s: leftover temp files %s' % (out, tmps[:5]))
    extra = sorted(set(jsons) - set(done))
    if extra:
        probs.append('%s: outputs of images that are not done: %s' % (out, extra[:5]))
    for iid, r in done.items():
        p = Path(out) / (iid + '.json')
        if not p.exists():
            probs.append('%s: done image %s has no output' % (out, iid))
            continue
        obj = _load(p)
        if obj.get('job') != job_id:
            probs.append('%s: %s written by job %s' % (out, iid, obj.get('job')))
        if obj.get('image') != r['path']:
            probs.append('%s: %s is of image %s, manifest says %s' % (out, iid, obj.get('image'), r['path']))
        lines[iid] = obj['lines']
    return probs, lines, man


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', default='http://127.0.0.1:8775')
    ap.add_argument('--images', required=True, help='list file ("<path>" or "<id>\\t<path>" per line)')
    ap.add_argument('--out-base', required=True)
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--cancel-after', type=float, default=3.0)
    ap.add_argument('--cancel-priority', type=int, default=1,
                    help='priority of the job to cancel (default 1: it runs first, so it is cancelled mid-way with '
                         'images in flight; 0: it queues behind the others)')
    ap.add_argument('--instances', action='store_true')
    ap.add_argument('--masks', action='store_true')
    ap.add_argument('--report', default=None)
    a = ap.parse_args(argv)
    lf = LineFormerClient(a.url)
    items = _read_list(a.images)
    base = Path(a.out_base).resolve()
    outs = [base / ('job%d' % k) for k in range(1, a.jobs + 1)] + [base / 'cancel']
    for o in outs:
        if o.exists():
            raise SystemExit('%s exists; the check needs fresh output directories' % o)
    h0 = lf.health()
    t0 = time.time()
    ids = [lf.submit(items, o, instances=a.instances, masks=a.masks, name='serve_check-%s' % o.name,
                     priority=a.cancel_priority if o.name == 'cancel' else 0) for o in outs]
    time.sleep(a.cancel_after)
    before = lf.status(ids[-1])['counts']
    cancel_resp = lf.cancel(ids[-1])
    if a.cancel_priority > 0 and not before['done'] + before['running']:
        print('note: the cancelled job had not started; raise --cancel-after', file=sys.stderr)
    finals = [lf.wait(j, poll=0.5, accept=FINAL) for j in ids]
    t1 = time.time()
    probs, lines_by_job, mans = [], [], []
    for j, o, f in zip(ids, outs, finals):
        p, lines, man = check_job(o, j, 'cancelled' if o.name == 'cancel' else 'done', items)
        probs += p
        lines_by_job.append(lines)
        mans.append(man)
    ref = lines_by_job[0]
    for k, lines in enumerate(lines_by_job[1:], 2):
        diff = [i for i in lines if lines[i] != ref.get(i)]
        if diff:
            probs.append('job %d: lines differ from job 1 on %d images: %s' % (k, len(diff), diff[:5]))
    starts = [m['timing']['started'] for m in mans if m['timing'].get('started')]
    ends = [m['timing']['finished'] for m in mans if m['timing'].get('finished')]
    n_done = sum(m['counts']['done'] for m in mans)
    span = max(ends) - min(starts) if starts and ends else None
    rep = {'url': a.url, 'jobs': ids, 'outs': [str(o) for o in outs], 'n_images_per_job': len(items),
           'cancel_job_counts_before_cancel': before,
           'cancel_response_counts': cancel_resp['job']['counts'],
           'cancel_response_status': cancel_resp['job']['status'], 'final': [
               {'job': f['job'], 'status': f['status'], 'counts': f['counts'],
                'images_per_s': f['timing'].get('images_per_s'), 'wall_s': f['timing'].get('wall_s')}
               for f in finals],
           'n_done_total': n_done, 'span_s': span, 'images_per_s_all_jobs': n_done / span if span else None,
           'client_wall_s': t1 - t0, 'health_before': {k: h0.get(k) for k in ('state', 'device_free_MB')},
           'health_after': {k: v for k, v in lf.health().items() if k in ('state', 'device_free_MB',
                                                                           'worker_stats', 'images_pending')},
           'problems': probs, 'verdict': 'PASS' if not probs else 'FAIL'}
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        with open(a.report, 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=1)
    print(json.dumps({k: rep[k] for k in ('verdict', 'cancel_job_counts_before_cancel', 'cancel_response_counts',
                                          'final', 'n_done_total', 'span_s', 'images_per_s_all_jobs',
                                          'problems')}, indent=1))
    return 0 if not probs else 1


if __name__ == '__main__':
    sys.exit(main())
