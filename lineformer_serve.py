"""`lineformer serve`: one long-lived owner of the GPU that takes LineFormer jobs over HTTP (localhost).

    lineformer serve --ckpt iter_3000.pth --port 8775 --gpu-workers 2 --kept-only

The models are loaded once (lineformer_engine.Engine); callers submit lists of images with an output directory
and poll. Model options (device, input size, kept-queries threshold, tiling) are server-wide: a job that needs
other model options needs another server (another port). Standard library only (http.server); JSON in and out.

API (all bodies JSON; errors are {"error": "..."} with status 400 bad request, 404 unknown job, 409 output
directory in use by an active job or model options differ, 503 engine not accepting jobs):
  POST /jobs                {"images": ["/abs/a.png", {"id": "x", "path": "/abs/b.png"}, ...],
                             "out": "/abs/out/dir", "outputs": {"instances": bool, "masks": bool},
                             "force": bool, "priority": int (higher first, default 0), "ids": "stem"|"parent_stem",
                             "name": str, "model": {...}}
                            -> 201 {"job": id, "status": ..., "counts": {...}}
                            Paths must be absolute (server side). "model" (optional) lists model options the
                            caller relies on, e.g. {"kept_thr": 0.3, "input_size": "config"}; a mismatch -> 409.
  GET  /jobs                -> {"jobs": [{job, name, status, out, counts, priority, created}, ...]}
  GET  /jobs/<id>           -> job summary: status, counts (pending running done failed skipped duplicate
                               cancelled total), timing (wall_s, images_per_s, latency / pre / gpu / post stats),
                               first 20 errors; ?images=1 adds the per-image records
  POST /jobs/<id>/cancel    -> {"cancelled": bool, "job": summary}: pending images are cancelled, running ones
                               finish (their outputs are written); the job ends "cancelled"
  GET  /health              -> state, device, versions, model options, fingerprint, workers (pids, MSDA path,
                               memory cap), last memory statistics (device_free_MB), queue sizes

Jobs run FIFO (higher priority first); images of all jobs share the GPU workers. One active job per output
directory. Every job writes <out>/job.json (lineformer_jobs.py describes the files).

Shutdown: SIGINT / SIGTERM (to the server process) stop accepting jobs (503), let in-flight images finish
(--drain-timeout), end unfinished jobs "interrupted" (manifests written; resubmitting the same job skips what is
done), then stop the workers and the HTTP server. A second signal abandons in-flight images at once.
The server binds 127.0.0.1 by default; it has no authentication, so do not bind it to a public interface.
"""
from __future__ import annotations

import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import lineformer_jobs as jobs
from lineformer_engine import EngineFailed

MAX_BODY = 64 * 2 ** 20
MODEL_KEYS = ('device', 'msda', 'kept_thr', 'input_size', 'tile', 'tile_overlap', 'tile_link_thr', 'tile_min_px',
              'ckpt', 'config')


class ModelMismatch(jobs.JobError):
    """The job relies on model options this server does not have."""


class _Handler(BaseHTTPRequestHandler):
    server_version = 'lineformer-serve/' + jobs.ENGINE_VERSION
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        if self.server.verbose:
            sys.stderr.write('[lineformer-serve] %s %s\n' % (self.address_string(), fmt % args))

    def _send(self, code, obj):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._send(code, {'error': msg})

    def _body(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n > MAX_BODY:
            raise jobs.JobError('request body larger than %d bytes' % MAX_BODY)
        raw = self.rfile.read(n) if n else b''
        if not raw:
            return {}
        try:
            obj = json.loads(raw.decode('utf-8'))
        except ValueError as e:
            raise jobs.JobError('body is not valid JSON: %s' % e)
        if not isinstance(obj, dict):
            raise jobs.JobError('body must be a JSON object')
        return obj

    def do_GET(self):
        eng = self.server.engine
        u = urlparse(self.path)
        parts = [p for p in u.path.split('/') if p]
        q = parse_qs(u.query)
        try:
            if parts == ['health']:
                return self._send(200, eng.health())
            if parts == ['jobs']:
                return self._send(200, {'jobs': eng.list_jobs()})
            if len(parts) == 2 and parts[0] == 'jobs':
                want = q.get('images', ['0'])[0] not in ('0', 'false', '')
                return self._send(200, eng.status(parts[1], images=want))
            return self._err(404, 'no such endpoint: GET %s' % u.path)
        except KeyError as e:
            return self._err(404, 'unknown job %s' % e)
        except Exception as e:  # never kill the server thread on a bad request
            return self._err(500, '%s: %s' % (type(e).__name__, e))

    def do_POST(self):
        eng = self.server.engine
        u = urlparse(self.path)
        parts = [p for p in u.path.split('/') if p]
        try:
            body = self._body()
            if parts == ['jobs']:
                return self._submit(eng, body)
            if len(parts) == 3 and parts[0] == 'jobs' and parts[2] == 'cancel':
                changed, summ = eng.cancel(parts[1])
                return self._send(200, {'cancelled': changed, 'job': summ})
            return self._err(404, 'no such endpoint: POST %s' % u.path)
        except KeyError as e:
            return self._err(404, 'unknown job %s' % e)
        except (jobs.OutDirBusy, ModelMismatch) as e:
            return self._err(409, str(e))
        except jobs.JobError as e:
            return self._err(400, str(e))
        except EngineFailed as e:
            return self._err(503, str(e))
        except Exception as e:
            return self._err(500, '%s: %s' % (type(e).__name__, e))

    def _submit(self, eng, body):
        known = {'images', 'out', 'outputs', 'force', 'priority', 'ids', 'name', 'model'}
        extra = set(body) - known
        if extra:
            raise jobs.JobError('unknown keys %s (known: %s)' % (sorted(extra), ', '.join(sorted(known))))
        if 'images' not in body or 'out' not in body:
            raise jobs.JobError('"images" and "out" are required')
        model = body.get('model') or {}
        if not isinstance(model, dict):
            raise jobs.JobError('"model" must be an object')
        mo = eng.engine_info['model_options']
        bad = sorted(set(model) - set(MODEL_KEYS))
        if bad:
            raise jobs.JobError('unknown model options %s (known: %s)' % (bad, ', '.join(MODEL_KEYS)))
        diff = ['%s: server %r, job %r' % (k, mo.get(k), v) for k, v in model.items() if mo.get(k) != v]
        if diff:
            raise ModelMismatch('model options differ from this server (%s); start another server with them'
                                % '; '.join(diff))
        outputs = body.get('outputs') or {}
        if not isinstance(outputs, dict):
            raise jobs.JobError('"outputs" must be an object like {"instances": true, "masks": false}')
        force = body.get('force', False)
        if not isinstance(force, bool):
            raise jobs.JobError('"force" must be true or false')
        jid = eng.submit(body['images'], body['out'], outputs=outputs, force=force,
                         priority=body.get('priority', 0), ids=body.get('ids', 'stem'), name=body.get('name'),
                         require_absolute=True)
        s = eng.status(jid)
        return self._send(201, {'job': jid, 'status': s['status'], 'counts': s['counts']})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, engine, verbose=False):
        super().__init__(addr, _Handler)
        self.engine = engine
        self.verbose = verbose


def serve(engine, host='127.0.0.1', port=8775, drain_timeout=600.0, verbose=False, ready_file=None):
    """Start the engine (if not started), serve until SIGINT / SIGTERM, then shut down. Returns an exit code."""
    httpd = Server((host, port), engine, verbose=verbose)  # bind first: a busy port fails before models load
    if engine.state == 'created':
        engine.start()
    log = engine.log
    stopping = {'n': 0}

    def stop_all():
        engine.shutdown(timeout=drain_timeout)
        httpd.shutdown()

    def on_signal(signum, frame):
        stopping['n'] += 1
        if stopping['n'] == 1:
            log('signal %d: draining (no new jobs; in-flight images finish, up to %d s)' % (signum, drain_timeout))
            threading.Thread(target=stop_all, daemon=True).start()
        else:
            log('signal %d again: abandoning in-flight images' % signum)
            engine._abandon.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    log('serving on http://%s:%d (GET /health, POST /jobs)' % (host, port))
    if ready_file:
        jobs.write_json_atomic(ready_file, {'url': 'http://%s:%d' % (host, port), 'time': time.time()})
    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        httpd.server_close()
    if engine.state not in ('stopped', 'failed'):
        engine.shutdown(timeout=drain_timeout)
    log('server stopped (engine %s)' % engine.state)
    return 2 if engine.state == 'failed' else 0
