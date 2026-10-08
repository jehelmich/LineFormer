# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
# Contains code adapted from OpenMMLab mmdetection (Apache-2.0, Copyright OpenMMLab).
"""Production job engine: images in, LineFormer outputs out, with the GPU work spread over worker processes.

    from lineformer_engine import Engine, ModelOptions
    eng = Engine(ModelOptions(ckpt='iter_3000.pth', device='cuda:0', kept_thr=0.3), gpu_workers=2)
    eng.start()
    job = eng.submit(['a.png', 'b.png'], out='out/', outputs={'instances': True})
    eng.wait(job)                       # -> the job summary; eng.status(job) while it runs
    eng.shutdown()

Pipeline (the per-image maths of infer.get_dataseries, not re-implemented):
  pre workers   read the image (cv2.imread, BGR), sha256 of the file, mmdet's test pipeline (the first half of
                mmdet.apis.inference_detector) - or, with tiling, the crops of tiling.crop_boxes, each through it;
  GPU workers   N processes, each with its own model: collate + forward (the second half of inference_detector);
                masks go to the post workers through POSIX shared memory;
  post workers  infer.get_dataseries with the forward swapped for this result (lines: instances with score > 0.3),
                tiling.merge_instances for tiled images, then the output files (lineformer_jobs.py lists them),
                with lines and instances in a deterministic geometric order (instance_order; lineformer_jobs.py
                "Order"): lines by (leftmost x, mean y, -score), line i = instance i, the instances without a
                line after them by (box x1, box centre y, -score). The model's own order differs between CPU and
                GPU; infer.get_dataseries is not changed.
The main process holds no model and never initialises the GPU runtime; it feeds images (bounded number in
flight), collects results and writes the manifests (<out>/job.json).

Model options (ModelOptions; one set per engine - a different input size or threshold needs another engine):
  device, msda       as infer.load_model (msda_compat.py); device 'auto' = 'cuda:0' if PyTorch sees a GPU, else
                     'cpu' (resolve_device, probed in a subprocess)
  kept_thr           kept-queries mode (kept_queries.py), None = OFF (the default; the environment variable is NOT
                     read here). A threshold in (0, 1), e.g. 0.3, or 0.1 to keep low-score instances in the
                     .instances.npz: only queries whose class score reaches it are post-processed and returned.
  input_size         'config' (the config's 512 fit, default), N (fit N x N) or 'native' (scale_compat.py;
                     'native' is EXPERIMENTAL, see docs/VALIDATION.md "Input scale")
  tile, tile_overlap EXPERIMENTAL: native crops of tile x tile px with that overlap, merged (tiling.py); needs input_size
                     'native' (set automatically); per-crop instances below the score threshold of the merge
                     (kept_thr if set, else 0.3) are dropped before the merge.
Every GPU worker builds its model with scale_compat.build_model and calls kept_queries.configure(model, thr)
explicitly (off = False), then checks the threshold and the parameter device.

Memory: gpu_mem_budget is the share of the device all GPU workers together may use (a fraction <= 1, default
0.85, or an absolute size such as "4G"); each worker caps its caching allocator at budget / N
(torch.cuda.set_per_process_memory_fraction). Kept-queries mode needs ~1 GB per worker, the full mmdet path up to
~12 GB per worker on 1.5-3.5k px images.

Failure rules:
  * an image that cannot be read or fails in a stage: that image fails (traceback recorded), the job goes on;
  * out of memory on the device, a worker that raises outside an image, or a worker process that dies: the
    ENGINE fails - every active job ends with status "failed" and the error, its manifest is written, the
    workers are stopped, nothing is retried;
  * shutdown(): no new images are fed; in-flight images finish (or, after the timeout / abandon=True, are left);
    unfinished jobs end "interrupted" with their pending images listed; outputs already written stay valid, so
    a rerun of the same job skips them (skip-if-done).
"""
from __future__ import annotations

import hashlib
import math
import os
import platform
import queue as queue_mod
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import lineformer_jobs as jobs

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = str(HERE / 'lineformer_swin_t_config.py')
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
LINE_THR = 0.3  # infer.get_dataseries -> do_instance(score_thr=0.3); parse_result keeps score > 0.3


# ================================================================== per-image pieces (also used by the unit tests)

def worker_signals():
    """Worker processes ignore SIGINT: Ctrl-C reaches the whole process group, and the main process decides
    (graceful shutdown). They are stopped by a sentinel or terminate() from the main process."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def cpu_only_torch():
    """For processes that never use a GPU, before mmcv is imported. Under WSL + ROCm 7.2 the first
    torch.cuda.is_available() call (mmcv does it at import) starts the HSA runtime, whose two threads then spin at
    100 % for the life of the process. Here the probe answers False."""
    import torch
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0


def apply_threads(n):
    import cv2
    import torch
    torch.set_num_threads(n)
    cv2.setNumThreads(n)


def sync(device):
    import torch
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def build_test_pipeline(cfg):
    """First half of mmdet.apis.inference_detector for ndarray input (verbatim)."""
    from mmdet.datasets import replace_ImageToTensor
    from mmdet.datasets.pipelines import Compose
    cfg = cfg.copy()
    cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    return Compose(cfg.data.test.pipeline)


def preprocess(pipeline, img):
    data = dict(img=img)
    return pipeline(data)


def data_shape(data):
    imgs = data['img']
    if len(imgs) != 1:
        raise RuntimeError('expected one test-time augmentation, got %d' % len(imgs))
    return tuple(imgs[0].data.shape)


def forward(model, datas, check_no_pad=True):
    """Second half of mmdet.apis.inference_detector (verbatim), for a list of pre-processed images that must all
    have one tensor shape (so mmcv's collate pads nothing). Returns the list of per-image results.

    check_no_pad: raise if an image's img_shape differs from the batch input (any padding); a batch of several
    images needs it. A single image whose own pipeline pads (scale_compat 'native', Pad to 32) passes False."""
    import torch
    from mmcv.parallel import collate, scatter
    shapes = sorted(set(data_shape(d) for d in datas))
    if len(shapes) != 1:
        raise RuntimeError('batch with several tensor shapes %s: collate would pad' % shapes)
    device = next(model.parameters()).device
    data = collate(datas, samples_per_gpu=len(datas))
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)
    if check_no_pad:
        for metas in data['img_metas']:
            for m in metas:
                if tuple(m['img_shape'][:2]) != tuple(m['batch_input_shape']) or \
                        tuple(m.get('pad_shape', m['img_shape'])[:2]) != tuple(m['img_shape'][:2]):
                    raise RuntimeError('padding in batch: img_shape %s pad_shape %s batch_input_shape %s' % (
                        m['img_shape'], m.get('pad_shape'), m['batch_input_shape']))
    if len(results) != len(datas):
        raise RuntimeError('%d results for %d images' % (len(results), len(datas)))
    return results


def dataseries_from_result(infer, result):
    """infer.get_dataseries(img, to_clean=False, return_masks=True) with the forward replaced by `result`."""
    orig = infer.do_instance
    infer.do_instance = lambda model, img, score_thr=0.3: infer.parse_result(result, score_thr)
    if not hasattr(infer, 'model'):
        infer.model = None
    try:
        return infer.get_dataseries(None, to_clean=False, return_masks=True)
    finally:
        infer.do_instance = orig


def line_sort_key(line, score):
    """Order key of one line: (leftmost x, mean y of its points, -score of its instance). Empty lines last."""
    if not line:
        return (math.inf, math.inf, -float(score))
    return (min(p['x'] for p in line), sum(p['y'] for p in line) / len(line), -float(score))


def box_sort_key(box):
    """Order key of an instance without a line: (box x1, box centre y, -score)."""
    return (float(box[0]), 0.5 * (float(box[1]) + float(box[3])), -float(box[4]))


def instance_order(boxes, labels, lines, line_thr=LINE_THR):
    """The output order (see ORDER in the module docstring) -> (instance permutation, line permutation).

    boxes (N, 5), labels (N,) as split_result gives them, lines as infer.get_dataseries returned them for the same
    result (one line per class-0 instance with score > line_thr, in instance order). The instances behind the lines
    come first, in line order; the others follow by box_sort_key. Exact ties keep the model's order."""
    import numpy as np
    boxes = np.asarray(boxes).reshape(-1, 5)
    labels = np.asarray(labels).reshape(-1)
    line_idx = [int(i) for i in np.nonzero((boxes[:, 4] > line_thr) & (labels == 0))[0]]
    if len(line_idx) != len(lines):
        raise RuntimeError('%d lines for %d class-0 instances with score > %g' % (len(lines), len(line_idx), line_thr))
    lperm = sorted(range(len(lines)), key=lambda k: line_sort_key(lines[k], boxes[line_idx[k], 4]) + (k,))
    taken = set(line_idx)
    rest = sorted((i for i in range(len(boxes)) if i not in taken), key=lambda i: box_sort_key(boxes[i]) + (i,))
    return [line_idx[k] for k in lperm] + rest, lperm


def ds_hashes(ds):
    import json
    s = json.dumps(ds, sort_keys=True)
    lines = sorted(json.dumps(line, sort_keys=True) for line in ds)
    return hashlib.sha1(s.encode()).hexdigest()[:16], hashlib.sha1('\n'.join(lines).encode()).hexdigest()[:16]


def split_result(result):
    """mmdet 2.x instance result (bbox_results, mask_results) -> boxes (N, 5), labels (N,), masks (list, may hold
    None for masks that were not transferred)."""
    import numpy as np
    bbox_res, mask_res = result[0], result[1]
    boxes, labels, masks = [], [], []
    for cls_i, (b, m) in enumerate(zip(bbox_res, mask_res)):
        b = np.asarray(b).reshape(-1, 5)
        if len(m) != len(b):
            raise ValueError('class %d: %d boxes but %d masks' % (cls_i, len(b), len(m)))
        boxes.append(b)
        labels.extend([cls_i] * len(b))
        masks.extend(m)
    boxes = np.concatenate(boxes, 0) if boxes else np.zeros((0, 5), np.float32)
    return boxes, np.asarray(labels, dtype=np.int64), masks


# ------------------------------------------------------------------ result transfer (shared memory)

def shm_create(nbytes):
    from multiprocessing import resource_tracker, shared_memory
    shm = shared_memory.SharedMemory(create=True, size=max(1, nbytes))
    try:  # the consumer unlinks it; keep the creator's tracker from unlinking or warning (py < 3.13)
        resource_tracker.unregister(shm._name, 'shared_memory')
    except Exception:
        pass
    return shm


def pack_result(result, transfer, thr=LINE_THR):
    """mmdet instance result (bbox_results, mask_results) -> picklable dict, masks in one shm block.
    transfer 'all': every mask; 'kept': only the masks infer.parse_result keeps (score > thr), others None."""
    import numpy as np
    bbox_results, mask_results = result
    keep = []
    for b, ms in zip(bbox_results, mask_results):
        if len(ms) != len(b):
            raise RuntimeError('%d boxes but %d masks' % (len(b), len(ms)))
        if transfer == 'all':
            keep.append(list(range(len(ms))))
        elif transfer == 'kept':
            keep.append([int(i) for i in np.nonzero(b[:, 4] > thr)[0]])
        else:
            raise ValueError(transfer)
    sel = [ms[j] for ms, k in zip(mask_results, keep) for j in k]
    hw = tuple(sel[0].shape) if sel else (0, 0)
    for m in sel:
        if m.shape != hw or m.dtype != bool:
            raise RuntimeError('mask %s %s, expected %s bool' % (m.shape, m.dtype, hw))
    n = len(sel)
    shm = shm_create(n * hw[0] * hw[1])
    arr = np.ndarray((n,) + hw, dtype=bool, buffer=shm.buf)
    for i, m in enumerate(sel):
        arr[i] = m
    del arr
    name = shm.name
    shm.close()
    n_masks = [len(ms) for ms in mask_results]
    return dict(bbox=bbox_results, keep=keep, n_masks=n_masks, shm=name, shape=(n,) + hw)


def unpack_result(p):
    """-> (bbox_results, mask_results) as mmdet returns it; masks not transferred are None."""
    import numpy as np
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=p['shm'])
    try:
        arr = np.array(np.ndarray(tuple(p['shape']), dtype=bool, buffer=shm.buf))  # copy, then free the block
    finally:
        shm.close()
        shm.unlink()
    mask_results, k = [], 0
    for n, keep in zip(p['n_masks'], p['keep']):
        ms = [None] * n
        for j in keep:
            ms[j] = arr[k]
            k += 1
        mask_results.append(ms)
    return p['bbox'], mask_results


def pack_tiles(tiles):
    """tiles: list of (box (y0, x0, y1, x1), scores (n,), masks (n, h, w) bool) -> picklable dict, masks of all
    crops in one shm block."""
    import numpy as np
    sizes = [int(np.asarray(m).size) for _, _, m in tiles]
    shm = shm_create(sum(sizes))
    off, items = 0, []
    for (box, scores, masks), n in zip(tiles, sizes):
        masks = np.asarray(masks, dtype=bool)
        if n:
            np.ndarray(masks.shape, dtype=bool, buffer=shm.buf, offset=off)[...] = masks
        items.append((tuple(int(v) for v in box), np.asarray(scores, dtype=np.float64), tuple(masks.shape), off))
        off += n
    name = shm.name
    shm.close()
    return dict(shm=name, tiles=items)


def unpack_tiles(p):
    import numpy as np
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=p['shm'])
    try:
        out = [(box, scores, np.array(np.ndarray(shape, dtype=bool, buffer=shm.buf, offset=off))
                if int(np.prod(shape)) else np.zeros(shape, dtype=bool))
               for box, scores, shape, off in p['tiles']]
    finally:
        shm.close()
        shm.unlink()
    return out


def free_packed(p):
    """Unlink the shm block of a packed result that will not be unpacked (errors, abandoned images)."""
    if not p or 'shm' not in p:
        return
    from multiprocessing import shared_memory
    try:
        shm = shared_memory.SharedMemory(name=p['shm'])
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass


def tile_crop_result(res, h, w, score_thr):
    """One crop's detector result -> (scores (k,), masks (k, h, w) bool) of its instances with score >= score_thr,
    as tiling.run_tiled selects them."""
    import numpy as np
    bb, mm = res[0][0], res[1][0]
    keep = np.nonzero(np.asarray(bb)[:, 4] >= score_thr)[0] if len(bb) else np.zeros(0, int)
    masks = np.array([np.asarray(mm[k], dtype=bool) for k in keep]).reshape(-1, h, w)
    return (np.asarray(bb)[keep, 4] if len(keep) else np.zeros(0)), masks


# ================================================================== options

def parse_input_size(v):
    """'config' | 'native' | N (int or digit string) -> 'config' | 'native' | int."""
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ('config', 'native'):
            return s
        if s.isdigit() and int(s) > 0:
            return int(s)
    elif isinstance(v, int) and not isinstance(v, bool) and v > 0:
        return v
    raise ValueError("input size must be 'config', 'native' or a positive integer, got %r" % (v,))


def parse_mem_budget(v):
    """0 < fraction <= 1 (float or string), or an absolute size '<n>G' / '<n>M' -> ('frac', x) | ('bytes', n)."""
    if isinstance(v, str):
        s = v.strip().upper().rstrip('B').rstrip('I')
        for suf, mul in (('G', 2 ** 30), ('M', 2 ** 20)):
            if s.endswith(suf):
                n = float(s[:-1])
                if n <= 0:
                    raise ValueError('memory budget must be positive, got %r' % v)
                return ('bytes', int(n * mul))
        v = float(s)
    v = float(v)
    if not 0.0 < v <= 1.0:
        raise ValueError('memory budget must be a fraction in (0, 1] or a size like "4G", got %r' % (v,))
    return ('frac', v)


def resolve_device(device):
    """'auto' -> 'cuda:0' if PyTorch sees a GPU (CUDA or ROCm), else 'cpu'; any other value is returned unchanged.
    The probe runs in a short-lived subprocess, so the calling process never initialises the GPU runtime."""
    if device != 'auto':
        return device
    import subprocess
    r = subprocess.run([sys.executable, '-c', 'import torch; print(int(torch.cuda.is_available()))'],
                       capture_output=True, text=True, timeout=600)
    out = r.stdout.strip().splitlines()
    if r.returncode != 0 or not out or out[-1] not in ('0', '1'):
        raise RuntimeError('device auto: probing torch.cuda.is_available() failed:\n%s'
                           % (r.stderr or r.stdout)[-2000:])
    return 'cuda:0' if out[-1] == '1' else 'cpu'


@dataclass
class ModelOptions:
    ckpt: str
    config: str = DEFAULT_CONFIG
    device: str = 'cuda:0'
    msda: Optional[str] = None
    kept_thr: Optional[float] = None
    input_size: Union[str, int] = 'config'
    tile: Optional[int] = None
    tile_overlap: int = 128
    tile_link_thr: float = 0.5
    tile_min_px: int = 20

    def resolved(self):
        """Validated copy (absolute paths, parsed size); raises ValueError on bad options."""
        o = ModelOptions(**asdict(self))
        o.ckpt = str(Path(o.ckpt).expanduser().resolve())
        o.config = str(Path(o.config).expanduser().resolve())
        for p in (o.ckpt, o.config):
            if not os.path.isfile(p):
                raise ValueError('file not found: %s' % p)
        o.device = resolve_device(o.device)
        if o.msda not in (None, 'auto', 'compiled', 'pytorch'):
            raise ValueError('msda must be auto, compiled or pytorch, got %r' % (o.msda,))
        if o.kept_thr is not None:
            if isinstance(o.kept_thr, bool) or not 0.0 < float(o.kept_thr) < 1.0:
                raise ValueError('kept_thr must be None (off) or a threshold in (0, 1), got %r' % (o.kept_thr,))
            o.kept_thr = float(o.kept_thr)
        if o.tile is not None:
            if o.input_size not in ('native', 'config') and parse_input_size(o.input_size) != 'native':
                raise ValueError('tiling runs the crops at native resolution; input_size %r is not allowed with '
                                 'tile (leave it unset or "native")' % (o.input_size,))
            o.input_size = 'native'
            o.tile, o.tile_overlap = int(o.tile), int(o.tile_overlap)
            if o.tile <= 0 or not 0 <= o.tile_overlap < o.tile:
                raise ValueError('need tile > 0 and 0 <= tile_overlap < tile, got %r, %r' % (o.tile, o.tile_overlap))
        o.input_size = parse_input_size(o.input_size)
        return o

    @property
    def tile_score_thr(self):
        return self.kept_thr if self.kept_thr is not None else LINE_THR


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(mo, ckpt_sha256, config_sha256):
    """What decides the outputs (skip-if-done compares it). Device and MSDA path are not in it: they give
    equivalent results (tools/equivalence acceptance); they are recorded in the manifest."""
    fp = {'engine': jobs.ENGINE_VERSION, 'ckpt_sha256': ckpt_sha256, 'config_sha256': config_sha256,
          'input_size': mo.input_size, 'kept_thr': mo.kept_thr, 'line_thr': LINE_THR, 'tile': None}
    if mo.tile is not None:
        fp['tile'] = {'size': mo.tile, 'overlap': mo.tile_overlap, 'link_thr': mo.tile_link_thr,
                      'min_px': mo.tile_min_px, 'score_thr': mo.tile_score_thr}
    return fp


PACKAGES = ('lineformer', 'torch', 'torchvision', 'mmcv', 'mmdet', 'numpy', 'opencv-python',
            'opencv-python-headless', 'scipy', 'scikit-image', 'matplotlib')


def git_state(path=HERE):
    """-> {'commit': HEAD or None, 'dirty': True/False (tracked files modified) or None} of the checkout at path.
    None when path is not a git checkout (e.g. an installed copy) or git is missing."""
    import subprocess

    def run(*args):
        r = subprocess.run(['git', '-C', str(path)] + list(args), capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        return r.stdout.strip()
    try:
        commit = run('rev-parse', 'HEAD') or None
    except Exception:
        return {'commit': None, 'dirty': None}
    try:
        dirty = bool(run('status', '--porcelain', '--untracked-files=no'))
    except Exception:
        dirty = None
    return {'commit': commit, 'dirty': dirty}


def versions():
    """Package versions without importing torch / mmcv in this process; the fork's git commit."""
    from importlib import metadata
    out = {'python': sys.version.split()[0], 'platform': platform.platform(), 'engine': jobs.ENGINE_VERSION}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None  # not installed (e.g. one of the two OpenCV wheels)
        except Exception as e:
            out[name] = 'unavailable: %r' % e
    g = git_state()
    out['lineformer_git'], out['lineformer_git_dirty'] = g['commit'], g['dirty']
    return out


# ================================================================== worker processes

def _pipeline_cfg(mo):
    import scale_compat
    return scale_compat.build_config(mo['config'], mo['input_size'])


def pre_worker(mo, threads, task_q, pre_q, stat_q=None):
    worker_signals()
    cpu_only_torch()
    import cv2
    import numpy as np
    import tiling
    apply_threads(threads)
    pipeline = build_test_pipeline(_pipeline_cfg(mo))
    if stat_q is not None:
        stat_q.put(('ready_cpu', 'pre', os.getpid(), {}))
    while True:
        rec = task_q.get()
        if rec is None:
            break
        rec['t_pre_start'] = time.time()
        datas = None
        try:
            if not os.path.isfile(rec['path']):
                raise FileNotFoundError('image not found: %s' % rec['path'])
            img = cv2.imread(rec['path'])
            if img is None:
                raise ValueError('cannot read image (cv2.imread returned None): %s' % rec['path'])
            rec['image_sha256'] = sha256_file(rec['path'])
            rec['shape'] = list(img.shape)
            if mo['tile']:
                boxes = tiling.crop_boxes(img.shape[0], img.shape[1], mo['tile'], mo['tile_overlap'])
                datas = [preprocess(pipeline, np.ascontiguousarray(img[y0:y1, x0:x1])) for y0, x0, y1, x1 in boxes]
                rec['tiles'] = boxes
            else:
                datas = [preprocess(pipeline, img)]
            rec['tensor_shapes'] = [list(data_shape(d)) for d in datas]
        except Exception:
            rec['error'] = traceback.format_exc()
            datas = None
        rec['t_pre_end'] = time.time()
        pre_q.put((rec, datas))


def _is_oom(e):
    import torch
    oom = getattr(torch, 'OutOfMemoryError', None) or getattr(torch.cuda, 'OutOfMemoryError', None)
    return (oom is not None and isinstance(e, oom)) or 'out of memory' in str(e).lower()


def _mem_info(device):
    import torch
    if not str(device).startswith('cuda'):
        return {}
    free, total = torch.cuda.mem_get_info(torch.device(device))
    return {'device_free_MB': free / 2 ** 20, 'device_total_MB': total / 2 ** 20,
            'allocated_MB': torch.cuda.memory_allocated() / 2 ** 20,
            'reserved_MB': torch.cuda.memory_reserved() / 2 ** 20,
            'max_allocated_MB': torch.cuda.max_memory_allocated() / 2 ** 20,
            'max_reserved_MB': torch.cuda.max_memory_reserved() / 2 ** 20}


def build_worker_model(mo):
    """The model of one GPU worker: scale_compat.build_model + kept_queries.configure (explicit, never the
    environment). Returns (model, info)."""
    import kept_queries
    import msda_compat
    import scale_compat
    model = scale_compat.build_model(mo['config'], mo['ckpt'], mo['device'], mo['input_size'], msda=mo['msda'])
    thr = kept_queries.configure(model, mo['kept_thr'] if mo['kept_thr'] is not None else False)
    if thr != mo['kept_thr'] or kept_queries.get_threshold(model) != mo['kept_thr']:
        raise RuntimeError('kept-queries threshold %r, asked for %r' % (thr, mo['kept_thr']))
    devs = sorted(set(str(p.device) for p in model.parameters()))
    if str(mo['device']).startswith('cuda') and not all(d.startswith('cuda') for d in devs):
        raise RuntimeError('asked for %s, parameters on %s' % (mo['device'], devs))
    return model, {'param_devices': devs, 'msda_path': msda_compat.get_msda_path(), 'kept_thr': thr,
                   'input_size': getattr(model, 'lineformer_input_size', None)}


def gpu_worker(wid, mo, budget, n_workers, threads, pre_q, post_q, stat_q):
    worker_signals()
    try:
        device = mo['device']
        if not str(device).startswith('cuda'):
            cpu_only_torch()
        import torch
        info = {}
        if str(device).startswith('cuda'):
            dev = torch.device(device)
            total = torch.cuda.get_device_properties(dev).total_memory
            kind, val = budget
            frac = (val if kind == 'frac' else val / total) / n_workers
            if not 0.0 < frac <= 1.0:
                raise RuntimeError('memory budget %r for %d workers gives a fraction %.3f per worker'
                                   % (budget, n_workers, frac))
            torch.cuda.set_per_process_memory_fraction(frac, dev)
            info.update(mem_fraction=frac, mem_cap_MB=frac * total / 2 ** 20,
                        device_name=torch.cuda.get_device_name(dev))
        apply_threads(threads)
        model, minfo = build_worker_model(mo)
        info.update(minfo)
        info.update(_mem_info(device))
        stat_q.put(('ready', wid, os.getpid(), info))
        tile_thr = mo['tile_score_thr']
        n_img, t_busy, last_stat = 0, 0.0, time.time()
        while True:
            try:
                item = pre_q.get(timeout=1.0)
            except queue_mod.Empty:
                item = 'idle'
            if time.time() - last_stat > 2.0:
                stat_q.put(('stats', wid, os.getpid(), dict(_mem_info(device), n_images=n_img, t_busy_s=t_busy)))
                last_stat = time.time()
            if item == 'idle':
                continue
            if item is None:
                break
            rec, datas = item
            if datas is None:
                post_q.put(rec)
                continue
            t0 = time.time()
            try:
                results = [forward(model, [d], check_no_pad=False)[0] for d in datas]
                sync(device)
                t1 = time.time()
                if rec.get('tiles'):
                    tiles = []
                    for box, res in zip(rec['tiles'], results):
                        scores, masks = tile_crop_result(res, box[2] - box[0], box[3] - box[1], tile_thr)
                        tiles.append((box, scores, masks))
                    rec['packed_tiles'] = pack_tiles(tiles)
                else:
                    rec['packed'] = pack_result(results[0], 'all' if rec['masks'] else 'kept')
                rec['t_gpu_start'], rec['t_gpu_end'], rec['t_pack_end'] = t0, t1, time.time()
            except Exception as e:
                if _is_oom(e):
                    raise  # the configuration does not fit: the engine fails instead of failing image after image
                rec['error'] = traceback.format_exc()
            rec['gpu_worker'] = wid
            n_img += 1
            t_busy += time.time() - t0
            post_q.put(rec)
        stat_q.put(('done', wid, os.getpid(), dict(_mem_info(device), n_images=n_img, t_busy_s=t_busy)))
    except BaseException:
        stat_q.put(('fatal', wid, os.getpid(), traceback.format_exc()))


def _savez_atomic(path, **arrays):
    import numpy as np
    path = str(path)
    tmp = '%s.tmp%d' % (path, os.getpid())
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def write_outputs(infer, rec, result, tile_info=None):
    """Lines (+ optional instances / masks) of one image -> files, in the output order (ORDER); the JSON is
    returned, not written: the caller writes it last (it marks the image done). -> (json path, json object)."""
    import numpy as np
    boxes, labels, masks = split_result(result)
    sel = boxes[:, 4] > LINE_THR
    if any(s and m is None for s, m in zip(sel.tolist(), masks)):
        raise RuntimeError('a mask above the line threshold was not transferred')
    t = time.time()
    ds, _ = dataseries_from_result(infer, result)
    rec['t_lines_s'] = time.time() - t
    perm, lperm = instance_order(boxes, labels, ds)
    perm = np.asarray(perm, dtype=np.int64)
    boxes, labels, masks, sel = boxes[perm], labels[perm], [masks[i] for i in perm], sel[perm]
    ds = [ds[k] for k in lperm]
    H, W = rec['shape'][:2]
    paths = jobs.output_paths(rec['out'], rec['id'])
    written = []
    if rec['instances']:
        _savez_atomic(paths['instances'], boxes=np.asarray(boxes, dtype=np.float32),
                      labels=np.asarray(labels, dtype=np.int64))
        written.append('instances')
    if rec['masks']:
        if any(m is None for m in masks):
            raise RuntimeError('masks requested but not all were transferred')
        marr = np.stack([np.asarray(m, dtype=bool) for m in masks], 0) if masks else np.zeros((0, H, W), bool)
        if marr.shape[1:] != (H, W):
            raise RuntimeError('mask shape %s != image %s' % (marr.shape[1:], (H, W)))
        _savez_atomic(paths['masks'], masks_packed=np.packbits(marr.reshape(-1)),
                      mask_shape=np.asarray(marr.shape, dtype=np.int64))
        written.append('masks')
    for k in jobs.OUTPUT_KINDS:  # an older run's file of a kind not asked for now would not match this JSON
        if k not in written and paths[k].exists():
            os.remove(str(paths[k]))
    lines =[[{'x': float(q['x']), 'y': float(q['y'])} for q in ln] for ln in ds]
    rec['n_lines'] = len(lines)
    rec['n_instances'] = int(len(boxes))
    timings = {'pre_s': rec['t_pre_end'] - rec['t_pre_start'],
               'gpu_s': rec['t_gpu_end'] - rec['t_gpu_start'],
               'post_s': None}  # filled by the post worker
    obj = {'id': rec['id'], 'image': rec['path'], 'image_sha256': rec['image_sha256'], 'shape': rec['shape'],
           'lines': lines, 'n_lines': len(lines), 'n_instances': rec['n_instances'],
           'n_instances_gt_line_thr': int(sel.sum()), 'order': jobs.ORDER, 'outputs': written,
           'fingerprint': rec['fingerprint'],
           'job': rec['job'], 'timings': timings}
    if tile_info is not None:
        obj['tiles'] = tile_info
    return paths['json'], obj


def post_worker(mo, threads, post_q, done_q, stat_q=None):
    worker_signals()
    cpu_only_torch()
    apply_threads(threads)
    import infer
    import tiling
    if stat_q is not None:
        stat_q.put(('ready_cpu', 'post', os.getpid(), {}))
    while True:
        rec = post_q.get()
        if rec is None:
            break
        rec['t_post_start'] = time.time()
        packed, ptiles = rec.pop('packed', None), rec.pop('packed_tiles', None)
        try:
            if rec.get('error'):
                free_packed(packed)
                free_packed(ptiles)
                raise RuntimeError('stage error')
            tile_info = None
            if ptiles is not None:
                tiles = unpack_tiles(ptiles)
                H, W = rec['shape'][:2]
                bboxes, masks, info = tiling.merge_instances(tiles, H, W, mo['tile_score_thr'], mo['tile_link_thr'],
                                                             mo['tile_min_px'])
                result = ([bboxes], [masks])
                tile_info = {'n_tiles': len(tiles), 'boxes': [list(t[0]) for t in tiles],
                             'n_crop_instances': [int(len(t[1])) for t in tiles], 'n_links': len(info['links'])}
            elif packed is not None:
                result = unpack_result(packed)
            else:
                raise RuntimeError('no detector result')
            jpath, obj = write_outputs(infer, rec, result, tile_info)
            rec['t_post_end'] = time.time()
            obj['timings']['post_s'] = rec['t_post_end'] - rec['t_post_start']
            jobs.write_json_atomic(jpath, obj)
            rec['outputs_sha256'] = jobs.outputs_sha256(rec['out'], rec['id'], obj['outputs'])
            rec['timings'] = obj['timings']
            rec['status'] = 'done'
        except Exception:
            rec['status'] = 'failed'
            if not rec.get('error'):
                rec['error'] = traceback.format_exc()
        rec.setdefault('t_post_end', time.time())
        done_q.put({k: rec.get(k) for k in ('job', 'idx', 'status', 'error', 'image_sha256', 'shape', 'n_lines',
                                            'n_instances', 'gpu_worker', 'timings', 'outputs_sha256') if k in rec})


# ================================================================== the engine (main process)

class EngineFailed(RuntimeError):
    pass


class Engine:
    """Owns the worker processes and the job queue. Thread-safe: submit / status / cancel / wait may be called
    from any thread (the HTTP server does)."""

    def __init__(self, model_options, gpu_workers=1, gpu_mem_budget=0.85, pre_workers=4, post_workers=4,
                 pre_threads=1, post_threads=1, gpu_threads=2, max_inflight=None, manifest_every_s=2.0,
                 ready_timeout_s=900, log=None):
        self.mo = model_options.resolved()
        for k, v in (('gpu_workers', gpu_workers), ('pre_workers', pre_workers), ('post_workers', post_workers)):
            if int(v) < 1:
                raise ValueError('%s must be >= 1, got %r' % (k, v))
        self.n_gpu, self.n_pre, self.n_post = int(gpu_workers), int(pre_workers), int(post_workers)
        self.threads = {'pre': int(pre_threads), 'post': int(post_threads), 'gpu': int(gpu_threads)}
        self.budget = parse_mem_budget(gpu_mem_budget)
        self.budget_arg = gpu_mem_budget
        self.max_inflight = int(max_inflight or 2 * (self.n_pre + self.n_gpu + self.n_post))
        self.manifest_every_s = manifest_every_s
        self.ready_timeout_s = ready_timeout_s
        self.log = log or (lambda *a: print('[lineformer-engine]', *a, file=sys.stderr, flush=True))
        self.state = 'created'
        self.error = None
        self.sched = jobs.Scheduler()
        self.cond = threading.Condition()
        self.inflight = 0
        self.procs = {'pre': [], 'gpu': [], 'post': []}
        self.worker_info = {}
        self.worker_stats = {}
        self._accepting = False
        self._feeding = False
        self._threads = []
        self._abandon = threading.Event()
        self._stopped = threading.Event()
        self._stopping_workers = False
        self._job_seq = 0
        self.engine_info = None

    # ---------------------------------------------------------- start / stop
    def prepare(self):
        """Fingerprint and versions, no worker processes yet. Jobs can be submitted from here on (skip-if-done is
        decided at submit); start() starts the workers. A caller whose jobs are all done need not start them."""
        if self.state != 'created':
            return self
        mo = self.mo
        self.ckpt_sha256 = sha256_file(mo.ckpt)
        self.config_sha256 = sha256_file(mo.config)
        self.fingerprint = fingerprint(mo, self.ckpt_sha256, self.config_sha256)
        self.versions = versions()
        self.engine_info = {
            'model_options': asdict(mo), 'fingerprint': self.fingerprint, 'versions': self.versions,
            'device': mo.device, 'msda': mo.msda, 'msda_env': os.environ.get('LINEFORMER_MSDA'), 'msda_path': None,
            'kept_thr': mo.kept_thr, 'input_size': mo.input_size,
            'tile': self.fingerprint['tile'], 'line_thr': LINE_THR, 'order': jobs.ORDER, 'gpu_workers': self.n_gpu,
            'pre_workers': self.n_pre, 'post_workers': self.n_post, 'gpu_mem_budget': self.budget_arg,
            'threads': self.threads, 'max_inflight': self.max_inflight, 'workers': None, 'pids': None}
        self.state = 'prepared'
        self._accepting = True
        return self

    def start(self):
        import multiprocessing as mp
        self.prepare()
        if self.state != 'prepared':
            raise RuntimeError('engine already started (%s)' % self.state)
        self.state = 'starting'
        mo = self.mo
        ctx = mp.get_context('spawn')
        self.task_q = ctx.Queue()
        self.pre_q = ctx.Queue(maxsize=max(4, 2 * self.n_gpu))
        self.post_q = ctx.Queue(maxsize=max(4, 2 * self.n_post))
        self.done_q, self.stat_q = ctx.Queue(), ctx.Queue()
        mod = dict(asdict(mo), tile_score_thr=mo.tile_score_thr)

        def spawn(role, target, args):
            old = {k: os.environ.get(k) for k in THREAD_ENV}
            for k in THREAD_ENV:
                os.environ[k] = str(self.threads[role])
            try:
                p = ctx.Process(target=target, args=args, daemon=True, name='lineformer-%s-%d'
                                % (role, len(self.procs[role])))
                p.start()
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            self.procs[role].append(p)

        self.log('starting %d GPU worker(s) on %s (input size %s, kept_thr %s, tile %s, memory budget %s)' % (
            self.n_gpu, mo.device, mo.input_size, mo.kept_thr, mo.tile, self.budget_arg))
        for w in range(self.n_gpu):
            spawn('gpu', gpu_worker, (w, mod, self.budget, self.n_gpu, self.threads['gpu'], self.pre_q, self.post_q,
                                      self.stat_q))
        # the CPU workers import mmcv / mmdet while the models load; feeding starts when every worker is ready, so
        # a job's clock does not include start-up
        for _ in range(self.n_pre):
            spawn('pre', pre_worker, (mod, self.threads['pre'], self.task_q, self.pre_q, self.stat_q))
        for _ in range(self.n_post):
            spawn('post', post_worker, (mod, self.threads['post'], self.post_q, self.done_q, self.stat_q))
        t0 = time.time()
        n_cpu_ready = 0
        while len(self.worker_info) < self.n_gpu or n_cpu_ready < self.n_pre + self.n_post:
            try:
                kind, wid, pid, info = self.stat_q.get(timeout=1.0)
            except queue_mod.Empty:
                dead = [(r, p.pid) for r, ps in self.procs.items() for p in ps if not p.is_alive()]
                if dead or time.time() - t0 > self.ready_timeout_s:
                    self._start_failed('worker(s) %s before ready' % (
                        'died (%s)' % dead if dead else 'not ready after %d s' % self.ready_timeout_s))
                continue
            if kind == 'fatal':
                self._start_failed('GPU worker %d failed while loading:\n%s' % (wid, info))
            if kind == 'ready_cpu':
                n_cpu_ready += 1
            if kind == 'ready':
                self.worker_info[wid] = dict(info, pid=pid, ready_s=time.time() - t0)
                self.log('GPU worker %d ready (pid %d, MSDA %s, %.1f s)' % (wid, pid, info.get('msda_path'),
                                                                          time.time() - t0))
        self.log('all workers ready after %.1f s' % (time.time() - t0))
        if 'device_free_MB' in self.worker_info[0]:
            grow = sum(i.get('mem_cap_MB', 0) - i.get('reserved_MB', 0) for i in self.worker_info.values())
            free = min(i['device_free_MB'] for i in self.worker_info.values())
            if grow > free:
                self.log('WARNING: the GPU workers may still grow by %.0f MB, the device had %.0f MB free after '
                         'loading; an out-of-memory error ends the engine' % (grow, free))
        with self.cond:
            if self.state != 'starting':  # shutdown() was called while the models loaded
                self.log('start: engine is %s, not feeding' % self.state)
                return self
            self.engine_info = dict(
                self.engine_info, msda_path=self.worker_info[0].get('msda_path'),
                workers={str(k): {kk: v.get(kk) for kk in ('pid', 'param_devices', 'msda_path', 'mem_fraction',
                                                           'mem_cap_MB', 'device_name', 'ready_s')}
                         for k, v in self.worker_info.items()},
                pids={r: [p.pid for p in ps] for r, ps in self.procs.items()})
            self._accepting = self._feeding = True
            self.state = 'running'
            for j in self.sched.active():  # jobs submitted before start: their manifests get the worker info
                j.dirty = True
        for target, name in ((self._feeder, 'feeder'), (self._collector, 'collector')):
            t = threading.Thread(target=target, name='lineformer-' + name, daemon=True)
            t.start()
            self._threads.append(t)
        self.log('engine running: %d pre, %d GPU, %d post workers' % (self.n_pre, self.n_gpu, self.n_post))
        return self

    def __enter__(self):
        return self.start() if self.state == 'created' else self

    def __exit__(self, *exc):
        self.shutdown()

    def shutdown(self, timeout=600.0, abandon=False):
        """Stop: no new images; in-flight images finish within timeout (abandon=True: do not wait); unfinished
        jobs end 'interrupted' (manifests written); worker processes are stopped. Idempotent."""
        with self.cond:
            if self.state in ('stopped',):
                return
            already = self.state == 'stopping'
            self._accepting = False
            self._feeding = False
            if self.state not in ('failed',):
                self.state = 'stopping'
            self.cond.notify_all()
        if abandon:
            self._abandon.set()
        if already:
            self._stopped.wait(timeout)
            return
        if self.state == 'stopping' and self.procs['gpu']:
            t_end = time.time() + timeout
            with self.cond:
                while self.inflight > 0 and not self._abandon.is_set() and time.time() < t_end \
                        and self.state == 'stopping':
                    self.cond.wait(0.2)
                left = self.inflight
            if left:
                self.log('shutdown: abandoning %d in-flight image(s)' % left)
        with self.cond:
            for j in self.sched.active():
                j.stop('interrupted', 'engine stopped before the job finished')
            self._write_manifests(force=True)
        self._stop_workers(graceful=self.inflight == 0 and self.state != 'failed')
        with self.cond:
            if self.state != 'failed':
                self.state = 'stopped'
            self.cond.notify_all()
        self._stopped.set()

    def _stop_workers(self, graceful=True):
        self._stopping_workers = True
        procs = self.procs
        if graceful:
            try:
                for _ in procs['pre']:
                    self.task_q.put(None)
                for p in procs['pre']:
                    p.join(30)
                for _ in procs['gpu']:
                    self.pre_q.put(None, timeout=10)
                for p in procs['gpu']:
                    p.join(60)
                for _ in procs['post']:
                    self.post_q.put(None, timeout=10)
                for p in procs['post']:
                    p.join(60)
            except Exception as e:
                self.log('graceful worker stop failed (%r); terminating' % e)
        self._kill_all()

    def _kill_all(self):
        for ps in self.procs.values():
            for p in ps:
                if p.is_alive():
                    p.terminate()
        for ps in self.procs.values():
            for p in ps:
                p.join(10)
                if p.is_alive():
                    p.kill()
        # results of abandoned images still in the queue: free their shared-memory blocks
        for q in (getattr(self, 'post_q', None),):
            while q is not None:
                try:
                    item = q.get_nowait()
                except Exception:
                    break
                if isinstance(item, dict):
                    free_packed(item.get('packed'))
                    free_packed(item.get('packed_tiles'))

    def _start_failed(self, msg):
        with self.cond:
            self._fail(msg)
        self._kill_all()
        self._stopped.set()
        raise EngineFailed(msg)

    def _fail(self, msg):
        """Called with self.cond held."""
        if self.state == 'failed':
            return
        self.log('ENGINE FAILED: %s' % msg)
        self.state = 'failed'
        self.error = msg
        self._accepting = self._feeding = False
        for j in self.sched.active():
            j.stop('failed', msg)
        self._write_manifests(force=True)
        self.cond.notify_all()
        threading.Thread(target=self._kill_all, daemon=True).start()

    # ---------------------------------------------------------- threads
    def _feeder(self):
        while True:
            with self.cond:
                while True:
                    if not self._feeding or self.state != 'running':
                        return
                    if self.inflight < self.max_inflight:
                        job, rec = self.sched.next_task()
                        if rec is not None:
                            break
                    self.cond.wait(0.5)
                self.inflight += 1
                task = {'job': job.id, 'idx': rec['idx'], 'id': rec['id'], 'path': rec['path'], 'out': job.out,
                        'instances': job.outputs['instances'], 'masks': job.outputs['masks'],
                        'fingerprint': self.fingerprint}
            self.task_q.put(task)

    def _collector(self):
        last_write = 0.0
        while True:
            try:
                res = self.done_q.get(timeout=0.5)
            except queue_mod.Empty:
                res = None
            except (EOFError, OSError):
                return
            with self.cond:
                self._drain_stats()
                if res is not None:
                    self.inflight -= 1
                    job = self.sched.jobs.get(res['job'])
                    if job is None or job.final:
                        pass  # the job ended (engine stop) before this image came back; its outputs are on disk
                    else:
                        job.finish_record(res)
                        if job.final:
                            job.write_manifest(self._info())
                            self.log('job %s %s: %s' % (job.id, job.status, _fmt_counts(job.counts())))
                    self.cond.notify_all()
                if self.state in ('running', 'stopping') and not self._stopping_workers:
                    dead = [(r, p.pid, p.exitcode) for r, ps in self.procs.items() for p in ps if not p.is_alive()]
                    if dead and self.state == 'running':
                        self._fail('worker process(es) died: %s' % ', '.join('%s pid %d exit %s' % d for d in dead))
                    elif dead and not self._abandon.is_set():
                        self.log('worker process(es) died while stopping: %s; abandoning in-flight images' % dead)
                        self._abandon.set()
                if time.time() - last_write > self.manifest_every_s:
                    self._write_manifests()
                    last_write = time.time()
                if self.state in ('stopped', 'failed') or (self.state == 'stopping' and self._stopped.is_set()):
                    return

    def _drain_stats(self):
        while True:
            try:
                kind, wid, pid, info = self.stat_q.get_nowait()
            except queue_mod.Empty:
                return
            except (EOFError, OSError):
                return
            if kind == 'fatal':
                self._fail('GPU worker %d (pid %d) failed:\n%s' % (wid, pid, info))
            elif kind in ('stats', 'done'):
                self.worker_stats[wid] = dict(info, t=time.time(), pid=pid)

    def _info(self):
        """engine_info + the GPU workers' last memory statistics (at most ~2 s old)."""
        stats = {str(k): {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                 for k, v in self.worker_stats.items()}
        return dict(self.engine_info or {}, worker_stats=stats)

    def _write_manifests(self, force=False):
        for j in self.sched.jobs.values():
            if j.dirty or force:
                try:
                    j.write_manifest(self._info())
                except Exception as e:
                    self.log('cannot write manifest of job %s: %r' % (j.id, e))

    # ---------------------------------------------------------- API
    def submit(self, images, out, outputs=None, force=False, priority=0, ids='stem', name=None,
               require_absolute=False):
        """Queue a job; returns its id. images: paths or {"id", "path"} dicts. Raises JobError on bad input,
        OutDirBusy if an active job writes into `out`, EngineFailed if the engine is not running."""
        if self.state == 'created':
            self.prepare()
        if not self._accepting:
            raise EngineFailed('engine is %s%s' % (self.state, ': ' + self.error if self.error else ''))
        if not isinstance(out, (str, Path)) or not str(out):
            raise jobs.JobError('out must be a directory path')
        if require_absolute and not os.path.isabs(str(out)):
            raise jobs.JobError('out must be an absolute path: %r' % (out,))
        recs = jobs.normalize_items(images, ids, require_absolute=require_absolute)
        outputs = dict(outputs or {})
        job = jobs.Job('pending', out, recs, outputs=outputs, force=force, priority=priority, name=name)
        Path(job.out).mkdir(parents=True, exist_ok=True)
        for r in recs:
            if r['status'] == 'pending':
                jobs.done_state(r, job.out, self.fingerprint, job.outputs, force=job.force)
        job.pending = type(job.pending)(r['idx'] for r in recs if r['status'] == 'pending')
        with self.cond:
            if not self._accepting:
                raise EngineFailed('engine is %s' % self.state)
            self.sched.check_out_dir(job.out)
            self._job_seq += 1
            job.id = jobs.new_job_id(self._job_seq)
            self.sched.add(job)
            job.maybe_finish()
            job.write_manifest(self._info())
            self.cond.notify_all()
        self.log('job %s queued: %s -> %s' % (job.id, _fmt_counts(job.counts()), job.out))
        return job.id

    def _job(self, job_id):
        j = self.sched.jobs.get(job_id)
        if j is None:
            raise KeyError(job_id)
        return j

    def status(self, job_id, images=False):
        with self.cond:
            j = self._job(job_id)
            s = j.summary()
            if images:
                s['images'] = [jobs._public(r) for r in j.records]
            return s

    def list_jobs(self):
        with self.cond:
            return [{'job': j.id, 'name': j.name, 'status': j.status, 'out': j.out, 'counts': j.counts(),
                     'priority': j.priority, 'created': j.created} for j in self.sched.jobs.values()]

    def cancel(self, job_id):
        with self.cond:
            j = self._job(job_id)
            changed = j.cancel()
            j.write_manifest(self._info())
            self.cond.notify_all()
            return changed, j.summary()

    def wait(self, job_id, timeout=None):
        j = self._job(job_id)
        if not j.event.wait(timeout):
            raise TimeoutError('job %s still %s after %s s' % (job_id, j.status, timeout))
        return self.status(job_id)

    def health(self):
        with self.cond:
            stats = {str(k): v for k, v in self.worker_stats.items()}
            free = [v.get('device_free_MB') for v in self.worker_stats.values() if v.get('device_free_MB')]
            if not free:
                free = [v.get('device_free_MB') for v in self.worker_info.values() if v.get('device_free_MB')]
            return {'state': self.state, 'error': self.error, 'engine': self.engine_info,
                    'device_free_MB': min(free) if free else None, 'worker_stats': stats,
                    'jobs_active': len(self.sched.active()), 'images_pending': self.sched.n_pending(),
                    'images_inflight': self.inflight, 'accepting': self._accepting}


def _fmt_counts(c):
    return ', '.join('%s %d' % (k, v) for k, v in c.items() if v and k != 'total') + ' (of %d)' % c['total']


def load_outputs(out, image_id):
    """Read one image's outputs: dict(lines=..., json=..., boxes, scores, labels if .instances.npz exists,
    masks (N, H, W) bool if .masks.npz exists)."""
    import json
    import numpy as np
    p = jobs.output_paths(out, image_id)
    with open(str(p['json']), encoding='utf-8') as f:
        obj = json.load(f)
    res = {'json': obj, 'lines': obj['lines']}
    if 'instances' in obj.get('outputs', []):
        with np.load(str(p['instances'])) as z:
            res['boxes'] = z['boxes']
            res['scores'] = z['boxes'][:, 4].copy()
            res['labels'] = z['labels']
    if 'masks' in obj.get('outputs', []):
        with np.load(str(p['masks'])) as z:
            shape = tuple(int(v) for v in z['mask_shape'])
            res['masks'] = np.unpackbits(z['masks_packed'], count=int(np.prod(shape))).astype(bool).reshape(shape)
    return res
