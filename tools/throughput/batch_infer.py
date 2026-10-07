"""Throughput runner for LineFormer inference: the same per-image maths as infer.get_dataseries, spread over
processes so that the GPU and the CPU cores work at the same time.

Modes
  getds     the production call, one process: infer.get_dataseries(cv2.imread(path), to_clean=False) per image.
            Timing reference only (it yields no detector output, so it cannot write the harness format).
  serial    one process, the stages below run inline one image after another (same functions as `pipeline`).
  pipeline  E1: --pre-workers processes read the image and run mmdet's test pipeline; --gpu-workers processes
            (E2: N > 1, each holds its own model) only collate + forward; --post-workers processes turn the
            detector result into the harness outputs and the dataseries. Bounded queues between the stages.
            --batch B > 1 (E3): a GPU worker forwards up to B images whose post-pipeline tensor shape is
            identical; a batch with more than one shape, or any image whose img_shape differs from the batch
            input shape (= padding), raises.

The maths is not re-implemented:
  - pre:  the first half of mmdet.apis.inference_detector verbatim (LoadImageFromWebcam, replace_ImageToTensor,
          Compose(cfg.data.test.pipeline)) on mmcv.Config.fromfile(config), as init_detector reads it;
  - gpu:  the second half verbatim (collate, img_metas/img unwrapping, scatter, model(return_loss=False,
          rescale=True));
  - post: infer.get_dataseries itself, with infer.do_instance replaced for the call by
          infer.parse_result(<the detector result of this image>) - the forward is the only thing swapped out;
          the harness files with tools/equivalence/run.split_result and common.save_instances.
  --selfcheck K (CPU) proves the split: for K images the split path is compared bit for bit with
  inference_detector and infer.get_dataseries, and (E3) a batch of two equal-shape images with the single forward.

Transfer between processes: boxes through the queue, masks through POSIX shared memory. --transfer kept sends
only the masks infer.parse_result keeps (score > 0.3, the same expression); the others travel as None and the
post worker raises if one of them is ever selected. --transfer all sends every mask (needed for the .npz). The
output pass (pass 0, --save full) always uses all.

Passes: pass 0 = warm-up + harness output (<out>/<id>.npz, .dataseries.json, .meta.json, run_meta.json, the
format of tools/equivalence/common.py); passes 1..R (--repeat) are timed, write nothing, and record a hash of each
dataseries (exact and line-order-insensitive) against pass 0. Throughput = R * n_images / (time from the first
timed task to the last timed result). CPU: /proc/stat over the timed window (whole WSL VM) and /proc/<pid>/stat
per role; GPU memory per GPU process (max_memory_allocated / reserved over the whole run).

Threads: every child gets OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS, torch.set_num_threads and cv2.setNumThreads set
to its role's --*-threads (defaults 1 for pre/post, 2 for GPU processes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import queue as queue_mod
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
EQ = HERE.parent / "equivalence"
sys.path.insert(0, str(EQ))
import common  # noqa: E402  (numpy only)

THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
SCORE_THR = 0.3  # infer.get_dataseries calls do_instance(..., score_thr=0.3); parse_result keeps score > thr


# ------------------------------------------------------------------ setup helpers

def _setup_repo(repo):
    repo = str(Path(repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.chdir(repo)


def _cpu_only_torch():
    """For the CPU worker processes (pre/post), before mmcv is imported. Under WSL + ROCm 7.2, the first
    torch.cuda.is_available() call (mmcv does it at import) starts the HSA runtime, whose two threads then spin at
    100 % for the life of the process (measured: 4.0 CPU-s per 2 s idle, with or without visible devices). These
    workers never touch a GPU, so the probe answers False here. GPU processes are left alone."""
    import torch
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0


def _apply_threads(n):
    import torch
    import cv2
    torch.set_num_threads(n)
    cv2.setNumThreads(n)


def _sync(device):
    import torch
    if str(device).startswith("cuda"):
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
        raise RuntimeError("expected one test-time augmentation, got %d" % len(imgs))
    return tuple(imgs[0].data.shape)


def forward(model, datas):
    """Second half of mmdet.apis.inference_detector (verbatim), for a list of pre-processed images that must all
    have one tensor shape (so mmcv's collate pads nothing). Returns the list of per-image results."""
    import torch
    from mmcv.parallel import collate, scatter
    shapes = sorted(set(data_shape(d) for d in datas))
    if len(shapes) != 1:
        raise RuntimeError("batch with several tensor shapes %s: collate would pad" % shapes)
    device = next(model.parameters()).device
    data = collate(datas, samples_per_gpu=len(datas))
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)
    # no padding: every image fills the batch input exactly
    for metas in data['img_metas']:
        for m in metas:
            if tuple(m['img_shape'][:2]) != tuple(m['batch_input_shape']) or \
                    tuple(m.get('pad_shape', m['img_shape'])[:2]) != tuple(m['img_shape'][:2]):
                raise RuntimeError("padding in batch: img_shape %s pad_shape %s batch_input_shape %s" % (
                    m['img_shape'], m.get('pad_shape'), m['batch_input_shape']))
    if len(results) != len(datas):
        raise RuntimeError("%d results for %d images" % (len(results), len(datas)))
    return results


def dataseries_from_result(infer, result):
    """infer.get_dataseries(img, to_clean=False, return_masks=True) with the forward replaced by `result`."""
    orig = infer.do_instance
    infer.do_instance = lambda model, img, score_thr=0.3: infer.parse_result(result, score_thr)
    if not hasattr(infer, "model"):
        infer.model = None
    try:
        return infer.get_dataseries(None, to_clean=False, return_masks=True)
    finally:
        infer.do_instance = orig


def ds_hashes(ds):
    s = json.dumps(ds, sort_keys=True)
    lines = sorted(json.dumps(line, sort_keys=True) for line in ds)
    return hashlib.sha1(s.encode()).hexdigest()[:16], hashlib.sha1("\n".join(lines).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ result transfer (shared memory)

def _shm_create(nbytes):
    from multiprocessing import shared_memory, resource_tracker
    shm = shared_memory.SharedMemory(create=True, size=max(1, nbytes))
    try:  # the consumer unlinks it; keep the creator's tracker from unlinking or warning (py < 3.13)
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass
    return shm


def pack_result(result, transfer):
    """mmdet instance result (bbox_results, mask_results) -> picklable dict, masks in one shm block."""
    import numpy as np
    bbox_results, mask_results = result
    keep = []
    for b, ms in zip(bbox_results, mask_results):
        if len(ms) != len(b):
            raise RuntimeError("%d boxes but %d masks" % (len(b), len(ms)))
        if transfer == "all":
            keep.append(list(range(len(ms))))
        elif transfer == "kept":
            keep.append([int(i) for i in np.nonzero(b[:, 4] > SCORE_THR)[0]])
        else:
            raise ValueError(transfer)
    sel = [ms[j] for ms, k in zip(mask_results, keep) for j in k]
    hw = tuple(sel[0].shape) if sel else (0, 0)
    for m in sel:
        if m.shape != hw or m.dtype != bool:
            raise RuntimeError("mask %s %s, expected %s bool" % (m.shape, m.dtype, hw))
    n = len(sel)
    shm = _shm_create(n * hw[0] * hw[1])
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
    shm = shared_memory.SharedMemory(name=p["shm"])
    try:
        arr = np.array(np.ndarray(tuple(p["shape"]), dtype=bool, buffer=shm.buf))  # copy, then free the block
    finally:
        shm.close()
        shm.unlink()
    mask_results, k = [], 0
    for n, keep in zip(p["n_masks"], p["keep"]):
        ms = [None] * n
        for j in keep:
            ms[j] = arr[k]
            k += 1
        mask_results.append(ms)
    return p["bbox"], mask_results


# ------------------------------------------------------------------ post-processing (shared by all modes)

def postprocess(infer, split_result, rec, result, save, out):
    """rec: per-image record (id, path, ...). Computes the dataseries; with save, writes the harness files."""
    import numpy as np
    bbox_results, mask_results = result
    # fail loud if parse_result would select a mask that was not transferred
    sel = (bbox_results[0][:, 4] > SCORE_THR).tolist()
    if any(s and m is None for s, m in zip(sel, mask_results[0])):
        raise RuntimeError("a kept mask was not transferred")
    t = time.time()
    if save:
        boxes, labels, masks = split_result(result)
        H, W = rec["shape"][:2]
        for m in masks:
            if m.shape != (H, W):
                raise ValueError("mask shape %s != image %s" % (m.shape, (H, W)))
        marr = np.stack(masks, 0) if masks else np.zeros((0, H, W), bool)
        paths = common.run_paths(out, rec["id"])
        common.save_instances(paths["npz"], boxes, labels, marr)
        rec["n_instances"] = int(len(boxes))
        rec["n_ge_0.3"] = int((boxes[:, 4] >= 0.3).sum())
        rec["n_gt_0.3"] = int((boxes[:, 4] > 0.3).sum())
    ds, ds_masks = dataseries_from_result(infer, result)
    rec["t_post_compute_s"] = time.time() - t
    rec["n_lines"] = len(ds)
    rec["n_dataseries_masks"] = len(ds_masks)
    rec["ds_hash"], rec["ds_hash_unordered"] = ds_hashes(ds)
    if save:
        paths = common.run_paths(out, rec["id"])
        common.write_json(paths["ds"], ds)
        rec["status"] = "ok"
        common.write_json(paths["meta"], rec)
    rec["status"] = "ok"
    return rec


# ------------------------------------------------------------------ worker processes

def pre_worker(cfg_path, repo, threads, task_q, pre_q):
    _setup_repo(repo)
    _cpu_only_torch()
    import cv2
    import mmcv
    _apply_threads(threads)
    pipeline = build_test_pipeline(mmcv.Config.fromfile(cfg_path))
    while True:
        item = task_q.get()
        if item is None:
            break
        rep, iid, path, save = item
        rec = {"id": iid, "path": path, "pass": rep, "status": "error", "t_pre_start": time.time()}
        data = None
        try:
            img = cv2.imread(path)
            if img is None:
                raise ValueError("cv2.imread returned None for %s" % path)
            if save:
                rec["file_sha256"] = common.sha256_file(path)
                rec["pixels_sha256"] = common.sha256_array(img)
            rec["shape"] = list(img.shape)
            data = preprocess(pipeline, img)
            rec["tensor_shape"] = list(data_shape(data))
        except Exception:
            rec["error"] = traceback.format_exc()
            data = None
        rec["t_pre_end"] = time.time()
        pre_q.put((rec, data))


def _load_model(args, threads):
    _setup_repo(args["repo"])
    import torch
    _apply_threads(threads)
    import infer
    sig_kw = {"msda": args["msda"]} if args["msda"] else {}
    infer.load_model(args["config"], args["ckpt"], args["device"], **sig_kw)
    model = infer.model
    devs = sorted(set(str(p.device) for p in model.parameters()))
    if str(args["device"]).startswith("cuda") and not any(d.startswith("cuda") for d in devs):
        raise RuntimeError("asked for %s, parameters on %s" % (args["device"], devs))
    msda = infer.get_msda_path() if hasattr(infer, "get_msda_path") else None
    _instrument_d2h(model, args["device"])
    return torch, infer, model, devs, msda


D2H_MARK = {"t": None}


def _instrument_d2h(model, device):
    """Pass-through wrapper: synchronise after the fusion head (upsample + scores on the GPU) and record the time,
    so that forward end minus this mark = MaskFormer.simple_test's per-mask .cpu() loop (device->host)."""
    head = getattr(model, "panoptic_fusion_head", None)
    if head is None:
        return
    orig = head.simple_test

    def timed(*a, **k):
        r = orig(*a, **k)
        _sync(device)
        D2H_MARK["t"] = time.time()
        return r
    head.simple_test = timed


def gpu_worker(wid, args, threads, pre_q, post_q, stat_q):
    try:
        if str(args["device"]).startswith("cuda"):
            # All GPU workers together stay inside --gpu-mem-budget of the device. Without a cap each caching
            # allocator kept ~12 GB, two workers overcommitted the 24 GB card and WDDM paged it to host RAM,
            # which stalled the whole machine.
            import torch
            frac = args["gpu_mem_budget"] / args["gpu_workers"]
            torch.cuda.set_per_process_memory_fraction(frac, torch.device(args["device"]))
        torch, infer, model, devs, msda = _load_model(args, threads)
        stat_q.put(("ready", wid, os.getpid(), {"param_devices": devs, "msda_path": msda}))
        B, transfer = args["batch"], args["transfer"]
        pending = {}  # tensor shape -> [(rec, data)]
        n_batches, batch_sizes, t_busy = 0, [], 0.0
        done = False

        def run(shape):
            nonlocal n_batches, t_busy
            items = pending.pop(shape)
            t0 = time.time()
            try:
                D2H_MARK["t"] = None
                results = forward(model, [d for _, d in items])
                _sync(args["device"])
                t1 = time.time()
                for (rec, _), res in zip(items, results):
                    save = rec["pass"] == 0
                    rec["packed"] = pack_result(res, "all" if save else transfer)
                    rec["t_gpu_start"], rec["t_gpu_end"], rec["t_pack_end"] = t0, t1, time.time()
                    rec["t_fusion_end"] = D2H_MARK["t"]
                    rec["batch_size"], rec["gpu_worker"] = len(items), wid
                    post_q.put(rec)
            except torch.cuda.OutOfMemoryError:
                raise  # the configuration does not fit: end the run (fatal) instead of failing batch after batch
            except Exception:
                err = traceback.format_exc()
                for rec, _ in items:
                    rec["error"] = err
                    post_q.put(rec)
            t_busy += time.time() - t0
            n_batches += 1
            batch_sizes.append(len(items))

        while True:
            if done and not pending:
                break
            try:
                item = None if done else pre_q.get(timeout=0.05)
            except queue_mod.Empty:
                item = "idle"
            if item is None:
                done = True
                for s in list(pending):
                    run(s)
                continue
            if item == "idle":
                if pending:  # nothing waiting: run the fullest bucket rather than idle
                    run(max(pending, key=lambda s: len(pending[s])))
                continue
            rec, data = item
            if data is None:
                post_q.put(rec)
                continue
            s = tuple(rec["tensor_shape"])
            pending.setdefault(s, []).append((rec, data))
            if len(pending[s]) >= B:
                run(s)
        stats = {"n_batches": n_batches, "batch_size_hist": {str(k): batch_sizes.count(k)
                                                             for k in sorted(set(batch_sizes))},
                 "t_busy_s": t_busy}
        if str(args["device"]).startswith("cuda"):
            stats["max_memory_allocated_MB"] = torch.cuda.max_memory_allocated() / 2 ** 20
            stats["max_memory_reserved_MB"] = torch.cuda.max_memory_reserved() / 2 ** 20
        stat_q.put(("done", wid, os.getpid(), stats))
    except Exception:
        stat_q.put(("fatal", wid, os.getpid(), traceback.format_exc()))


def post_worker(repo, threads, out, post_q, done_q):
    _setup_repo(repo)
    _cpu_only_torch()
    _apply_threads(threads)
    import infer
    from run import split_result
    while True:
        rec = post_q.get()
        if rec is None:
            break
        rec["t_post_start"] = time.time()
        save = rec["pass"] == 0
        packed = rec.pop("packed", None)
        try:
            if packed is None:
                raise RuntimeError("no detector result: %s" % rec.get("error", "?"))
            result = unpack_result(packed)
            postprocess(infer, split_result, rec, result, save, out)
        except Exception:
            rec["status"] = "error"
            rec["error"] = rec.get("error") or traceback.format_exc()
            if packed is not None:
                rec["error"] = traceback.format_exc()
            if save:
                common.write_json(common.run_paths(out, rec["id"])["meta"], rec)
        rec["t_post_end"] = time.time()
        done_q.put(rec)


# ------------------------------------------------------------------ measurement helpers

def _proc_stat():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle = v[3] + v[4]
    return sum(v), idle


def _pid_cpu_s(pid):
    try:
        with open("/proc/%d/stat" % pid) as f:
            parts = f.read().rsplit(")", 1)[1].split()
        return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")
    except Exception:
        return None


def _loadavg():
    with open("/proc/loadavg") as f:
        return f.read().split()[:3]


def _versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    from importlib import metadata  # no imports of torch/mmcv in the main process (see _cpu_only_torch)
    for name in ("torch", "mmcv-full", "mmdet", "numpy", "opencv-python"):
        try:
            out[name] = metadata.version(name)
        except Exception as e:
            out[name] = "unavailable: %r" % e
    return out


def _stats(xs):
    import numpy as np
    if not xs:
        return None
    a = np.asarray(xs, float)
    return {"n": int(a.size), "median": float(np.median(a)), "mean": float(a.mean()),
            "p90": float(np.percentile(a, 90)), "max": float(a.max())}


# ------------------------------------------------------------------ drivers

def run_inline(args, items, meta):
    """modes getds / serial: everything in this process."""
    _setup_repo(args.repo)
    import cv2
    import mmcv
    _apply_threads(args.gpu_threads)
    a = vars(args)
    torch, infer, model, devs, msda = _load_model(a, args.gpu_threads)
    from run import split_result
    meta.update(model_param_devices=devs, msda_path=msda)
    pipeline = build_test_pipeline(mmcv.Config.fromfile(args.config))
    recs = []
    for rep in range(args.repeat + 1):
        if rep == 1:
            win = _window_start([])
        for iid, path in items:
            rec = {"id": iid, "path": path, "pass": rep, "status": "error", "t_pre_start": time.time()}
            try:
                img = cv2.imread(path)
                if img is None:
                    raise ValueError("cv2.imread returned None for %s" % path)
                rec["shape"] = list(img.shape)
                if args.mode == "getds":
                    t = time.time()
                    ds = infer.get_dataseries(img, to_clean=False)
                    _sync(args.device)
                    rec["t_gpu_start"], rec["t_gpu_end"] = t, time.time()
                    rec["n_lines"] = len(ds)
                    rec["ds_hash"], rec["ds_hash_unordered"] = ds_hashes(ds)
                    rec["status"] = "ok"
                else:
                    save = rep == 0
                    if save:
                        rec["file_sha256"] = common.sha256_file(path)
                        rec["pixels_sha256"] = common.sha256_array(img)
                    data = preprocess(pipeline, img)
                    rec["tensor_shape"] = list(data_shape(data))
                    rec["t_pre_end"] = rec["t_gpu_start"] = time.time()
                    D2H_MARK["t"] = None
                    res = forward(model, [data])[0]
                    _sync(args.device)
                    rec["t_gpu_end"] = rec["t_post_start"] = time.time()
                    rec["t_fusion_end"] = D2H_MARK["t"]
                    postprocess(infer, split_result, rec, res, save, args.out)
            except Exception:
                rec["error"] = traceback.format_exc()
                print(rec["error"], file=sys.stderr, flush=True)
                if rep == 0 and args.mode != "getds":
                    common.write_json(common.run_paths(args.out, iid)["meta"], rec)
            rec["t_post_end"] = time.time()
            recs.append(rec)
        if rep == 0:
            _progress("pass 0 (warm-up + output) done", recs)
    win = _window_end(win, [])
    gpu_stats = {}
    if str(args.device).startswith("cuda"):
        gpu_stats = {"0": {"max_memory_allocated_MB": torch.cuda.max_memory_allocated() / 2 ** 20,
                           "max_memory_reserved_MB": torch.cuda.max_memory_reserved() / 2 ** 20}}
    return recs, win, gpu_stats, {"main": [os.getpid()]}


def _window_start(pids):
    return {"t0": time.time(), "stat0": _proc_stat(), "load0": _loadavg(),
            "cpu0": {p: _pid_cpu_s(p) for p in pids}, "self0": os.times()}


def _window_end(w, pids):
    t1, (tot1, idle1) = time.time(), _proc_stat()
    tot0, idle0 = w["stat0"]
    s1 = os.times()
    w.update(t1=t1, load1=_loadavg(), wall_s=t1 - w["t0"],
             wsl_cpu_busy_frac=1.0 - (idle1 - idle0) / max(1, tot1 - tot0),
             cpu_s={p: (_pid_cpu_s(p) or 0) - (w["cpu0"].get(p) or 0) for p in pids},
             self_cpu_s=(s1.user + s1.system) - (w["self0"].user + w["self0"].system))
    del w["cpu0"], w["self0"]
    return w


def _progress(msg, recs):
    n_err = sum(1 for r in recs if r.get("status") != "ok")
    print("%s: %d records, %d errors" % (msg, len(recs), n_err), flush=True)


def run_pipeline(args, items, meta):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    a = {k: getattr(args, k) for k in ("repo", "config", "ckpt", "device", "msda", "batch", "transfer",
                                       "gpu_workers", "gpu_mem_budget")}
    task_q = ctx.Queue()
    pre_q = ctx.Queue(maxsize=args.pre_queue or max(4, 2 * args.gpu_workers * args.batch))
    post_q = ctx.Queue(maxsize=args.post_queue or max(4, 2 * args.post_workers))
    done_q, stat_q = ctx.Queue(), ctx.Queue()
    procs = {"pre": [], "gpu": [], "post": []}

    def start(role, target, targs, threads):
        old = {k: os.environ.get(k) for k in THREAD_ENV}
        for k in THREAD_ENV:
            os.environ[k] = str(threads)
        p = ctx.Process(target=target, args=targs, daemon=True)
        p.start()
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        procs[role].append(p)

    for w in range(args.gpu_workers):
        start("gpu", gpu_worker, (w, a, args.gpu_threads, pre_q, post_q, stat_q), args.gpu_threads)
    ready, gpu_stats, gpu_info = 0, {}, {}
    while ready < args.gpu_workers:  # load models one after the other is not needed; wait for all
        kind, wid, pid, info = stat_q.get(timeout=600)
        if kind == "fatal":
            raise RuntimeError("GPU worker %d failed:\n%s" % (wid, info))
        gpu_info[str(wid)] = dict(info, pid=pid)
        ready += 1
    meta["gpu_workers_info"] = gpu_info
    for _ in range(args.pre_workers):
        start("pre", pre_worker, (args.config, args.repo, args.pre_threads, task_q, pre_q), args.pre_threads)
    for _ in range(args.post_workers):
        start("post", post_worker, (args.repo, args.post_threads, args.out, post_q, done_q), args.post_threads)
    pids = {r: [p.pid for p in ps] for r, ps in procs.items()}
    all_pids = [p for ps in pids.values() for p in ps]

    def collect(n):
        out = []
        while len(out) < n:
            try:
                out.append(done_q.get(timeout=5))
            except queue_mod.Empty:
                while not stat_q.empty():
                    kind, wid, pid, info = stat_q.get()
                    if kind == "fatal":
                        raise RuntimeError("GPU worker %d failed:\n%s" % (wid, info))
                dead = [p.pid for ps in procs.values() for p in ps if not p.is_alive()]
                if dead:
                    raise RuntimeError("worker processes died: %s" % dead)
        return out

    recs = []
    for iid, path in items:  # pass 0: warm-up and the harness output
        task_q.put((0, iid, path, True))
    recs += collect(len(items))
    _progress("pass 0 (warm-up + output) done", recs)
    win = _window_start(all_pids)
    for rep in range(1, args.repeat + 1):
        for iid, path in items:
            task_q.put((rep, iid, path, False))
    recs += collect(len(items) * args.repeat)
    win = _window_end(win, all_pids)
    for _ in procs["pre"]:
        task_q.put(None)
    for p in procs["pre"]:
        p.join()
    for _ in procs["gpu"]:
        pre_q.put(None)
    n_done = 0
    while n_done < args.gpu_workers:
        kind, wid, pid, info = stat_q.get(timeout=600)
        if kind == "fatal":
            raise RuntimeError("GPU worker %d failed:\n%s" % (wid, info))
        if kind == "done":
            gpu_stats[str(wid)] = info
            n_done += 1
    for p in procs["gpu"]:
        p.join()
    for _ in procs["post"]:
        post_q.put(None)
    for p in procs["post"]:
        p.join()
    win["cpu_s_by_role"] = {r: sum(win["cpu_s"].get(p, 0) for p in ps) for r, ps in pids.items()}
    return recs, win, gpu_stats, pids


def selfcheck(args, items):
    """CPU proof that the split path equals inference_detector / get_dataseries bit for bit, and what an
    equal-shape batch of two does."""
    import numpy as np
    _setup_repo(args.repo)
    import cv2
    import mmcv
    from mmdet.apis import inference_detector
    a = vars(args)
    torch, infer, model, devs, msda = _load_model(a, args.gpu_threads)
    from run import split_result
    pipeline = build_test_pipeline(mmcv.Config.fromfile(args.config))
    report = {"device": args.device, "images": []}
    by_shape = {}
    for iid, path in items[: args.selfcheck]:
        img = cv2.imread(path)
        ref = inference_detector(model, img)
        ref_ds = infer.get_dataseries(img, to_clean=False)
        data = preprocess(pipeline, img)
        got = forward(model, [data])[0]
        ds, _ = dataseries_from_result(infer, got)
        bA, lA, mA = split_result(ref)
        bB, lB, mB = split_result(got)
        r = {"id": iid, "tensor_shape": data_shape(data),
             "boxes_identical": bool(np.array_equal(bA, bB)),
             "masks_identical": bool(len(mA) == len(mB) and all(np.array_equal(x, y) for x, y in zip(mA, mB))),
             "dataseries_identical": ds == ref_ds}
        kept = pack_result(got, "kept")
        r["kept_transfer_dataseries_identical"] = dataseries_from_result(infer, unpack_result(kept))[0] == ref_ds
        report["images"].append(r)
        by_shape.setdefault(data_shape(data), []).append((iid, data, got))
        print(r, flush=True)
    report["batch"] = []
    for s, lst in by_shape.items():
        if len(lst) < 2:
            continue
        res = forward(model, [lst[0][1], lst[1][1]])
        for (iid, _, single), b in zip(lst[:2], res):
            bA, _, mA = split_result(single)
            bB, _, mB = split_result(b)
            r = {"id": iid, "shape": s, "boxes_identical": bool(np.array_equal(bA, bB)),
                 "max_abs_dscore": float(np.max(np.abs(bA[:, 4] - bB[:, 4]))) if len(bA) == len(bB) else None,
                 "masks_identical": bool(len(mA) == len(mB) and all(np.array_equal(x, y)
                                                                      for x, y in zip(mA, mB)))}
            report["batch"].append(r)
            print("batch2", r, flush=True)
        break
    ok = all(r["boxes_identical"] and r["masks_identical"] and r["dataseries_identical"]
             and r["kept_transfer_dataseries_identical"] for r in report["images"])
    report["split_path_bit_identical"] = ok
    print("split path bit-identical on %d images: %s" % (len(report["images"]), ok), flush=True)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--msda", choices=["auto", "compiled", "pytorch"], default=None)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True, help="run directory (harness format) of pass 0")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--mode", choices=["getds", "serial", "pipeline"], default="pipeline")
    ap.add_argument("--gpu-workers", type=int, default=1)
    ap.add_argument("--gpu-mem-budget", type=float, default=0.85,
                    help="fraction of device memory shared by all GPU workers (per worker: budget / workers)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--pre-workers", type=int, default=4)
    ap.add_argument("--post-workers", type=int, default=4)
    ap.add_argument("--pre-threads", type=int, default=1)
    ap.add_argument("--post-threads", type=int, default=1)
    ap.add_argument("--gpu-threads", type=int, default=2)
    ap.add_argument("--pre-queue", type=int, default=0)
    ap.add_argument("--post-queue", type=int, default=0)
    ap.add_argument("--transfer", choices=["kept", "all"], default="kept", help="masks sent in the timed passes")
    ap.add_argument("--repeat", type=int, default=5, help="timed passes after pass 0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selfcheck", type=int, default=0, help="compare split path vs inference_detector on K images")
    args = ap.parse_args(argv)
    for k in ("repo", "config", "ckpt", "images", "out"):
        setattr(args, k, str(Path(getattr(args, k)).resolve()))
    items = common.read_image_list(args.images)
    if args.limit:
        items = items[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.selfcheck:
        rep = selfcheck(args, items)
        common.write_json(out / "selfcheck.json", rep)
        return 0 if rep["split_path_bit_identical"] else 1
    if args.batch > 1 and args.mode != "pipeline":
        raise SystemExit("--batch needs --mode pipeline")
    meta = {"tag": args.tag or out.name, "argv": sys.argv, "repo": args.repo, "config": args.config,
            "ckpt": args.ckpt, "ckpt_sha256": common.sha256_file(args.ckpt), "device_requested": args.device,
            "msda_requested": args.msda, "images_list": args.images, "image_ids": [i for i, _ in items],
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "running", "versions": _versions(),
            "runner": "tools/throughput/batch_infer.py",
            "config_throughput": {k: getattr(args, k) for k in (
                "mode", "gpu_workers", "batch", "pre_workers", "post_workers", "pre_threads", "post_threads",
                "gpu_threads", "transfer", "repeat")}}
    try:
        meta["repo_git"] = subprocess.run(["git", "-C", args.repo, "rev-parse", "HEAD"], capture_output=True,
                                          text=True).stdout.strip()
    except Exception as e:
        meta["repo_git"] = "unavailable: %r" % e
    common.write_json(out / "run_meta.json", meta)
    if args.mode == "pipeline":
        recs, win, gpu_stats, pids = run_pipeline(args, items, meta)
    else:
        recs, win, gpu_stats, pids = run_inline(args, items, meta)

    n_img = len(items)
    p0 = [r for r in recs if r["pass"] == 0]
    timed = [r for r in recs if r["pass"] > 0]
    errs = [r for r in recs if r.get("status") != "ok"]
    ref_hash = {r["id"]: (r.get("ds_hash"), r.get("ds_hash_unordered")) for r in p0}
    same_exact = sum(1 for r in timed if ref_hash.get(r["id"], (None,))[0] == r.get("ds_hash"))
    same_unord = sum(1 for r in timed if ref_hash.get(r["id"], (None, None))[1] == r.get("ds_hash_unordered"))
    lat = [r["t_post_end"] - r["t_pre_start"] for r in timed if "t_post_end" in r]
    gpu_t = [r["t_gpu_end"] - r["t_gpu_start"] for r in timed if "t_gpu_end" in r]
    post_t = [r.get("t_post_compute_s") for r in timed if r.get("t_post_compute_s") is not None]
    pack_t = [r["t_pack_end"] - r["t_gpu_end"] for r in timed if "t_pack_end" in r]
    # per batch (dedupe by start time): forward split into compute (incl. upsample) and the mmdet d2h loop
    batches = {}
    for r in timed:
        if r.get("t_fusion_end") and "t_gpu_end" in r:
            batches[(r.get("gpu_worker"), r["t_gpu_start"])] = (r["t_gpu_start"], r["t_fusion_end"], r["t_gpu_end"])
    comp = sum(f - s for s, f, e in batches.values())
    d2h = sum(e - f for s, f, e in batches.values())
    wait_post = [r["t_post_start"] - r.get("t_pack_end", r["t_gpu_end"]) for r in timed if "t_post_start" in r]
    # GPU stage busy: union of [t_gpu_start, t_gpu_end] intervals inside the window (forward incl. mmdet D2H)
    iv = sorted(set((r["t_gpu_start"], r["t_gpu_end"]) for r in timed if "t_gpu_end" in r))
    busy, cur = 0.0, None
    for s, e in iv:
        if cur is None or s > cur[1]:
            if cur:
                busy += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur:
        busy += cur[1] - cur[0]
    summary = {
        "n_images": n_img, "repeat": args.repeat, "n_timed": len(timed), "n_errors": len(errs),
        "wall_s": win["wall_s"], "images_per_s": len(timed) / win["wall_s"] if win["wall_s"] else None,
        "latency_s": _stats(lat), "gpu_stage_s": _stats(gpu_t), "post_compute_s": _stats(post_t),
        "pack_s": _stats(pack_t), "wait_before_post_s": _stats(wait_post),
        "forward_compute_s_total": comp, "forward_d2h_s_total": d2h,
        "d2h_share_of_forward": d2h / (comp + d2h) if (comp + d2h) else None,
        "d2h_s_per_image": d2h / len(timed) if timed else None,
        "compute_s_per_image": comp / len(timed) if timed else None,
        "gpu_stage_occupancy_frac": busy / win["wall_s"] if win["wall_s"] else None,
        "wsl_cpu_busy_frac": win["wsl_cpu_busy_frac"], "loadavg_start": win["load0"], "loadavg_end": win["load1"],
        "cpu_s_by_role": win.get("cpu_s_by_role"), "main_cpu_s": win["self_cpu_s"],
        "window_unix": [win["t0"], win["t1"]],
        "gpu_workers": gpu_stats,
        "timed_dataseries_identical_to_pass0": [same_exact, len(timed)],
        "timed_dataseries_identical_to_pass0_line_order_free": [same_unord, len(timed)],
    }
    if gpu_stats:
        summary["max_memory_allocated_MB_max"] = max(v.get("max_memory_allocated_MB", 0) for v in gpu_stats.values())
        summary["max_memory_reserved_MB_sum"] = sum(v.get("max_memory_reserved_MB", 0) for v in gpu_stats.values())
    meta["throughput"] = summary
    meta["pids"] = pids
    meta["per_image_pass0"] = [{k: r.get(k) for k in ("id", "status", "tensor_shape", "batch_size", "gpu_worker",
                                                       "n_lines", "n_ge_0.3")} for r in p0]
    meta["errors"] = [{"id": r["id"], "pass": r["pass"], "error": r.get("error")} for r in errs][:20]
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p0_err = sum(1 for r in p0 if r.get("status") != "ok")
    meta["status"] = "done" if (p0_err == 0 and args.mode != "getds") else (
        "timing_only" if args.mode == "getds" else "done_with_errors")
    meta["n_errors"] = len(errs)
    common.write_json(out / "run_meta.json", meta)
    print(json.dumps({k: summary[k] for k in ("images_per_s", "wall_s", "n_errors", "gpu_stage_occupancy_frac",
                                              "wsl_cpu_busy_frac", "timed_dataseries_identical_to_pass0")}),
          flush=True)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
