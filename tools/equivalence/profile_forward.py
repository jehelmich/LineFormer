# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""Where does LineFormer spend its time, and does the GPU really do the work? One device per run.

(Named profile_forward.py, not profile.py: a profile.py next to the script would shadow the standard library module
that cProfile imports, and torch imports cProfile.)

After --warmup untimed images, three passes over the same --n images, each image through
infer.get_dataseries(img, to_clean=False) (which runs mmdet inference_detector inside):

1. Wall clock, with the GPU synchronised at every boundary so times are attributable (the synchronisation adds a
   little overhead, the same for every part). Per image:
     get_dataseries      the whole call
     inference_detector  the detector call inside it
     model_forward       model.forward (network + mmdet postprocessing incl. the mask copies to the host)
     d2h                 every Tensor.cpu() call made inside model.forward (masks, boxes); GPU synchronised first,
                         so pending kernels are not counted as copy time
   derived: preprocessing = inference_detector - model_forward (test pipeline, collate, host->device);
            forward_compute = model_forward - d2h; cpu_postprocessing = get_dataseries - inference_detector
            (mask -> data series in line_utils / scipy).
   Parts inside the forward: backbone, pixel_decoder (+ .encoder), msda (all MultiScaleDeformableAttention-like
   modules), transformer_decoder.layers. Nested parts are not additive.
2. Op census on the first timed image: a TorchDispatchMode active only inside model.forward records every aten op
   and the device of its tensors (inputs; outputs for ops without tensor inputs). Every op that ran on the CPU
   during a GPU forward is listed by name with its count. (Needs torch >= 2.0; recorded as unavailable otherwise.)
3. torch.profiler (CPU + CUDA/HIP activities on a GPU): device events (kernels, copies, memsets) are summed and
   assigned to the model_forward / get_dataseries windows by start time; GPU busy fraction = device time in the
   windows / window wall time. Device events per forward = events in the forward windows / number of forwards.
   Top operators by self device time (GPU) or self CPU time.
   Where the runtime cannot trace device activity (ROCm under WSL), device time and busy fraction stay null and
   kernel launches are counted from the HIP/CUDA runtime API calls (…LaunchKernel, Memcpy, Memset) instead.
4. --sustain S: an idle phase (S/2), then S seconds of back-to-back inference_detector calls, then S seconds of
   get_dataseries calls, with no synchronisation added; the unix start/end of each phase is saved so that an
   external utilisation sampler (on Windows/WSL: the "GPU Engine" compute counters) gives the GPU busy fraction.
Output: a JSON file (--out) and a short summary on stdout. Python 3.8 compatible (pass 2 needs torch 2).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import run as runmod  # noqa: E402


def _evt_time(evt, names):
    for n in names:
        v = getattr(evt, n, None)
        if v is not None:
            return float(v)
    return 0.0


def _med(xs):
    xs = sorted(xs)
    if not xs:
        return None
    k = len(xs) // 2
    return xs[k] if len(xs) % 2 else 0.5 * (xs[k - 1] + xs[k])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=5, help="images to time (after warm-up)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--msda", choices=["auto", "compiled", "pytorch"], default=None)
    ap.add_argument("--kept-only", type=float, default=None, metavar="THR",
                    help="infer.load_model(kept_only=THR), see kept_queries.py")
    ap.add_argument("--out", required=True, help="json file")
    ap.add_argument("--trace", default=None, help="optional chrome trace file of the profiler pass")
    ap.add_argument("--no-profiler", action="store_true", help="skip pass 3")
    ap.add_argument("--sustain", type=float, default=0.0,
                    help="seconds per phase of pass 4 (0 = off): idle, inference_detector loop, get_dataseries loop")
    args = ap.parse_args(argv)

    import cv2
    import torch

    items = common.read_image_list(args.images)
    if len(items) < args.warmup + args.n:
        raise SystemExit("need %d images, list has %d" % (args.warmup + args.n, len(items)))
    out_path = Path(args.out).resolve()
    trace_path = str(Path(args.trace).resolve()) if args.trace else None
    images_spec = str(Path(args.images).resolve())
    meta = {"device_requested": args.device, "msda_requested": args.msda, "kept_only_requested": args.kept_only,
            "images_list": images_spec,
            "versions": runmod._versions()}
    infer, model = runmod.load(args, meta)
    gpu = str(args.device).startswith("cuda")
    meta["model_param_devices"] = sorted(set(str(p.device) for p in model.parameters()))

    def sync():
        if gpu and not state.get("nosync"):
            torch.cuda.synchronize()

    wall = defaultdict(float)
    calls = defaultdict(int)
    state = {"in_forward": False, "census": None, "nosync": False}

    def timed(fn, name, on_enter=None, on_exit=None):
        def f(*a, **k):
            with torch.profiler.record_function("EQ::" + name):
                sync()
                t = time.perf_counter()
                if on_enter:
                    on_enter()
                try:
                    r = fn(*a, **k)
                    sync()
                finally:
                    if on_exit:
                        on_exit()
                wall[name] += time.perf_counter() - t
                calls[name] += 1
                return r
        return f

    # --- census mode (pass 2)
    census_mode = None
    try:
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_flatten

        class Census(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.ops = Counter()

            def __torch_dispatch__(self, func, types, a=(), kw=None):
                kw = kw or {}
                r = func(*a, **kw)
                flat, _ = tree_flatten((a, kw))
                devs = set(x.device.type for x in flat if isinstance(x, torch.Tensor))
                if not devs:
                    of, _ = tree_flatten(r)
                    devs = set(x.device.type for x in of if isinstance(x, torch.Tensor)) or {"none"}
                self.ops[(str(func), ",".join(sorted(devs)))] += 1
                return r
        census_mode = Census
    except Exception as e:  # torch 1.x
        meta["census_unavailable"] = repr(e)

    def fwd_enter():
        state["in_forward"] = True
        if state["census"] is not None:
            state["census"].__enter__()

    def fwd_exit():
        if state["census"] is not None:
            state["census"].__exit__(None, None, None)
        state["in_forward"] = False

    model.forward = timed(model.forward, "model_forward", fwd_enter, fwd_exit)
    head = getattr(model, "panoptic_head", None)
    pd = getattr(head, "pixel_decoder", None)
    td = getattr(head, "transformer_decoder", None)
    parts = {"backbone": getattr(model, "backbone", None), "pixel_decoder": pd,
             "pixel_decoder.encoder": getattr(pd, "encoder", None)}
    missing = [k for k, v in parts.items() if v is None]
    for k, v in parts.items():
        if v is not None:
            v.forward = timed(v.forward, k)
    # Mask2FormerHead calls its transformer decoder's layers directly, not transformer_decoder.forward
    for layer in (getattr(td, "layers", None) or []):
        layer.forward = timed(layer.forward, "transformer_decoder.layers")
    msda = [(n, m) for n, m in model.named_modules() if "DeformableAttention" in type(m).__name__]
    for _, m in msda:
        m.forward = timed(m.forward, "msda")
    if not msda:
        raise SystemExit("no MultiScaleDeformableAttention-like module found in the model")
    meta["wrapped"] = {"missing_parts": missing, "n_msda_modules": len(msda),
                       "msda_classes": sorted(set("%s.%s" % (type(m).__module__, type(m).__name__)
                                                  for _, m in msda))}
    # detector inside get_dataseries (infer.do_instance looks the name up in infer's globals)
    infer.inference_detector = timed(infer.inference_detector, "inference_detector")
    # device -> host copies inside the forward
    orig_cpu = torch.Tensor.cpu

    def cpu_timed(self, *a, **k):
        if not state["in_forward"]:
            return orig_cpu(self, *a, **k)
        sync()
        t = time.perf_counter()
        r = orig_cpu(self, *a, **k)
        wall["d2h"] += time.perf_counter() - t
        calls["d2h"] += 1
        if self.device.type != "cpu":
            calls["d2h_from_device"] += 1
        return r
    torch.Tensor.cpu = cpu_timed

    def get_ds(im):
        sync()
        t = time.perf_counter()
        with torch.profiler.record_function("EQ::get_dataseries"):
            ds = infer.get_dataseries(im, to_clean=False)
            sync()
        wall["get_dataseries"] += time.perf_counter() - t
        calls["get_dataseries"] += 1
        return ds

    imgs = []
    for iid, p in items[: args.warmup + args.n]:
        im = cv2.imread(p)
        if im is None:
            raise SystemExit("cannot read %s" % p)
        imgs.append((iid, im))

    try:
        with torch.no_grad():
            for _, im in imgs[: args.warmup]:
                get_ds(im)
            # every timed image once untimed too: on a GPU the first call with a new input size pays for kernel
            # selection / compilation, which is not what the split below is about (run.py's per-image times do
            # include it)
            for _, im in imgs[args.warmup:]:
                get_ds(im)
            # ---- pass 1
            per_image = []
            for iid, im in imgs[args.warmup:]:
                wall.clear()
                calls.clear()
                get_ds(im)
                w = dict(wall)
                rec = {"id": iid, "shape": list(im.shape), "seconds": w, "calls": dict(calls)}
                rec["derived"] = {
                    "preprocessing": w["inference_detector"] - w["model_forward"],
                    "forward_compute": w["model_forward"] - w.get("d2h", 0.0),
                    "d2h": w.get("d2h", 0.0),
                    "cpu_postprocessing": w["get_dataseries"] - w["inference_detector"],
                }
                per_image.append(rec)
            keys = sorted(set(k for r in per_image for k in r["seconds"]))
            dkeys = ["preprocessing", "forward_compute", "d2h", "cpu_postprocessing"]
            med = {k: _med([r["seconds"].get(k, 0.0) for r in per_image]) for k in keys}
            dmed = {k: _med([r["derived"][k] for r in per_image]) for k in dkeys}
            tot = {k: sum(r["seconds"].get(k, 0.0) for r in per_image) for k in keys}
            fwd = tot.get("model_forward", 0.0)
            gds = tot.get("get_dataseries", 0.0)
            meta["wall"] = {
                "n_images": len(per_image), "per_image": per_image,
                "median_s": med, "median_split_s": dmed,
                "share_of_model_forward": {k: (tot[k] / fwd if fwd else None) for k in keys},
                "share_of_get_dataseries": {k: (tot[k] / gds if gds else None) for k in keys},
                "split_share_of_get_dataseries": {k: (sum(r["derived"][k] for r in per_image) / gds if gds else None)
                                                  for k in dkeys},
            }
            # ---- pass 2
            if census_mode is not None:
                state["census"] = census_mode()
                infer.get_dataseries(imgs[args.warmup][1], to_clean=False)
                ops = state["census"].ops
                state["census"] = None
                by_dev = Counter()
                for (name, dev), c in ops.items():
                    by_dev[dev] += c
                n_all = sum(by_dev.values())
                meta["op_census_one_forward"] = {
                    "image": imgs[args.warmup][0],
                    "n_ops": n_all,
                    "ops_by_device": dict(by_dev),
                    "share_by_device": {k: v / n_all for k, v in by_dev.items()} if n_all else {},
                    "non_gpu_ops": sorted([{"op": n, "device": d, "count": c} for (n, d), c in ops.items()
                                           if "cuda" not in d], key=lambda r: -r["count"]) if gpu else
                    "device is cpu: every op is a CPU op",
                }
            # ---- pass 4 (before the profiler, which slows everything down)
            if args.sustain > 0:
                state["nosync"] = True
                phases = []

                def phase(name, fn):
                    t0 = time.time()
                    k = 0
                    while time.time() - t0 < args.sustain:
                        fn(imgs[args.warmup + k % args.n][1])
                        k += 1
                    if gpu:
                        torch.cuda.synchronize()
                    phases.append({"phase": name, "unix_start": t0, "unix_end": time.time(), "iterations": k})
                    print("phase %s: %d iterations in %.1fs" % (name, k, time.time() - t0), flush=True)
                t0 = time.time()
                time.sleep(args.sustain / 2)
                phases.append({"phase": "idle", "unix_start": t0, "unix_end": time.time(), "iterations": 0})
                phase("inference_detector_loop", lambda im: infer.inference_detector(model, im))
                phase("get_dataseries_loop", lambda im: infer.get_dataseries(im, to_clean=False))
                state["nosync"] = False
                meta["sustained_phases"] = {
                    "phases": phases,
                    "note": "no synchronisation inside these loops (the wrappers skip it); GPU busy fraction per "
                            "phase comes from an external sampler (e.g. Windows GPU engine counters for WSL), "
                            "matched by unix time"}
            # ---- pass 3
            if not args.no_profiler:
                acts = [torch.profiler.ProfilerActivity.CPU]
                if gpu:
                    acts.append(torch.profiler.ProfilerActivity.CUDA)
                try:
                    with torch.profiler.profile(activities=acts) as prof:
                        for _, im in imgs[args.warmup:]:
                            get_ds(im)
                    meta["profiler"] = summarise_profiler(prof, gpu, trace_path)
                except Exception as e:
                    import traceback
                    meta["profiler"] = {"error": traceback.format_exc()}
                    print("profiler pass failed:", repr(e), file=sys.stderr)
    finally:
        torch.Tensor.cpu = orig_cpu

    common.write_json(out_path, meta)
    print_summary(meta, args.device)
    print("written", out_path)
    return 0


def summarise_profiler(prof, gpu, trace_path):
    import torch
    res = {}
    events = list(prof.events())
    windows = defaultdict(list)
    for e in events:
        if e.name in ("EQ::model_forward", "EQ::get_dataseries"):
            windows[e.name].append((e.time_range.start, e.time_range.end))
    dev_type = getattr(torch.autograd, "DeviceType", None)
    dev_events = []
    if gpu and dev_type is not None:
        dev_events = [e for e in events if getattr(e, "device_type", None) == dev_type.CUDA]
    res["n_device_events_total"] = len(dev_events)
    if gpu and not dev_events:
        res["device_tracing"] = "no device events captured (e.g. ROCm under WSL has no kfd sysfs for roctracer); "                                 "kernel launches are counted from runtime API calls instead, device time unavailable"
    rt = [e for e in events if any(t in e.name for t in ("LaunchKernel", "Memcpy", "Memset", "ModuleLaunch"))]

    def in_windows(e, ws):
        return any(a <= e.time_range.start <= b for a, b in ws)

    for wname, ws in windows.items():
        wall_us = sum(b - a for a, b in ws)
        evs = [e for e in dev_events if in_windows(e, ws)]
        dev_us = sum(e.time_range.elapsed_us() for e in evs)
        kinds = Counter()
        for e in evs:
            nm = e.name.lower()
            if "memcpy" in nm or ("copy" in nm and "kernel" not in nm):
                kinds["memcpy"] += 1
            elif "memset" in nm:
                kinds["memset"] += 1
            else:
                kinds["kernel"] += 1
        rts = Counter(e.name for e in rt if in_windows(e, ws))
        res[wname] = {"n_windows": len(ws), "wall_ms": wall_us / 1000.0,
                      "device_ms": dev_us / 1000.0 if dev_events else None,
                      "gpu_busy_fraction": (dev_us / wall_us) if (wall_us and gpu and dev_events) else None,
                      "runtime_api_calls_per_window": {k: v / max(1, len(ws)) for k, v in rts.items()},
                      "device_events_per_window": {k: v / max(1, len(ws)) for k, v in kinds.items()}}
    ka = prof.key_averages()
    dev_names = ("self_device_time_total", "self_cuda_time_total")
    dev_tot_names = ("device_time_total", "cuda_time_total")
    rows = []
    for e in ka:
        rows.append({"name": e.key, "count": e.count,
                     "self_cpu_ms": e.self_cpu_time_total / 1000.0,
                     "cpu_total_ms": e.cpu_time_total / 1000.0,
                     "self_device_ms": _evt_time(e, dev_names) / 1000.0,
                     "device_total_ms": _evt_time(e, dev_tot_names) / 1000.0})
    sort_key = "self_device_ms" if (gpu and dev_events) else "self_cpu_ms"
    ops = sorted([r for r in rows if not r["name"].startswith("EQ::")], key=lambda r: -r[sort_key])
    tot = sum(r[sort_key] for r in ops)
    for r in ops:
        r["share_of_" + sort_key] = r[sort_key] / tot if tot else None
    res.update({"sorted_by": sort_key, "total_" + sort_key: tot,
                "ranges": [r for r in rows if r["name"].startswith("EQ::")], "top_ops": ops[:30],
                "note": "device events assigned to a window by their start time; busy fraction assumes one stream"})
    if trace_path:
        prof.export_chrome_trace(trace_path)
    return res


def print_summary(meta, device):
    w = meta["wall"]
    print("device %s, %d images (median per image)" % (device, w["n_images"]))
    for k in ("get_dataseries", "inference_detector", "model_forward", "d2h", "backbone", "pixel_decoder",
              "pixel_decoder.encoder", "msda", "transformer_decoder.layers"):
        if k in w["median_s"]:
            print("  %-26s %8.3fs  %5.1f%% of forward" % (k, w["median_s"][k],
                                                          100 * (w["share_of_model_forward"].get(k) or 0)))
    print("  split:", {k: round(v, 3) for k, v in w["median_split_s"].items()})
    c = meta.get("op_census_one_forward")
    if c:
        print("  op census:", c["ops_by_device"])
        if isinstance(c["non_gpu_ops"], list):
            for r in c["non_gpu_ops"][:15]:
                print("    non-GPU op %-50s %-8s %d" % (r["op"], r["device"], r["count"]))
    p = meta.get("profiler") or {}
    if "error" in p:
        print("  profiler error:", p["error"].splitlines()[-1])
    for k in ("EQ::model_forward", "EQ::get_dataseries"):
        if k in p:
            print("  %s: wall %.1f ms, device %s ms, busy %s, device events/window %s, runtime calls/window %s" % (
                k, p[k]["wall_ms"], p[k]["device_ms"], p[k]["gpu_busy_fraction"], p[k]["device_events_per_window"],
                p[k].get("runtime_api_calls_per_window")))
    for r in (p.get("top_ops") or [])[:10]:
        print("    %-45s %10.1f ms  %5.1f%%" % (r["name"][:45], r[p["sorted_by"]],
                                                 100 * (r.get("share_of_" + p["sorted_by"]) or 0)))


if __name__ == "__main__":
    sys.exit(main())
