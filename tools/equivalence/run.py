# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""Run a LineFormer checkout over a list of images and save everything needed to compare two runs.

Per image (see common.py for the file layout):
  (a) mmdet inference_detector -> all instances (no score threshold): boxes + scores, labels, masks (packed);
  (b) infer.get_dataseries(img, to_clean=False) -> points (a second, independent forward pass, as the repo does it);
  timings: detector and dataseries separately, GPU synchronised before each clock read; the first image is
  flagged as warm-up and excluded from the summary.
run_meta.json: versions (python, torch, mmcv, mmdet, numpy, cv2), device asked for, evidence of the device that
was actually used (parameter devices, devices seen by forward hooks on the backbone and on every
MultiScaleDeformableAttention-like module, torch.cuda.is_available, torch.version.hip), MSDA mode handling, the
calls counted on mmcv's compiled and pure-PyTorch MSDA functions (pass-through counters; --no-instrument
switches every hook off), model load time, per-image summary.

Rules:
- --msda is passed to infer.load_model only if its signature takes `msda` or `msda_mode`. A checkout without
  that option is accepted for --msda auto (recorded as "not supported by repo"); asking for compiled/pytorch
  from such a checkout raises.
- An image that fails is recorded with status "error" and its traceback; the run continues and exits non-zero.
- --resume skips images whose <id>.meta.json already says "ok"; everything else is (re)computed.
- Images are read with cv2.imread (BGR, as the repo's demo does); the sha256 of the file and of the decoded
  pixel array are recorded, so a decoder difference between stacks is visible.
Python 3.8 compatible.
"""
from __future__ import annotations

import argparse
import inspect
import os
import platform
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402


def _sync(device):
    import torch
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("torch", "torchvision", "mmcv", "mmdet", "numpy", "cv2", "scipy", "skimage"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception as e:  # recorded, not fatal
            out[name] = "unavailable: %s" % e
    try:
        import torch
        out["torch_cuda_available"] = bool(torch.cuda.is_available())
        out["torch_version_hip"] = getattr(torch.version, "hip", None)
        out["torch_version_cuda"] = getattr(torch.version, "cuda", None)
        out["torch_num_threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            out["cuda_device_count"] = torch.cuda.device_count()
            out["cuda_device_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as e:
        out["torch_info_error"] = repr(e)
    try:
        import importlib
        ext = importlib.import_module("mmcv._ext")
        out["mmcv_ext"] = {"loaded": True, "has_ms_deform_attn_forward": hasattr(ext, "ms_deform_attn_forward")}
    except Exception as e:
        out["mmcv_ext"] = {"loaded": False, "error": repr(e)}
    return out


class Instrument:
    """Pass-through hooks that record which devices and MSDA code paths were used."""

    def __init__(self):
        self.devices = {"backbone_input": set(), "msda_output": set()}
        self.msda_calls = {"mmcv_compiled_function": 0, "mmcv_pytorch_function": 0}
        self.msda_modules = []
        self._restore = []

    def attach(self, model):
        backbone = getattr(model, "backbone", None)
        if backbone is not None:
            def bb_hook(mod, args):
                for a in args:
                    if hasattr(a, "device"):
                        self.devices["backbone_input"].add(str(a.device))
            backbone.register_forward_pre_hook(bb_hook)
        for name, mod in model.named_modules():
            cls = type(mod)
            cname = cls.__name__
            if "DeformableAttention" in cname or ("DeformAttn" in cname and "Decoder" not in cname):
                self.msda_modules.append({"name": name, "class": "%s.%s" % (cls.__module__, cls.__name__)})

                def msda_hook(m, args, out):
                    t = out[0] if isinstance(out, (tuple, list)) else out
                    if hasattr(t, "device"):
                        self.devices["msda_output"].add(str(t.device))
                mod.register_forward_hook(msda_hook)
        try:
            import mmcv.ops.multi_scale_deform_attn as msda_mod
        except Exception:
            msda_mod = None
        if msda_mod is not None:
            fn = getattr(msda_mod, "multi_scale_deformable_attn_pytorch", None)
            if fn is not None:
                def counted_pt(*a, **k):
                    self.msda_calls["mmcv_pytorch_function"] += 1
                    return fn(*a, **k)
                msda_mod.multi_scale_deformable_attn_pytorch = counted_pt
                self._restore.append((msda_mod, "multi_scale_deformable_attn_pytorch", fn))
            F = getattr(msda_mod, "MultiScaleDeformableAttnFunction", None)
            if F is not None:
                orig_apply = F.apply

                def counted_apply(*a, **k):
                    self.msda_calls["mmcv_compiled_function"] += 1
                    return orig_apply(*a, **k)
                F.apply = staticmethod(counted_apply)
                self._restore.append((F, "apply", orig_apply))

    def report(self):
        return {
            "devices_seen": {k: sorted(v) for k, v in self.devices.items()},
            "msda_function_calls": dict(self.msda_calls),
            "msda_modules": self.msda_modules,
            "note": "counters only see mmcv.ops.multi_scale_deform_attn; an MSDA implemented elsewhere shows 0 here "
                    "but still appears under devices_seen.msda_output if its class name matches",
        }


def load(args, meta):
    config, ckpt = str(Path(args.config).resolve()), str(Path(args.ckpt).resolve())
    sys.path.insert(0, str(Path(args.repo).resolve()))
    os.chdir(str(Path(args.repo).resolve()))  # the repo's infer imports its siblings and may use relative paths
    import torch
    import infer
    meta["infer_file"] = str(Path(infer.__file__).resolve())
    sig = inspect.signature(infer.load_model)
    kw = {}
    msda_param = next((p for p in ("msda", "msda_mode") if p in sig.parameters), None)
    if msda_param is not None:
        if args.msda is not None:
            kw[msda_param] = args.msda
        meta["msda_handling"] = "passed %s=%r" % (msda_param, args.msda) if args.msda is not None else \
            "repo supports %s, not passed (repo default)" % msda_param
    else:
        if args.msda not in (None, "auto"):
            raise SystemExit("--msda %s asked for, but infer.load_model%s has no msda option" % (args.msda, sig))
        meta["msda_handling"] = "not supported by repo (load_model%s)" % sig
    kept = getattr(args, "kept_only", None)
    if kept is not None:
        if "kept_only" not in sig.parameters:
            raise SystemExit("--kept-only asked for, but infer.load_model%s has no kept_only option" % (sig,))
        kw["kept_only"] = kept
    _sync(args.device)
    t = time.perf_counter()
    ret = infer.load_model(config, ckpt, args.device, **kw)
    _sync(args.device)
    meta["t_model_load_s"] = time.perf_counter() - t
    model = infer.model
    devs = sorted(set(str(p.device) for p in model.parameters()))
    meta["model_param_devices"] = devs
    meta["model_buffer_devices"] = sorted(set(str(b.device) for b in model.buffers()))
    meta["load_model_return"] = repr(ret)[:500]
    # whatever the checkout reports about its MSDA path (module-level names containing "msda")
    rep = {}
    for name in dir(infer):
        if "msda" in name.lower():
            v = getattr(infer, name)
            if isinstance(v, (str, bool, int, float, type(None), dict, list, tuple)):
                rep[name] = v
            elif callable(v) and name.lower().startswith(("get_", "describe")):
                try:
                    rep[name + "()"] = v()
                except Exception as e:
                    rep[name + "()"] = "error: %r" % e
    meta["infer_msda_report"] = rep
    # kept-queries mode (kept_queries.py): None = off or not supported by the checkout
    meta["kept_queries_threshold"] = infer.get_kept_threshold() if hasattr(infer, "get_kept_threshold") else None
    meta["torch_deterministic_algorithms"] = bool(getattr(torch, "are_deterministic_algorithms_enabled", lambda: None)())
    want_gpu = str(args.device).startswith("cuda")
    if want_gpu and not any(d.startswith("cuda") for d in devs):
        raise SystemExit("asked for %s but the model parameters are on %s" % (args.device, devs))
    return infer, model


def split_result(result):
    """mmdet 2.x result -> boxes (N,5), labels (N,), masks list of HxW bool."""
    import numpy as np
    if isinstance(result, dict):
        if "ins_results" not in result:
            raise ValueError("result dict without ins_results: keys %s" % list(result))
        result = result["ins_results"]
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], (tuple, dict)):
        result = result[0]
        if isinstance(result, dict):
            result = result["ins_results"]
    bbox_res, mask_res = result[0], result[1]
    boxes, labels, masks = [], [], []
    for cls_i, (b, m) in enumerate(zip(bbox_res, mask_res)):
        b = np.asarray(b).reshape(-1, 5)
        if len(m) != len(b):
            raise ValueError("class %d: %d boxes but %d masks" % (cls_i, len(b), len(m)))
        boxes.append(b)
        labels.extend([cls_i] * len(b))
        masks.extend([np.asarray(x, dtype=bool) for x in m])
    boxes = np.concatenate(boxes, 0) if boxes else np.zeros((0, 5), np.float32)
    return boxes, np.asarray(labels, dtype=np.int64), masks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--images", required=True, help="list file (path or id<TAB>path per line) or a directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--msda", choices=["auto", "compiled", "pytorch"], default=None)
    ap.add_argument("--kept-only", type=float, default=None, metavar="THR",
                    help="infer.load_model(kept_only=THR): only queries with class score >= THR are post-processed "
                         "and saved (the .npz then lacks the instances below THR). Default: the repo default")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images of the list")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-instrument", action="store_true")
    args = ap.parse_args(argv)

    import numpy as np
    import cv2

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    items = common.read_image_list(args.images)
    if args.limit:
        items = items[: args.limit]
    images_arg = str(Path(args.images).resolve())
    meta = {
        "tag": args.tag or out.name,
        "argv": sys.argv,
        "repo": str(Path(args.repo).resolve()),
        "config": str(Path(args.config).resolve()),
        "ckpt": str(Path(args.ckpt).resolve()),
        "ckpt_sha256": common.sha256_file(args.ckpt),
        "device_requested": args.device,
        "msda_requested": args.msda,
        "kept_only_requested": args.kept_only,
        "images_list": images_arg,
        "image_ids": [i for i, _ in items],
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
    }
    meta["versions"] = _versions()
    try:
        import subprocess
        meta["repo_git"] = subprocess.run(["git", "-C", meta["repo"], "rev-parse", "HEAD"], capture_output=True,
                                          text=True).stdout.strip()
        meta["repo_git_dirty"] = subprocess.run(["git", "-C", meta["repo"], "status", "--porcelain"],
                                                capture_output=True, text=True).stdout.strip()
    except Exception as e:
        meta["repo_git"] = "unavailable: %r" % e
    common.write_json(out / "run_meta.json", meta)

    infer, model = load(args, meta)
    from mmdet.apis import inference_detector
    inst = None
    if not args.no_instrument:
        inst = Instrument()
        inst.attach(model)
    common.write_json(out / "run_meta.json", meta)

    import torch
    n_err = 0
    first = True
    per_image = []
    for iid, path in items:
        paths = common.run_paths(out, iid)
        if args.resume and paths["meta"].exists():
            try:
                old = common.read_json(paths["meta"])
                if old.get("status") == "ok":
                    per_image.append({"id": iid, "status": "ok", "resumed": True})
                    continue
            except Exception:
                pass
        rec = {"id": iid, "path": path, "warmup": first, "status": "error"}
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            rec["file_sha256"] = common.sha256_file(path)
            img = cv2.imread(path)
            if img is None:
                raise ValueError("cv2.imread returned None for %s" % path)
            rec["pixels_sha256"] = common.sha256_array(img)
            rec["shape"] = list(img.shape)
            with torch.no_grad():
                _sync(args.device)
                t = time.perf_counter()
                result = inference_detector(model, img)
                _sync(args.device)
                rec["t_detector_s"] = time.perf_counter() - t
            boxes, labels, masks = split_result(result)
            H, W = img.shape[:2]
            for m in masks:
                if m.shape != (H, W):
                    raise ValueError("mask shape %s != image %s" % (m.shape, (H, W)))
            marr = np.stack(masks, 0) if masks else np.zeros((0, H, W), bool)
            common.save_instances(paths["npz"], boxes, labels, marr)
            rec["n_instances"] = int(len(boxes))
            rec["n_ge_0.3"] = int((boxes[:, 4] >= 0.3).sum())
            rec["n_gt_0.3"] = int((boxes[:, 4] > 0.3).sum())
            _sync(args.device)
            t = time.perf_counter()
            ds, ds_masks = infer.get_dataseries(img, to_clean=False, return_masks=True)
            _sync(args.device)
            rec["t_dataseries_s"] = time.perf_counter() - t
            rec["n_lines"] = len(ds)
            rec["n_dataseries_masks"] = len(ds_masks)
            common.write_json(paths["ds"], ds)
            rec["status"] = "ok"
        except Exception:
            rec["error"] = traceback.format_exc()
            n_err += 1
            print(rec["error"], file=sys.stderr, flush=True)
        common.write_json(paths["meta"], rec)
        per_image.append({k: rec.get(k) for k in ("id", "status", "warmup", "t_detector_s", "t_dataseries_s",
                                                  "n_instances", "n_ge_0.3", "n_lines")})
        print("%s %s det %.2fs ds %.2fs inst>=0.3 %s lines %s" % (
            iid, rec["status"], rec.get("t_detector_s", -1), rec.get("t_dataseries_s", -1), rec.get("n_ge_0.3"),
            rec.get("n_lines")), flush=True)
        first = False
        if inst is not None:
            meta["instrument"] = inst.report()
        meta["per_image"] = per_image
        common.write_json(out / "run_meta.json", meta)

    det = [r["t_detector_s"] for r in per_image if r.get("status") == "ok" and not r.get("warmup")
           and r.get("t_detector_s") is not None]
    dst = [r["t_dataseries_s"] for r in per_image if r.get("status") == "ok" and not r.get("warmup")
           and r.get("t_dataseries_s") is not None]
    meta["timing_summary"] = {
        "excludes": "warm-up image and resumed images",
        "n": len(det),
        "detector_s_median": float(np.median(det)) if det else None,
        "detector_s_mean": float(np.mean(det)) if det else None,
        "dataseries_s_median": float(np.median(dst)) if dst else None,
        "dataseries_s_mean": float(np.mean(dst)) if dst else None,
    }
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        meta["cuda_max_memory_allocated_MB"] = torch.cuda.max_memory_allocated() / 2 ** 20
    if inst is not None:
        meta["instrument"] = inst.report()
    meta["n_errors"] = n_err
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["status"] = "done" if n_err == 0 else "done_with_errors"
    common.write_json(out / "run_meta.json", meta)
    print("done: %d images, %d errors, detector median %s s" % (len(items), n_err,
                                                                 meta["timing_summary"]["detector_s_median"]))
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
