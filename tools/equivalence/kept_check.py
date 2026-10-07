# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""In-process check of the kept-queries mode (kept_queries.py) against the unpatched mmdet path, image by image.

One model, one process: per image inference_detector runs unpatched (R, all instances), then with the mode on (K),
then once more unpatched (R2, to see the device's own run-to-run noise). Checks:
  - every instance of K is an instance of R with a bit-identical mask (matched by mask bytes), in R's order;
  - its box is identical and its score identical (or the difference is reported);
  - every instance of R with score >= --thr is in K (the instances the threshold keeps are all there);
  - infer.get_dataseries is identical with the mode on and off.
Output: a JSON file and one line per image. Exit 1 if a kept instance has no bit-identical mask in R, an instance
of R above the threshold is missing from K, or the dataseries differ. Python 3.8 compatible.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import run as runmod  # noqa: E402
from compare import kept_subset  # noqa: E402


def compare_kept(R, K, thr):
    """R, K: (boxes (N,5), labels, masks list) from run.split_result. -> compare.kept_subset's dict."""
    return kept_subset(R[0], R[2], K[0], K[2], thr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--msda", choices=["auto", "compiled", "pytorch"], default=None)
    ap.add_argument("--images", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--out", required=True, help="json file")
    ap.add_argument("--mem-fraction", type=float, default=None,
                    help="cap this process's GPU memory (torch.cuda.set_per_process_memory_fraction) on a shared card")
    args = ap.parse_args(argv)
    args.kept_only = False  # load unpatched; the mode is switched per call below
    import cv2
    import torch
    if args.mem_fraction and str(args.device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, torch.device(args.device))
    from mmdet.apis import inference_detector
    items = common.read_image_list(args.images)
    if args.limit:
        items = items[: args.limit]
    meta = {"device_requested": args.device, "thr": args.thr, "versions": runmod._versions()}
    infer, model = runmod.load(args, meta)
    import kept_queries
    rows, bad = [], 0
    for iid, path in items:
        img = cv2.imread(path)
        if img is None:
            raise SystemExit("cannot read %s" % path)
        with torch.no_grad():
            kept_queries.disable(model)
            R = runmod.split_result(inference_detector(model, img))
            ds_off = infer.get_dataseries(img, to_clean=False)
            kept_queries.enable(model, args.thr)
            K = runmod.split_result(inference_detector(model, img))
            ds_on = infer.get_dataseries(img, to_clean=False)
            kept_queries.disable(model)
            R2 = runmod.split_result(inference_detector(model, img))
        r = {"id": iid, "shape": list(img.shape)}
        r.update(compare_kept(R, K, args.thr))
        r["dataseries_identical"] = ds_on == ds_off
        r["unpatched_repeat_identical"] = bool(
            len(R[0]) == len(R2[0]) and (R[0] == R2[0]).all() and all((a == b).all() for a, b in zip(R[2], R2[2])))
        r["ok"] = (r["n_kept_mask_identical"] == r["n_kept"] and not r["ref_above_thr_missing"]
                   and r["dataseries_identical"])
        bad += not r["ok"]
        rows.append(r)
        print(iid, {k: r[k] for k in ("n_ref", "n_kept", "n_kept_mask_identical", "n_kept_score_identical",
                                      "max_abs_dscore", "order_preserved", "ref_above_thr_missing",
                                      "dataseries_identical", "unpatched_repeat_identical", "ok")}, flush=True)
    summ = {k: sum(r[k] for r in rows) for k in ("n_kept", "n_kept_mask_identical", "n_kept_box_identical",
                                                 "n_kept_score_identical", "n_ref_above_thr")}
    summ.update(n_images=len(rows), n_ok=len(rows) - bad,
                max_abs_dscore=max((r["max_abs_dscore"] for r in rows), default=0.0),
                n_images_order_preserved=sum(r["order_preserved"] for r in rows),
                n_images_dataseries_identical=sum(r["dataseries_identical"] for r in rows),
                n_images_unpatched_repeat_identical=sum(r["unpatched_repeat_identical"] for r in rows),
                max_ref_score_dropped=max((r["max_ref_score_dropped"] or 0.0 for r in rows), default=0.0))
    meta.update(summary=summ, images=rows)
    common.write_json(args.out, meta)
    print("summary", summ)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
