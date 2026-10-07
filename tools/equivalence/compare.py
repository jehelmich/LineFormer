"""Compare two run directories written by run.py, image by image, against fixed acceptance criteria.

Acceptance ("behaviour unchanged", fixed before measuring), on every image:
  1. Same number of instances with score >= 0.3, matched one-to-one by mask IoU.
  2. Every matched pair: mask IoU >= 0.98 and |score difference| <= 0.01.
  3. get_dataseries: same number of lines; >= 99 % of points within 1 px.
An image with a missing or failed output in either run fails ("never silently skipped").

Instance matching: instances with score >= 0.3 of each run; IoU matrix from sparse masks; one-to-one assignment
maximising total IoU (scipy linear_sum_assignment). An assigned pair with IoU 0 counts as unmatched.
The same matching over ALL instances (no threshold) is reported as information, not judged.
Near threshold: instances whose score is within 0.02 of 0.3 in either run (they explain count changes).
Kept-queries mode (kept_queries.py) returns only the instances whose class score reaches 0.3, so a candidate in
that mode has fewer instances in total; the verdict only looks at instances >= 0.3 and is not affected. As
information, "cand_subset_of_ref" pairs every candidate instance with a reference instance of identical mask
bytes and counts identical boxes and scores (per image and in the summary).

Dataseries matching: each line is a list of {x, y}. For a pair of lines (ref i, cand j) the cost is the mean
|y_ref - y_cand| over the x values both lines have (a line with several points at one x uses, per point, the
nearest y of the other line at that x); a pair with no shared x is not matchable. Lines are matched one-to-one by
linear_sum_assignment on that cost. Point agreement: a ref point counts as "within 1 px" when its matched cand
line has a point at the same x with |dy| <= 1; points of unmatched lines and points at x values the matched line
lacks count as misses. The same is done from the cand side; the reported fraction is the minimum of both
directions, over all points of the image (an image with no points in either run has fraction 1.0).

--expected: which image ids must be present. "union" (default): every id requested by either run; "cand": only
the ids the candidate run requested (for a partial re-run such as a determinism check); "ref": the reference's.
The choice is printed in the summary. Python 3.8 compatible.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

CRITERIA = {
    "score_thr": 0.3,
    "min_iou": 0.98,
    "max_dscore": 0.01,
    "min_points_within_1px": 0.99,
    "px_tol": 1,
    "near_thr_band": 0.02,
}


# ---------------------------------------------------------------- instances

def to_sparse(masks):
    """(N,H,W) bool -> CSR (N, H*W) float64 (built from the nonzero indices, much faster than from dense)."""
    from scipy import sparse
    n = len(masks)
    flat = masks.reshape(n, -1)
    r, c = np.nonzero(flat)
    return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=flat.shape)


def iou_matrix(ma, mb):
    """ma, mb: (N,H,W) / (M,H,W) bool arrays or CSR matrices from to_sparse -> (N,M) IoU."""
    n, m = ma.shape[0], mb.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m))
    if ma.shape[1:] != mb.shape[1:]:
        raise ValueError("mask sizes differ: %s vs %s" % (ma.shape, mb.shape))
    A = ma if hasattr(ma, "tocsr") else to_sparse(ma)
    B = mb if hasattr(mb, "tocsr") else to_sparse(mb)
    inter = np.asarray((A @ B.T).todense())
    sa = np.asarray(A.sum(1)).reshape(-1, 1)
    sb = np.asarray(B.sum(1)).reshape(1, -1)
    union = sa + sb - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / union, 1.0)  # two empty masks are identical
    return iou


def match_instances(ma, sa, mb, sb):
    """-> list of (i, j, iou, dscore), unmatched_a, unmatched_b."""
    from scipy.optimize import linear_sum_assignment
    iou = iou_matrix(ma, mb)
    pairs = []
    if iou.size:
        r, c = linear_sum_assignment(-iou)
        for i, j in zip(r, c):
            if iou[i, j] > 0:
                pairs.append((int(i), int(j), float(iou[i, j]), float(abs(sa[i] - sb[j]))))
    ua = sorted(set(range(ma.shape[0])) - set(p[0] for p in pairs))
    ub = sorted(set(range(mb.shape[0])) - set(p[1] for p in pairs))
    return pairs, ua, ub


def kept_subset(rb, rm, kb, km, thr):
    """Is the candidate (kb boxes (K,5), km masks) a subset of the reference instances (rb, rm) with bit-identical
    masks? For a run in kept-queries mode (kept_queries.py) against a full run. Instances are paired by the bytes
    of their masks (each ref instance used once). Information only, never part of the verdict."""
    import hashlib

    def h(m):
        m = np.asarray(m, dtype=bool)
        return hashlib.sha1(np.packbits(m).tobytes() + str(m.shape).encode()).hexdigest()
    by_hash = {}
    for i, m in enumerate(rm):
        by_hash.setdefault(h(m), []).append(i)
    idx, used = [], set()
    for m in km:
        free = [i for i in by_hash.get(h(m), []) if i not in used]
        idx.append(free[0] if free else None)
        if free:
            used.add(free[0])
    matched = [(j, i) for j, i in enumerate(idx) if i is not None]
    dscore = [abs(float(kb[j, 4]) - float(rb[i, 4])) for j, i in matched]
    above = set(int(i) for i in np.nonzero(np.asarray(rb)[:, 4] >= thr)[0]) if len(rb) else set()
    ri = [i for _, i in matched]
    dropped = [float(rb[i, 4]) for i in range(len(rb)) if i not in used]
    return {
        "n_ref": int(len(rb)), "n_kept": int(len(kb)),
        "n_kept_mask_identical": len(matched),
        "n_kept_box_identical": sum(1 for j, i in matched if np.array_equal(kb[j, :4], rb[i, :4])),
        "n_kept_score_identical": sum(1 for d in dscore if d == 0.0),
        "max_abs_dscore": max(dscore, default=0.0),
        "order_preserved": ri == sorted(ri),
        "n_ref_above_thr": len(above),
        "ref_above_thr_missing": sorted(above - used),
        "max_ref_score_dropped": max(dropped) if dropped else None,
    }


# ---------------------------------------------------------------- dataseries

def _line_dict(line):
    d = {}
    for p in line:
        d.setdefault(int(p["x"]), []).append(float(p["y"]))
    return d


def _pair_cost(da, db):
    xs = set(da) & set(db)
    if not xs:
        return None
    dist = []
    for x in xs:
        yb = np.asarray(db[x])
        dist.extend(float(np.min(np.abs(yb - ya))) for ya in da[x])
    return float(np.mean(dist))


def _points_within(da, db, tol):
    """number of points of da that have a point of db at the same x within tol px."""
    k = 0
    for x, ys in da.items():
        if x not in db:
            continue
        yb = np.asarray(db[x])
        k += sum(1 for y in ys if np.min(np.abs(yb - y)) <= tol)
    return k


def compare_dataseries(dsa, dsb, tol=1):
    from scipy.optimize import linear_sum_assignment
    la = [_line_dict(l) for l in dsa]
    lb = [_line_dict(l) for l in dsb]
    na = [sum(len(v) for v in d.values()) for d in la]
    nb = [sum(len(v) for v in d.values()) for d in lb]
    cost = np.full((len(la), len(lb)), np.inf)
    for i, da in enumerate(la):
        for j, db in enumerate(lb):
            c = _pair_cost(da, db)
            if c is not None:
                cost[i, j] = c
    pairs = []
    if cost.size and np.isfinite(cost).any():
        big = np.nanmax(cost[np.isfinite(cost)]) * 10 + 1e6
        r, c = linear_sum_assignment(np.where(np.isfinite(cost), cost, big))
        pairs = [(int(i), int(j), float(cost[i, j])) for i, j in zip(r, c) if np.isfinite(cost[i, j])]
    within_a = sum(_points_within(la[i], lb[j], tol) for i, j, _ in pairs)
    within_b = sum(_points_within(lb[j], la[i], tol) for i, j, _ in pairs)
    fa = within_a / sum(na) if sum(na) else 1.0
    fb = within_b / sum(nb) if sum(nb) else 1.0
    if not sum(na) and sum(nb):
        fa = 0.0
    if not sum(nb) and sum(na):
        fb = 0.0
    return {
        "n_lines_ref": len(la),
        "n_lines_cand": len(lb),
        "n_matched_lines": len(pairs),
        "n_points_ref": int(sum(na)),
        "n_points_cand": int(sum(nb)),
        "frac_within_px_ref_side": fa,
        "frac_within_px_cand_side": fb,
        "frac_within_px": min(fa, fb),
        "line_pairs": [{"ref": i, "cand": j, "mean_abs_dy": c} for i, j, c in pairs],
        "max_line_mean_abs_dy": max([c for _, _, c in pairs], default=0.0),
    }


# ---------------------------------------------------------------- one image

def compare_image(ref_dir, cand_dir, iid, crit=CRITERIA):
    res = {"id": iid, "pass": False, "reasons": [], "notes": []}
    metas = {}
    for side, d in (("ref", ref_dir), ("cand", cand_dir)):
        p = common.run_paths(d, iid)
        missing = [k for k in ("npz", "ds", "meta") if not p[k].exists()]
        if missing:
            res["reasons"].append("%s: missing output %s" % (side, ",".join(missing)))
            continue
        m = common.read_json(p["meta"])
        metas[side] = m
        if m.get("status") != "ok":
            res["reasons"].append("%s: run status %s" % (side, m.get("status")))
    if res["reasons"]:
        return res
    if metas["ref"].get("file_sha256") != metas["cand"].get("file_sha256"):
        res["notes"].append("input file_sha256 differs between runs (same pixels is still a valid comparison)")
    if metas["ref"].get("pixels_sha256") != metas["cand"].get("pixels_sha256"):
        res["reasons"].append("the runs read different input pixels (pixels_sha256 differs)")
        return res
    res["shape"] = metas["ref"].get("shape")
    pr, pc = common.run_paths(ref_dir, iid), common.run_paths(cand_dir, iid)
    A = common.load_instances(pr["npz"])
    B = common.load_instances(pc["npz"])
    if A["masks"].shape[1:] != B["masks"].shape[1:]:
        res["reasons"].append("mask size differs %s vs %s" % (A["masks"].shape[1:], B["masks"].shape[1:]))
        return res
    thr = crit["score_thr"]
    ia = np.nonzero(A["scores"] >= thr)[0]
    ib = np.nonzero(B["scores"] >= thr)[0]
    SA, SB = to_sparse(A["masks"]), to_sparse(B["masks"])
    pairs, ua, ub = match_instances(SA[ia], A["scores"][ia], SB[ib], B["scores"][ib])
    inst = {
        "n_ref": int(len(ia)),
        "n_cand": int(len(ib)),
        "n_matched": len(pairs),
        "pairs": [{"ref": int(ia[i]), "cand": int(ib[j]), "iou": iou, "dscore": ds,
                   "score_ref": float(A["scores"][ia[i]]), "score_cand": float(B["scores"][ib[j]]),
                   "max_box_dcoord": float(np.max(np.abs(A["boxes"][ia[i], :4] - B["boxes"][ib[j], :4])))}
                  for i, j, iou, ds in pairs],
        "unmatched_ref": [{"idx": int(ia[i]), "score": float(A["scores"][ia[i]]),
                           "area": int(A["masks"][ia[i]].sum())} for i in ua],
        "unmatched_cand": [{"idx": int(ib[j]), "score": float(B["scores"][ib[j]]),
                            "area": int(B["masks"][ib[j]].sum())} for j in ub],
    }
    inst["min_iou"] = min([p["iou"] for p in inst["pairs"]], default=None)
    inst["max_dscore"] = max([p["dscore"] for p in inst["pairs"]], default=None)
    band = crit["near_thr_band"]
    inst["near_threshold"] = {
        "ref": [{"idx": int(i), "score": float(s)} for i, s in enumerate(A["scores"]) if abs(s - thr) <= band],
        "cand": [{"idx": int(i), "score": float(s)} for i, s in enumerate(B["scores"]) if abs(s - thr) <= band],
    }
    res["instances"] = inst
    # all instances, information only
    pa, uaa, uba = match_instances(SA, A["scores"], SB, B["scores"])
    ious = np.array([p[2] for p in pa]) if pa else np.zeros(0)
    dss = np.array([p[3] for p in pa]) if pa else np.zeros(0)
    res["all_instances"] = {
        "n_ref": int(len(A["scores"])), "n_cand": int(len(B["scores"])), "n_matched": len(pa),
        "n_unmatched_ref": len(uaa), "n_unmatched_cand": len(uba),
        "min_iou": float(ious.min()) if ious.size else None,
        "n_iou_below_min": int((ious < crit["min_iou"]).sum()),
        "max_dscore": float(dss.max()) if dss.size else None,
        "n_dscore_above_max": int((dss > crit["max_dscore"]).sum()),
        "masks_bit_identical": bool(A["masks"].shape == B["masks"].shape and np.array_equal(A["masks"], B["masks"])),
        "scores_identical": bool(A["scores"].shape == B["scores"].shape and np.array_equal(A["scores"], B["scores"])),
    }
    # a candidate in kept-queries mode returns only part of the instances: is it a bit-identical subset?
    sub = kept_subset(A["boxes"], A["masks"], B["boxes"], B["masks"], thr)
    sub["is_bit_identical_subset"] = bool(sub["n_kept_mask_identical"] == sub["n_kept"] and
                                          sub["n_kept_box_identical"] == sub["n_kept"] and
                                          sub["n_kept_score_identical"] == sub["n_kept"])
    res["cand_subset_of_ref"] = sub
    # dataseries
    dsa = common.read_json(pr["ds"])
    dsb = common.read_json(pc["ds"])
    ds = compare_dataseries(dsa, dsb, tol=crit["px_tol"])
    ds["identical"] = dsa == dsb
    res["dataseries"] = ds

    # verdict
    R = res["reasons"]
    if inst["n_ref"] != inst["n_cand"]:
        R.append("instance count >=%.2f differs: ref %d cand %d" % (thr, inst["n_ref"], inst["n_cand"]))
    if inst["unmatched_ref"] or inst["unmatched_cand"]:
        R.append("unmatched instances: ref %s cand %s" % ([u["idx"] for u in inst["unmatched_ref"]],
                                                          [u["idx"] for u in inst["unmatched_cand"]]))
    bad_iou = [p for p in inst["pairs"] if p["iou"] < crit["min_iou"]]
    if bad_iou:
        R.append("%d matched masks with IoU < %.2f (min %.4f)" % (len(bad_iou), crit["min_iou"], inst["min_iou"]))
    bad_s = [p for p in inst["pairs"] if p["dscore"] > crit["max_dscore"]]
    if bad_s:
        R.append("%d matched pairs with |dscore| > %.2f (max %.4f)" % (len(bad_s), crit["max_dscore"],
                                                                        inst["max_dscore"]))
    if ds["n_lines_ref"] != ds["n_lines_cand"]:
        R.append("dataseries line count differs: ref %d cand %d" % (ds["n_lines_ref"], ds["n_lines_cand"]))
    if ds["frac_within_px"] < crit["min_points_within_1px"]:
        R.append("dataseries: %.2f%% of points within %d px (< %.0f%%)" % (
            100 * ds["frac_within_px"], crit["px_tol"], 100 * crit["min_points_within_1px"]))
    if inst["near_threshold"]["ref"] or inst["near_threshold"]["cand"]:
        res["notes"].append("instances within %.2f of the threshold: ref %d cand %d" % (
            band, len(inst["near_threshold"]["ref"]), len(inst["near_threshold"]["cand"])))
    res["pass"] = not R
    return res


def _subset_summary(images):
    subs = [r["cand_subset_of_ref"] for r in images if "cand_subset_of_ref" in r]
    out = {k: sum(s[k] for s in subs) for k in ("n_kept", "n_kept_mask_identical", "n_kept_box_identical",
                                                "n_kept_score_identical", "n_ref_above_thr")}
    out.update(n_images=len(subs),
               n_images_bit_identical_subset=sum(1 for s in subs if s["is_bit_identical_subset"]),
               n_images_order_preserved=sum(1 for s in subs if s["order_preserved"]),
               n_ref_above_thr_missing=sum(len(s["ref_above_thr_missing"]) for s in subs),
               max_abs_dscore=max((s["max_abs_dscore"] for s in subs), default=None),
               max_ref_score_dropped=max((s["max_ref_score_dropped"] for s in subs
                                          if s["max_ref_score_dropped"] is not None), default=None),
               note="information only: cand instances paired with ref instances by identical mask bytes")
    return out


def compare_runs(ref, cand, expected="union"):
    if Path(ref).resolve() == Path(cand).resolve():
        raise ValueError("ref and cand are the same run directory (%s): the comparison would pass vacuously" % ref)
    mr =common.read_json(Path(ref) / "run_meta.json") if (Path(ref) / "run_meta.json").exists() else {}
    mc = common.read_json(Path(cand) / "run_meta.json") if (Path(cand) / "run_meta.json").exists() else {}
    problems = []
    if not mr:
        problems.append("ref run_meta.json missing")
    if not mc:
        problems.append("cand run_meta.json missing")
    ids_r, ids_c = mr.get("image_ids", []), mc.get("image_ids", [])
    if expected == "union":
        ids = list(ids_r) + [i for i in ids_c if i not in ids_r]
    elif expected == "cand":
        ids = list(ids_c)
    elif expected == "ref":
        ids = list(ids_r)
    else:
        raise ValueError(expected)
    if not ids:
        problems.append("no image ids to compare (expected=%s)" % expected)
    for side, m in (("ref", mr), ("cand", mc)):
        if m and m.get("status") not in ("done",):
            problems.append("%s run status is %r" % (side, m.get("status")))
    images = [compare_image(ref, cand, iid) for iid in ids]
    n_pass = sum(1 for r in images if r["pass"])

    def key(r):
        v = (r.get("instances") or {}).get("min_iou")
        return 2.0 if v is None and r["pass"] else (-1.0 if v is None else v)

    worst_iou = sorted(images, key=key)[:10]
    worst_ds = sorted([r for r in images if "dataseries" in r], key=lambda r: r["dataseries"]["frac_within_px"])[:10]
    pairs_all = [p for r in images for p in (r.get("instances") or {}).get("pairs", [])]
    summary = {
        "ref": str(Path(ref).resolve()), "cand": str(Path(cand).resolve()),
        "ref_tag": mr.get("tag"), "cand_tag": mc.get("tag"),
        "expected_set": expected, "n_images": len(images), "n_pass": n_pass, "n_fail": len(images) - n_pass,
        "verdict": "PASS" if (n_pass == len(images) and images and not problems) else "FAIL",
        "run_problems": problems,
        "criteria": CRITERIA,
        "n_matched_pairs_total": len(pairs_all),
        "global_min_iou": min([p["iou"] for p in pairs_all], default=None),
        "global_max_dscore": max([p["dscore"] for p in pairs_all], default=None),
        "global_min_ds_frac": min([r["dataseries"]["frac_within_px"] for r in images if "dataseries" in r],
                                  default=None),
        "n_images_masks_bit_identical": sum(1 for r in images if (r.get("all_instances") or {}).get(
            "masks_bit_identical")),
        "n_images_scores_identical": sum(1 for r in images if (r.get("all_instances") or {}).get(
            "scores_identical")),
        "n_images_dataseries_identical": sum(1 for r in images if (r.get("dataseries") or {}).get("identical")),
        "cand_subset_of_ref": _subset_summary(images),
        "n_images_input_differs": sum(1 for r in images if any("input" in n for n in r.get("notes", []))),
        "worst_min_iou": [{"id": r["id"], "min_iou": (r.get("instances") or {}).get("min_iou"),
                           "pass": r["pass"]} for r in worst_iou],
        "worst_dataseries": [{"id": r["id"], "frac": r["dataseries"]["frac_within_px"]} for r in worst_ds],
        "failures": [{"id": r["id"], "reasons": r["reasons"]} for r in images if not r["pass"]],
        "ref_meta": {k: mr.get(k) for k in ("tag", "device_requested", "msda_handling", "versions",
                                            "model_param_devices", "instrument", "timing_summary",
                                            "t_model_load_s", "repo_git", "status", "timing_caveat",
                                            "cuda_max_memory_allocated_MB", "infer_msda_report")},
        "cand_meta": {k: mc.get(k) for k in ("tag", "device_requested", "msda_handling", "versions",
                                             "model_param_devices", "instrument", "timing_summary",
                                             "t_model_load_s", "repo_git", "status", "timing_caveat",
                                            "cuda_max_memory_allocated_MB", "infer_msda_report")},
    }
    return {"summary": summary, "images": images}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out", required=True, help="json file")
    ap.add_argument("--expected", choices=["union", "cand", "ref"], default="union")
    args = ap.parse_args(argv)
    res = compare_runs(args.ref, args.cand, args.expected)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    common.write_json(args.out, res)
    s = res["summary"]
    print("%s vs %s: %s  pass %d / %d (expected set: %s)" % (s["ref_tag"], s["cand_tag"], s["verdict"], s["n_pass"],
                                                             s["n_images"], s["expected_set"]))
    print("  min IoU %s  max |dscore| %s  min dataseries frac %s" % (s["global_min_iou"], s["global_max_dscore"],
                                                                     s["global_min_ds_frac"]))
    print("  bit-identical masks on %d images, identical scores on %d, identical dataseries on %d" % (
        s["n_images_masks_bit_identical"], s["n_images_scores_identical"], s["n_images_dataseries_identical"]))
    sub = s["cand_subset_of_ref"]
    if sub["n_kept"] < sum(r["all_instances"]["n_ref"] for r in res["images"] if "all_instances" in r):
        print("  cand returns fewer instances (kept-queries mode?): %d of its %d instances are bit-identical ref "
              "instances, scores identical %d (max |d| %s); ref instances >= %.2f missing: %d (info)" % (
                  sub["n_kept_mask_identical"], sub["n_kept"], sub["n_kept_score_identical"], sub["max_abs_dscore"],
                  CRITERIA["score_thr"], sub["n_ref_above_thr_missing"]))
    for p in s["run_problems"]:
        print("  RUN PROBLEM:", p)
    for f in s["failures"]:
        print("  FAIL %s: %s" % (f["id"], "; ".join(f["reasons"])))
    return 0 if s["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
