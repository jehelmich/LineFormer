# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""Unit tests for compare.py on synthetic run directories (no model needed).

pytest tools/equivalence/tests -q      or, without pytest:  python tools/equivalence/tests/test_compare.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
import compare  # noqa: E402

H, W = 60, 80


def _line_mask(y0, x0=5, x1=75, thick=3, slope=0.0):
    m = np.zeros((H, W), bool)
    for x in range(x0, x1):
        y = int(round(y0 + slope * (x - x0)))
        m[max(0, y - thick // 2): y + thick // 2 + 1, x] = True
    return m


def _ds_from_mask(m):
    pts = []
    for x in range(W):
        ys = np.nonzero(m[:, x])[0]
        if len(ys):
            pts.append({"x": int(x), "y": int(round(ys.mean()))})
    return pts


def _write_run(d, tag, images):
    """images: {id: (scores list, masks list)}"""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for iid, (scores, masks) in images.items():
        p = common.run_paths(d, iid)
        boxes = np.zeros((len(scores), 5), np.float32)
        for k, (s, m) in enumerate(zip(scores, masks)):
            ys, xs = np.nonzero(m)
            boxes[k] = [xs.min(), ys.min(), xs.max(), ys.max(), s]
        common.save_instances(p["npz"], boxes, np.zeros(len(scores)), np.stack(masks))
        ds = [_ds_from_mask(m) for s, m in zip(scores, masks) if s > 0.3]
        common.write_json(p["ds"], ds)
        common.write_json(p["meta"], {"id": iid, "status": "ok", "file_sha256": "x", "pixels_sha256": "y",
                                      "shape": [H, W, 3]})
    common.write_json(d / "run_meta.json", {"tag": tag, "status": "done", "image_ids": list(images)})


def _base():
    return {"img1": ([0.9, 0.8, 0.1], [_line_mask(10), _line_mask(30, slope=0.2), _line_mask(50)])}


class _Tmp:
    def __enter__(self):
        self.d = Path(tempfile.mkdtemp())
        return self.d

    def __exit__(self, *a):
        shutil.rmtree(str(self.d), ignore_errors=True)


def test_pack_roundtrip():
    with _Tmp() as t:
        m = np.random.RandomState(0).rand(3, 7, 13) > 0.5
        common.save_instances(t / "a.npz", np.zeros((3, 5)), np.zeros(3), m)
        assert np.array_equal(common.load_instances(t / "a.npz")["masks"], m)


def test_identical_runs_pass_with_iou_1():
    with _Tmp() as t:
        _write_run(t / "A", "A", _base())
        _write_run(t / "B", "B", _base())
        res = compare.compare_runs(t / "A", t / "B")
        s = res["summary"]
        assert s["verdict"] == "PASS", s["failures"]
        img = res["images"][0]
        assert img["instances"]["n_matched"] == 2
        assert img["instances"]["min_iou"] == 1.0
        assert img["instances"]["max_dscore"] == 0.0
        assert img["dataseries"]["frac_within_px"] == 1.0
        assert img["all_instances"]["masks_bit_identical"]


def test_dropped_instance_fails():
    with _Tmp() as t:
        _write_run(t / "A", "A", _base())
        b = _base()
        b["img1"] = ([0.9, 0.1], [b["img1"][1][0], b["img1"][1][2]])
        _write_run(t / "B", "B", b)
        res = compare.compare_runs(t / "A", t / "B")
        assert res["summary"]["verdict"] == "FAIL"
        reasons = " ".join(res["images"][0]["reasons"])
        assert "instance count" in reasons and "unmatched" in reasons and "line count" in reasons


def test_shifted_mask_detected():
    with _Tmp() as t:
        _write_run(t / "A", "A", _base())
        b = _base()
        b["img1"][1][1] = _line_mask(32, slope=0.2)  # 2 px down: 3-px-thick line -> IoU 0.2
        _write_run(t / "B", "B", b)
        res = compare.compare_runs(t / "A", t / "B")
        img = res["images"][0]
        assert not img["pass"]
        assert img["instances"]["n_matched"] == 2
        assert img["instances"]["min_iou"] < 0.98
        assert any("IoU" in r for r in img["reasons"])
        assert any("points within" in r for r in img["reasons"])  # 2 px shift > 1 px tolerance


def test_score_change_detected_and_small_one_passes():
    with _Tmp() as t:
        _write_run(t / "A", "A", _base())
        b = _base()
        b["img1"][0][0] = 0.88
        _write_run(t / "B", "B", b)
        assert not compare.compare_runs(t / "A", t / "B")["images"][0]["pass"]
        b["img1"][0][0] = 0.895
        _write_run(t / "C", "C", b)
        assert compare.compare_runs(t / "A", t / "C")["images"][0]["pass"]


def test_missing_output_is_failure_not_skip():
    with _Tmp() as t:
        two = _base()
        two["img2"] = two["img1"]
        _write_run(t / "A", "A", two)
        _write_run(t / "B", "B", two)
        (t / "B" / "img2.npz").unlink()
        res = compare.compare_runs(t / "A", t / "B")
        assert res["summary"]["n_images"] == 2 and res["summary"]["n_fail"] == 1
        assert "missing output" in res["images"][1]["reasons"][0]
        # an image only the ref requested is a failure too
        _write_run(t / "C", "C", _base())
        res = compare.compare_runs(t / "A", t / "C")
        assert res["summary"]["verdict"] == "FAIL"
        # but restricting to the candidate's list is explicit and recorded
        res = compare.compare_runs(t / "A", t / "C", expected="cand")
        assert res["summary"]["verdict"] == "PASS" and res["summary"]["expected_set"] == "cand"


def test_dataseries_one_px_tolerance():
    a = [[{"x": x, "y": 10} for x in range(100)]]
    b = [[{"x": x, "y": 11 if x < 50 else 10} for x in range(100)]]
    assert compare.compare_dataseries(a, b)["frac_within_px"] == 1.0
    c = [[{"x": x, "y": 12 if x < 2 else 10} for x in range(100)]]
    r = compare.compare_dataseries(a, c)
    assert abs(r["frac_within_px"] - 0.98) < 1e-9 and r["n_matched_lines"] == 1


def test_same_directory_is_refused():
    with _Tmp() as t:
        _write_run(t / "A", "A", _base())
        try:
            compare.compare_runs(t / "A", t / "A")
        except ValueError:
            return
        raise AssertionError("comparing a run with itself must raise")


def test_different_input_pixels_fail():
    with _Tmp() as t:
        _write_run(t / "A", "A", _base())
        _write_run(t / "B", "B", _base())
        p = common.run_paths(t / "B", "img1")["meta"]
        m = common.read_json(p)
        m["pixels_sha256"] = "other"
        common.write_json(p, m)
        img = compare.compare_runs(t / "A", t / "B")["images"][0]
        assert not img["pass"] and "pixels" in " ".join(img["reasons"])


def test_kept_only_candidate_passes_and_is_reported_as_subset():
    """A kept-queries run lacks the instances below 0.3: no failure, and the subset is reported as bit-identical."""
    with _Tmp() as t:
        _write_run(t / "A", "A", _base())
        b = _base()
        b["img1"] = (b["img1"][0][:2], b["img1"][1][:2])
        _write_run(t / "B", "B", b)
        res = compare.compare_runs(t / "A", t / "B")
        assert res["summary"]["verdict"] == "PASS", res["summary"]["failures"]
        sub = res["images"][0]["cand_subset_of_ref"]
        assert sub["is_bit_identical_subset"] and sub["n_kept"] == 2 and sub["order_preserved"]
        assert sub["ref_above_thr_missing"] == [] and abs(sub["max_ref_score_dropped"] - 0.1) < 1e-6
        s = res["summary"]["cand_subset_of_ref"]
        assert s["n_images_bit_identical_subset"] == 1 and s["n_ref_above_thr_missing"] == 0
        # a changed mask is not part of the bit-identical subset (and a full run is a subset of itself)
        b["img1"][1][1] = _line_mask(30, slope=0.21)
        _write_run(t / "C", "C", b)
        sub = compare.compare_runs(t / "A", t / "C")["images"][0]["cand_subset_of_ref"]
        assert not sub["is_bit_identical_subset"] and sub["n_kept_mask_identical"] == 1
        assert sub["ref_above_thr_missing"] == [1]
        _write_run(t / "D", "D", _base())
        assert compare.compare_runs(t / "A", t / "D")["images"][0]["cand_subset_of_ref"]["is_bit_identical_subset"]


def test_kept_only_candidate_missing_an_instance_above_threshold_fails():
    with _Tmp() as t:
        _write_run(t / "A", "A", _base())
        b = _base()
        b["img1"] = ([0.9], [b["img1"][1][0]])
        _write_run(t / "B", "B", b)
        res = compare.compare_runs(t / "A", t / "B")
        assert res["summary"]["verdict"] == "FAIL"
        assert res["images"][0]["cand_subset_of_ref"]["ref_above_thr_missing"] == [1]


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("ok", name)
    print("%d tests passed" % n)
