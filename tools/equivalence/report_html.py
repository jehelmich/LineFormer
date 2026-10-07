# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 LineFormer fork contributors (https://github.com/jehelmich/LineFormer)
"""Static inspection site for one or more compare.py results.

python report_html.py --compare AvsB.json --compare AvsC.json [...] --out site/
then: python -m http.server --directory site/ <port>

index.html: one section per run pair (verdict, criteria, devices, versions, timings) with a table of every image
(pass/fail, counts, min IoU, max |dscore|, dataseries agreement, reasons); "fail only" filter; click a numeric
column header to sort (min IoU sorts ascending first).
img/<id>.html: the input; the reference overlay; per pair the candidate overlay, a diff map and a dataseries
overlay, and the metrics.

Drawing rules:
- Overlays show instances with score >= 0.3, filled, labelled "#<index> <score>"; instances within 0.02 below the
  threshold are drawn grey. A candidate instance takes the colour of the reference instance it is matched to;
  an unmatched one is drawn magenta (candidate) or keeps its colour with an "UNMATCHED" label (reference).
- Diff map: the input dimmed; for every matched pair green = both, red = reference only, blue = candidate only
  (red/blue dilated 7x7 so single pixels show; the pixel counts on the page are undilated);
  unmatched reference instances orange, unmatched candidate instances purple. Downscaling keeps any pixel set in
  the full-resolution layer (area resize, > 0), so a one-pixel disagreement stays visible.
- Dataseries overlay: reference points green, candidate points magenta (drawn on top, smaller).
All files are local (no CDN), paths are relative, no browser dialogs. Python 3.8 compatible; needs cv2 + numpy.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

MAX_W = 1400
THR = 0.3
BAND = 0.02


def local_path(p):
    """Translate between Windows (C:\\...) and WSL (/mnt/c/...) spellings when the given one does not exist."""
    if p is None:
        return None
    if os.path.exists(p):
        return p
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m:
        q = "/mnt/%s/%s" % (m.group(1).lower(), m.group(2).replace("\\", "/"))
        if os.path.exists(q):
            return q
    m = re.match(r"^/mnt/([a-z])/(.*)$", p)
    if m:
        q = "%s:\\%s" % (m.group(1).upper(), m.group(2).replace("/", "\\"))
        if os.path.exists(q):
            return q
    return p


def slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def palette(k):
    """distinct BGR colour for the k-th instance (golden-angle hue steps)."""
    import cv2
    hsv = np.uint8([[[(k * 47) % 180, 220, 230]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _scale(shape):
    h, w = shape[:2]
    return min(1.0, MAX_W / float(w))


def _down_mask(m, s):
    import cv2
    if s >= 1.0:
        return m
    h, w = m.shape
    r = cv2.resize(m.astype(np.float32), (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                   interpolation=cv2.INTER_AREA)
    return r > 0


def _down_img(img, s):
    import cv2
    if s >= 1.0:
        return img.copy()
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)


def _dim(img):
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = (255 - (255 - g.astype(np.float32)) * 0.35).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _label(img, text, xy, col):
    import cv2
    x, y = int(xy[0]), int(max(12, xy[1]))
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


def _first_px(m):
    ys, xs = np.nonzero(m)
    if not len(xs):
        return (0, 0)
    k = int(np.argmin(xs))
    return xs[k], ys[k] - 4


def draw_overlay(base_small, s, inst, colours, unmatched_label=None, default_col=(255, 0, 255)):
    """inst: load_instances dict; colours: {idx: bgr}; returns image."""
    vis = _dim(base_small)
    order = np.argsort(inst["scores"])  # draw high scores last
    labels = []
    for i in order:
        sc = inst["scores"][i]
        if sc < THR - BAND:
            continue
        m = _down_mask(inst["masks"][i], s)
        if sc < THR:
            col = (150, 150, 150)
        else:
            col = colours.get(int(i), default_col)
        vis[m] = col
        txt = "#%d %.3f" % (i, sc)
        if unmatched_label and int(i) in unmatched_label:
            txt += " UNMATCHED"
        labels.append((txt, _first_px(m), col))
    placed = []
    for t, (x, y), col in labels:  # shift a label down while it would cover an earlier one
        y = max(12, int(y))
        while any(abs(x - px) < 9 * len(t) and abs(y - py) < 14 for px, py in placed):
            y += 14
        placed.append((x, y))
        _label(vis, t, (x, y), col)
    return vis


def _grow(m, k=7):
    import cv2
    if not m.any():
        return m
    return cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def draw_diff(base_small, s, A, B, img_res):
    vis = _dim(base_small)
    inst = img_res["instances"]
    counts = {"both": 0, "ref_only": 0, "cand_only": 0}
    for p in inst["pairs"]:
        ma, mb = A["masks"][p["ref"]], B["masks"][p["cand"]]
        both, ro, co = ma & mb, ma & ~mb, mb & ~ma
        counts["both"] += int(both.sum())
        counts["ref_only"] += int(ro.sum())
        counts["cand_only"] += int(co.sum())
        vis[_down_mask(both, s)] = (60, 170, 60)
        # disagreeing pixels are dilated (7x7) before drawing so that a one-pixel difference is visible
        vis[_down_mask(_grow(ro), s)] = (0, 0, 230)
        vis[_down_mask(_grow(co), s)] = (230, 80, 0)
    for u in inst["unmatched_ref"]:
        vis[_down_mask(A["masks"][u["idx"]], s)] = (0, 140, 255)
    for u in inst["unmatched_cand"]:
        vis[_down_mask(B["masks"][u["idx"]], s)] = (200, 0, 160)
    return vis, counts


def draw_ds(base_small, s, dsa, dsb):
    import cv2
    vis = _dim(base_small)
    for ds, col, r in ((dsa, (40, 170, 40), 2), (dsb, (220, 0, 220), 1)):
        for line in ds:
            for p in line:
                cv2.circle(vis, (int(p["x"] * s), int(p["y"] * s)), r, col, -1)
    return vis


def _fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return ("%." + str(nd) + "f") % v
    return html.escape(str(v))


CSS = """
:root{--bg:#fafafa;--fg:#1d1d1f;--mut:#666;--line:#ddd;--pass:#1a7f37;--fail:#c62828;--card:#fff}
@media (prefers-color-scheme: dark){:root{--bg:#161618;--fg:#e8e8ea;--mut:#9a9aa0;--line:#333;--pass:#4caf50;
--fail:#ef5350;--card:#1f1f22}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:20px}h2{font-size:17px;margin-top:28px}h3{font-size:15px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
th{cursor:pointer;user-select:none;position:sticky;top:0;background:var(--bg)}
.pass{color:var(--pass);font-weight:600}.fail{color:var(--fail);font-weight:600}
.mut{color:var(--mut)}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;
margin:10px 0;overflow-x:auto}
img{max-width:100%;height:auto;border:1px solid var(--line)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(420px,100%),1fr));gap:10px}
pre{white-space:pre-wrap;font-size:12px}
a{color:#2b6cb0}
.wrap{overflow-x:auto}
"""

JS = """
function applyFilter(sec){
  const only=document.getElementById('f_'+sec).checked;
  document.querySelectorAll('#t_'+sec+' tbody tr').forEach(tr=>{
    tr.style.display=(only && tr.dataset.pass==='1')?'none':'';});
}
function sortBy(sec,col){
  const tb=document.querySelector('#t_'+sec+' tbody');
  const rows=[...tb.querySelectorAll('tr')];
  const key=tb.dataset.key===String(col)?-1*(+tb.dataset.dir||1):1;
  tb.dataset.key=String(col);tb.dataset.dir=String(key);
  rows.sort((a,b)=>{const x=a.children[col].dataset.v,y=b.children[col].dataset.v;
    const nx=parseFloat(x),ny=parseFloat(y);
    if(!isNaN(nx)&&!isNaN(ny))return key*(nx-ny);return key*String(x).localeCompare(String(y));});
  rows.forEach(r=>tb.appendChild(r));
}
"""


def build(compares, out):
    import cv2
    out = Path(out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "files").mkdir(parents=True, exist_ok=True)
    pairs = []
    for cpath in compares:
        res = common.read_json(cpath)
        s = res["summary"]
        tag = "%s_vs_%s" % (s.get("ref_tag"), s.get("cand_tag"))
        pairs.append({"tag": slug(tag), "label": "%s vs %s" % (s.get("ref_tag"), s.get("cand_tag")),
                      "res": res, "ref": local_path(s["ref"]), "cand": local_path(s["cand"]), "src": str(cpath)})
    all_ids = []
    for p in pairs:
        for r in p["res"]["images"]:
            if r["id"] not in all_ids:
                all_ids.append(r["id"])

    done_input, done_ref = set(), set()
    per_image_html = {}
    for iid in all_ids:
        sections = []
        input_rel = None
        for p in pairs:
            r = next((x for x in p["res"]["images"] if x["id"] == iid), None)
            if r is None:
                continue
            ref_meta_p = common.run_paths(p["ref"], iid)["meta"]
            cand_meta_p = common.run_paths(p["cand"], iid)["meta"]
            path = None
            for mp in (ref_meta_p, cand_meta_p):
                if mp.exists():
                    cand_path = local_path(common.read_json(mp).get("path"))
                    if cand_path and os.path.exists(cand_path):
                        path = cand_path
                        break
            img = cv2.imread(path) if path else None
            block = ["<div class='card'><h3>%s: <span class='%s'>%s</span></h3>" % (
                html.escape(p["label"]), "pass" if r["pass"] else "fail", "PASS" if r["pass"] else "FAIL")]
            if r["reasons"]:
                block.append("<p><b>Reasons:</b> %s</p>" % "<br>".join(html.escape(x) for x in r["reasons"]))
            if r.get("notes"):
                block.append("<p class='mut'>Notes: %s</p>" % "; ".join(html.escape(x) for x in r["notes"]))
            if img is None:
                block.append("<p class='fail'>input image not found (%s)</p>" % html.escape(str(path)))
            if img is not None and "instances" in r:
                sc = _scale(img.shape)
                small = _down_img(img, sc)
                if iid not in done_input:
                    cv2.imwrite(str(out / "files" / (slug(iid) + "_input.jpg")), small,
                                [cv2.IMWRITE_JPEG_QUALITY, 88])
                    done_input.add(iid)
                input_rel = "../files/%s_input.jpg" % slug(iid)
                A = common.load_instances(common.run_paths(p["ref"], iid)["npz"])
                B = common.load_instances(common.run_paths(p["cand"], iid)["npz"])
                inst = r["instances"]
                ref_cols = {}
                for k, i in enumerate(sorted(np.nonzero(A["scores"] >= THR)[0], key=lambda i: -A["scores"][i])):
                    ref_cols[int(i)] = palette(k)
                cand_cols = {pp["cand"]: ref_cols.get(pp["ref"], (255, 0, 255)) for pp in inst["pairs"]}
                ref_key = (p["ref"], iid)
                ref_name = "%s_%s_ref.jpg" % (slug(iid), slug(p["res"]["summary"].get("ref_tag")))
                if ref_key not in done_ref:
                    cv2.imwrite(str(out / "files" / ref_name), draw_overlay(
                        small, sc, A, ref_cols, unmatched_label=set(u["idx"] for u in inst["unmatched_ref"])),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
                    done_ref.add(ref_key)
                cand_name = "%s_%s_cand.jpg" % (slug(iid), p["tag"])
                cv2.imwrite(str(out / "files" / cand_name), draw_overlay(small, sc, B, cand_cols),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
                diff, counts = draw_diff(small, sc, A, B, r)
                diff_name = "%s_%s_diff.png" % (slug(iid), p["tag"])
                cv2.imwrite(str(out / "files" / diff_name), diff)
                dsa = common.read_json(common.run_paths(p["ref"], iid)["ds"])
                dsb = common.read_json(common.run_paths(p["cand"], iid)["ds"])
                ds_name = "%s_%s_ds.jpg" % (slug(iid), p["tag"])
                cv2.imwrite(str(out / "files" / ds_name), draw_ds(small, sc, dsa, dsb), [cv2.IMWRITE_JPEG_QUALITY, 88])
                rt, ct = p["res"]["summary"].get("ref_tag"), p["res"]["summary"].get("cand_tag")
                block.append("<div class='grid'>")
                for title, f in (("%s (reference) instances" % rt, ref_name), ("%s instances" % ct, cand_name),
                                 ("diff: green both, red %s only, blue %s only, orange/purple unmatched" % (rt, ct),
                                  diff_name),
                                 ("dataseries: green %s, magenta %s" % (rt, ct), ds_name)):
                    block.append("<figure><figcaption>%s</figcaption><a href='../files/%s'><img src='../files/%s' "
                                 "loading='lazy'></a></figure>" % (html.escape(title), f, f))
                block.append("</div>")
                block.append("<p>Disagreeing pixels over matched pairs: %d %s-only, %d %s-only (%d shared)</p>" % (
                    counts["ref_only"], html.escape(str(rt)), counts["cand_only"], html.escape(str(ct)),
                    counts["both"]))
                rows = "".join("<tr><td>%d</td><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
                               "</tr>" % (q["ref"], q["cand"], _fmt(q["iou"]), _fmt(q["score_ref"]),
                                          _fmt(q["score_cand"]), _fmt(q["dscore"], 5), _fmt(q["max_box_dcoord"], 2))
                               for q in inst["pairs"])
                block.append("<div class='wrap'><table><thead><tr><th>ref #</th><th>cand #</th><th>IoU</th>"
                             "<th>score ref</th><th>score cand</th><th>|dscore|</th><th>max box dcoord px</th>"
                             "</tr></thead><tbody>%s</tbody></table></div>" % rows)
                for side in ("unmatched_ref", "unmatched_cand"):
                    if inst[side]:
                        block.append("<p class='fail'>%s: %s</p>" % (side, html.escape(json.dumps(inst[side]))))
                nt = inst["near_threshold"]
                if nt["ref"] or nt["cand"]:
                    block.append("<p>Near threshold (|score-0.3| &le; 0.02): ref %s; cand %s</p>" % (
                        html.escape(json.dumps(nt["ref"])), html.escape(json.dumps(nt["cand"]))))
                ds = r["dataseries"]
                block.append("<p>Dataseries: lines %d / %d, matched %d, points %d / %d, within 1 px %.4f "
                             "(ref side %.4f, cand side %.4f), identical: %s</p>" % (
                                 ds["n_lines_ref"], ds["n_lines_cand"], ds["n_matched_lines"], ds["n_points_ref"],
                                 ds["n_points_cand"], ds["frac_within_px"], ds["frac_within_px_ref_side"],
                                 ds["frac_within_px_cand_side"], ds["identical"]))
                block.append("<details><summary>all instances (no threshold, information only)</summary><pre>%s"
                             "</pre></details>" % html.escape(json.dumps(r["all_instances"], indent=1)))
            block.append("</div>")
            sections.append("".join(block))
        head = "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' " \
               "content='width=device-width,initial-scale=1'><title>%s</title><style>%s</style></head><body>" % (
                   html.escape(iid), CSS)
        k = all_ids.index(iid)
        nav = "<p><a href='../index.html'>index</a>%s%s</p>" % (
            " | <a href='%s.html'>prev</a>" % slug(all_ids[k - 1]) if k > 0 else "",
            " | <a href='%s.html'>next</a>" % slug(all_ids[k + 1]) if k + 1 < len(all_ids) else "")
        inp = "<figure><figcaption>input</figcaption><img src='%s'></figure>" % input_rel if input_rel else ""
        page = head + nav + "<h1>%s</h1>" % html.escape(iid) + inp + "".join(sections) + nav + "</body></html>"
        (out / "img" / (slug(iid) + ".html")).write_text(page, encoding="utf-8")
        per_image_html[iid] = "img/%s.html" % slug(iid)

    # index
    parts = ["<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
             "content='width=device-width,initial-scale=1'><title>LineFormer equivalence</title><style>%s</style>"
             "<script>%s</script></head><body><h1>LineFormer equivalence</h1>" % (CSS, JS)]
    parts.append("<div class='card'><b>Pairs</b><table><thead><tr><th>pair</th><th>verdict</th><th>pass</th>"
                 "<th>fail</th><th>min IoU</th><th>max |dscore|</th><th>min ds frac</th><th>bit-identical masks"
                 "</th></tr></thead><tbody>")
    for k, p in enumerate(pairs):
        s = p["res"]["summary"]
        parts.append("<tr><td><a href='#sec%d'>%s</a></td><td class='%s'>%s</td><td>%d</td><td>%d</td><td>%s</td>"
                     "<td>%s</td><td>%s</td><td>%d / %d</td></tr>" % (
                         k, html.escape(p["label"]), "pass" if s["verdict"] == "PASS" else "fail", s["verdict"],
                         s["n_pass"], s["n_fail"], _fmt(s["global_min_iou"]), _fmt(s["global_max_dscore"], 5),
                         _fmt(s["global_min_ds_frac"]), s["n_images_masks_bit_identical"], s["n_images"]))
    parts.append("</tbody></table></div>")
    for k, p in enumerate(pairs):
        s = p["res"]["summary"]
        parts.append("<h2 id='sec%d'>%s: <span class='%s'>%s</span></h2>" % (
            k, html.escape(p["label"]), "pass" if s["verdict"] == "PASS" else "fail", s["verdict"]))
        parts.append("<p>%d images (expected set: %s), %d pass, %d fail. Criteria: %s</p>" % (
            s["n_images"], s["expected_set"], s["n_pass"], s["n_fail"], html.escape(json.dumps(s["criteria"]))))
        if s["run_problems"]:
            parts.append("<p class='fail'>Run problems: %s</p>" % html.escape("; ".join(s["run_problems"])))
        parts.append("<details><summary>run metadata (devices, versions, timings, MSDA)</summary><pre>%s</pre>"
                     "</details>" % html.escape(json.dumps({"ref": s["ref_meta"], "cand": s["cand_meta"]}, indent=1)))
        parts.append("<label><input type='checkbox' id='f_%d' onchange='applyFilter(%d)'> fail only</label>" % (k, k))
        cols = ["image", "result", "inst ref", "inst cand", "matched", "min IoU", "max |dscore|", "lines ref",
                "lines cand", "ds within 1px", "reasons / notes"]
        parts.append("<div class='wrap'><table id='t_%d'><thead><tr>%s</tr></thead><tbody>" % (
            k, "".join("<th onclick='sortBy(%d,%d)'>%s</th>" % (k, c, h) for c, h in enumerate(cols))))
        for r in p["res"]["images"]:
            inst = r.get("instances") or {}
            ds = r.get("dataseries") or {}
            miou = inst.get("min_iou")
            cells = [
                ("<a href='%s'>%s</a>" % (per_image_html.get(r["id"], "#"), html.escape(r["id"])), r["id"]),
                ("<span class='%s'>%s</span>" % ("pass" if r["pass"] else "fail", "PASS" if r["pass"] else "FAIL"),
                 1 if r["pass"] else 0),
                (_fmt(inst.get("n_ref")), inst.get("n_ref", -1)), (_fmt(inst.get("n_cand")), inst.get("n_cand", -1)),
                (_fmt(inst.get("n_matched")), inst.get("n_matched", -1)),
                (_fmt(miou), miou if miou is not None else -1),
                (_fmt(inst.get("max_dscore"), 5), inst.get("max_dscore") if inst.get("max_dscore") is not None
                 else -1),
                (_fmt(ds.get("n_lines_ref")), ds.get("n_lines_ref", -1)),
                (_fmt(ds.get("n_lines_cand")), ds.get("n_lines_cand", -1)),
                (_fmt(ds.get("frac_within_px")), ds.get("frac_within_px", -1)),
                (html.escape("; ".join(r["reasons"] + r.get("notes", []))), ""),
            ]
            parts.append("<tr data-pass='%d'>%s</tr>" % (1 if r["pass"] else 0, "".join(
                "<td data-v='%s'>%s</td>" % (html.escape(str(v), quote=True), h) for h, v in cells)))
        parts.append("</tbody></table></div>")
    parts.append("<p class='mut'>Sources: %s</p></body></html>" % html.escape(", ".join(p["src"] for p in pairs)))
    (out / "index.html").write_text("".join(parts), encoding="utf-8")
    return out / "index.html"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", action="append", required=True, help="compare.py json (repeatable)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    idx = build(args.compare, args.out)
    print("written", idx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
