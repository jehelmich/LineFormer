"""Collect the throughput runs into results.json and a markdown table.

python summarize.py <throughput dir> [baseline name]
Reads <dir>/runs/*/run_meta.json and <dir>/compare/{A,C}_vs_<name>.json.
"""
import json
import sys
from pathlib import Path


def verdict(p):
    if not p.exists():
        return None
    s = json.load(open(p))["summary"]
    return {"verdict": s["verdict"], "n_pass": s["n_pass"], "n_images": s["n_images"],
            "min_iou": s["global_min_iou"], "max_dscore": s["global_max_dscore"], "min_ds_frac": s["global_min_ds_frac"],
            "n_masks_bit_identical": s["n_images_masks_bit_identical"],
            "n_dataseries_identical": s["n_images_dataseries_identical"], "failures": s["failures"][:10],
            "run_problems": s["run_problems"]}


def main(d, base="base_getds", *names):
    d = Path(d)
    rows = {}
    metas = [d / "runs" / n / "run_meta.json" for n in names] if names else sorted((d / "runs").glob("*/run_meta.json"))
    for m in metas:
        meta = json.load(open(m))
        name = m.parent.name
        t = meta.get("throughput", {})
        g = t.get("gpu_sampled") or {}
        gw = t.get("gpu_workers") or {}
        rows[name] = {
            "config": meta.get("config_throughput"), "status": meta.get("status"),
            "images_per_s": t.get("images_per_s"), "wall_s": t.get("wall_s"), "n_timed": t.get("n_timed"),
            "n_errors": t.get("n_errors"),
            "latency_s_median": (t.get("latency_s") or {}).get("median"),
            "gpu_stage_s_median": (t.get("gpu_stage_s") or {}).get("median"),
            "compute_s_per_image": t.get("compute_s_per_image"), "d2h_s_per_image": t.get("d2h_s_per_image"),
            "d2h_share_of_forward": t.get("d2h_share_of_forward"),
            "gpu_stage_occupancy_frac": t.get("gpu_stage_occupancy_frac"),
            "gpu_compute_engine_util_pct": g.get("busiest_engine_util_pct_mean"),
            "gpu_busiest_engine": g.get("busiest_engine"),
            "gpu_copy_engines": [e for e in g.get("other_engines_top", []) if "copy" in e["key"]],
            "host_cpu_pct": g.get("host_cpu_pct_mean"),
            "wsl_cpu_busy_pct": 100 * t["wsl_cpu_busy_frac"] if t.get("wsl_cpu_busy_frac") is not None else None,
            "cpu_s_by_role": t.get("cpu_s_by_role"), "main_cpu_s": t.get("main_cpu_s"),
            "loadavg_start": t.get("loadavg_start"), "loadavg_end": t.get("loadavg_end"),
            "vram_max_allocated_MB_per_proc": t.get("max_memory_allocated_MB_max"),
            "vram_reserved_MB_sum": t.get("max_memory_reserved_MB_sum"),
            "batch_size_hist": {k: v.get("batch_size_hist") for k, v in gw.items()},
            "timed_ds_identical_to_pass0": t.get("timed_dataseries_identical_to_pass0"),
            "timed_ds_identical_to_pass0_line_order_free": t.get("timed_dataseries_identical_to_pass0_line_order_free"),
            "vs_A": verdict(d / "compare" / ("A_vs_%s.json" % name)),
            "vs_C": verdict(d / "compare" / ("C_vs_%s.json" % name)),
        }
    b = rows.get(base, {}).get("images_per_s")
    for r in rows.values():
        r["speedup_vs_baseline"] = (r["images_per_s"] / b) if (b and r["images_per_s"]) else None
    json.dump({"baseline": base, "runs": rows}, open(d / "results.json", "w"), indent=1)

    def f(v, fmt):
        return "-" if v is None else fmt % v

    def acc(v):
        return "-" if v is None else "%s %d/%d" % (v["verdict"], v["n_pass"], v["n_images"])
    lines = ["| config | images/s | speed-up | latency med s | fwd compute / d2h s per img | GPU compute engine % "
             "| GPU stage occupancy % | VRAM alloc max MB / reserved sum MB | WSL CPU % | host CPU % | vs A | vs C |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in rows.items():
        lines.append("| %s | %s | %s | %s | %s / %s | %s | %s | %s / %s | %s | %s | %s | %s |" % (
            n, f(r["images_per_s"], "%.2f"), f(r["speedup_vs_baseline"], "%.2fx"), f(r["latency_s_median"], "%.2f"),
            f(r["compute_s_per_image"], "%.3f"), f(r["d2h_s_per_image"], "%.3f"),
            f(r["gpu_compute_engine_util_pct"], "%.0f"),
            f(r["gpu_stage_occupancy_frac"] and 100 * r["gpu_stage_occupancy_frac"], "%.0f"),
            f(r["vram_max_allocated_MB_per_proc"], "%.0f"), f(r["vram_reserved_MB_sum"], "%.0f"),
            f(r["wsl_cpu_busy_pct"], "%.0f"), f(r["host_cpu_pct"], "%.0f"), acc(r["vs_A"]), acc(r["vs_C"])))
    table = "\n".join(lines)
    (d / "table.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main(*sys.argv[1:])
