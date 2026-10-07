"""Merge gpu_engine_sampler.ps1 samples into a batch_infer.py run_meta.json (throughput.gpu_sampled).

python merge_samples.py <run_meta.json> <samples.csv>
Samples strictly inside [window start + 1 s, window end - 1 s] (counter lag, clock offset). Per engine key the
mean utilisation; the engine with the highest mean is reported as the busy one (the WSL GPU work), the others are
listed. host_cpu = Windows host CPU % (includes the WSL VM and any Windows load).
"""
import csv
import json
import sys
from collections import defaultdict


def main(meta_path, csv_path):
    meta = json.load(open(meta_path))
    a, b = meta["throughput"]["window_unix"]
    a, b = a + 1.0, b - 1.0
    per = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                t, v = float(r["unix_time"]), float(r["value"])
            except (ValueError, KeyError, TypeError):
                continue
            if a <= t <= b:
                per[r["key"]].append(v)
    means = {k: sum(v) / len(v) for k, v in per.items()}
    host = means.pop("host_cpu", None)
    eng = sorted(means.items(), key=lambda kv: -kv[1])
    out = {"n_samples_host": len(per.get("host_cpu", [])), "host_cpu_pct_mean": host,
           "busiest_engine": eng[0][0] if eng else None,
           "busiest_engine_util_pct_mean": eng[0][1] if eng else None,
           "busiest_engine_util_pct_min_max": [min(per[eng[0][0]]), max(per[eng[0][0]])] if eng else None,
           "other_engines_top": [{"key": k, "mean": v} for k, v in eng[1:6]],
           "note": "Windows GPU Engine utilisation (all processes summed per engine), ~1 s samples"}
    meta["throughput"]["gpu_sampled"] = out
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
