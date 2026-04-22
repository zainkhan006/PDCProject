#!/usr/bin/env bash
set -euo pipefail

LOGDIR="run_logs_member4_latest"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

make clean > "$LOGDIR/make_clean.log" 2>&1
make > "$LOGDIR/make.log" 2>&1

for p in 1 2 4 8; do
  mpirun -np "$p" ./fcm_mpi features.csv specialty_labels.csv 1 0 0 > "$LOGDIR/baseline_np${p}.log" 2>&1
done

for dist in 0 1 2; do
  mpirun -np 4 ./fcm_mpi features.csv specialty_labels.csv 1 "$dist" 0 > "$LOGDIR/member4_dist${dist}.log" 2>&1
done

python3 - <<'PY'
import json, re
from pathlib import Path

logdir = Path("run_logs_member4_latest")

def to_num(s):
    if s is None:
        return None
    s = s.strip()
    if s.lower() == "inf":
        return "inf"
    try:
        if re.fullmatch(r"[-+]?\d+", s):
            return int(s)
        return float(s)
    except Exception:
        return s


def extract(text, pattern, cast=True):
    m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    if not m:
        return None
    if len(m.groups()) == 1:
        return to_num(m.group(1)) if cast else m.group(1)
    vals = [to_num(g) if cast else g for g in m.groups()]
    return vals


def parse_log(path):
    t = path.read_text(errors="ignore")
    row_vals = extract(t, r"Row balance:\s*min=(\d+)\s+max=(\d+)\s+avg=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+max/avg=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|inf)")
    comp_vals = extract(t, r"Total compute time .*?:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*s\s*\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)%\)")
    comm_vals = extract(t, r"Total communication time .*?:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*s\s*\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)%\)")

    d = {
        "iterations": extract(t, r"Iterations\s*:\s*(\d+)"),
        "final_delta": extract(t, r"Final delta:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
        "total_wall_time_s": extract(t, r"Total wall time\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*s"),
        "avg_iter_s": extract(t, r"Avg time/iter\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*s"),
        "worst_iter_imbalance": extract(t, r"Worst iter imbalance \(max/avg\):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|inf)"),
        "row_min": row_vals[0] if row_vals else None,
        "row_max": row_vals[1] if row_vals else None,
        "row_avg": row_vals[2] if row_vals else None,
        "total_compute_s": comp_vals[0] if comp_vals else None,
        "total_comm_s": comm_vals[0] if comm_vals else None,
        "compute_percent": comp_vals[1] if comp_vals else None,
        "comm_percent": comm_vals[1] if comm_vals else None,
    }
    return d

baseline_scaling = []
for p in [1, 2, 4, 8]:
    metrics = parse_log(logdir / f"baseline_np{p}.log")
    baseline_scaling.append({"np": p, **metrics})

member4_modes = []
for dist in [0, 1, 2]:
    metrics = parse_log(logdir / f"member4_dist{dist}.log")
    member4_modes.append({"dist": dist, **metrics})

np4_time = next((x.get("total_wall_time_s") for x in baseline_scaling if x.get("np") == 4), None)
out = {
    "baseline_scaling": baseline_scaling,
    "member4_modes": member4_modes,
    "baseline_np4_time_s": np4_time,
}
print(json.dumps(out, separators=(",", ":")))
PY