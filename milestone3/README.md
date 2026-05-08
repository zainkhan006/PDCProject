# PDC Project 21 — Parallel Soft Clustering for Clinical Notes
## IBA Karachi | Spring 2026 | Complete Reproduction Guide

This guide tells you exactly what to do, in order, to build and run everything
and reproduce every result in the Milestone 2 and 3 reports.

---

## File Listing

```
fcm_mpi.h                    ← header: structs, constants, prototypes
fcm_mpi.c                    ← full parallel FCM implementation (C)
main_mpi.c                   ← entry point: argument parsing, timing
Makefile                     ← build and run targets
evaluate_metrics.py          ← Silhouette, Davies-Bouldin, ARI (Python)
generate_weak_data.py        ← true weak scaling dataset generator
snomed_validate.py           ← SNOMED CT concept normalisation validation
run_scaling_experiments.py   ← automated full experiment suite
README.md                    ← this file
```

**Data files (from Member 2 — NOT included, must be obtained separately):**
```
features.csv                 ← 4943 × 500 TF-IDF feature matrix
specialty_labels.csv         ← one specialty per document (with header)
feature_names.csv            ← column index → clinical term name
```

---

## Step 1: Environment Setup

### Option A — WSL2 on Windows (recommended)
```bash
# Open Ubuntu from Start menu, then:
sudo apt update
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev python3 python3-numpy

# Verify
mpicc --version      # should print GCC version
mpirun --version     # should print Open MPI 4.x
python3 -c "import numpy; print(numpy.__version__)"
```

### Option B — Docker (used in original experiments)
```bash
docker run -it --rm -v $(pwd):/workspace ubuntu:22.04 bash
# Inside container:
apt update && apt install -y openmpi-bin libopenmpi-dev python3 python3-numpy mpicc
cd /workspace
```

### Option C — Native Linux
```bash
sudo apt install -y openmpi-bin libopenmpi-dev python3-numpy
```

---

## Step 2: Place Data Files

Copy Member 2's output files into the same directory as the source files:
```
features.csv
specialty_labels.csv
feature_names.csv          (optional but enables named top-terms output)
```

The feature matrix must be a plain CSV with 4943 rows and 500 columns,
no header, values in [0, 1], L2-normalised.

---

## Step 3: Build

```bash
make
# Expected: "Build successful --> ./fcm_mpi"
```

If `make` fails:
```bash
# Manual compile
mpicc -O2 -std=c99 -o fcm_mpi fcm_mpi.c main_mpi.c -lm
```

---

## Step 4: Basic Run

```bash
# K-Means++ init, 4 processes, block distribution, baseline comm
mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 1 0 0
```

**Command-line arguments:**
```
arg 1: features.csv          path to feature matrix
arg 2: specialty_labels.csv  path to labels file
arg 3: init strategy         0=random  1=kmeanspp  2=domain-guided
arg 4: distribution          0=block   1=cyclic    2=dynamic
arg 5: communication         0=baseline  1=non-blocking
```

**Expected output (first few lines):**
```
╔══════════════════════════════════════════════════╗
║  Parallel FCM — Clinical Notes (OpenMPI)        ║
╚══════════════════════════════════════════════════╝
[main] Features : features.csv
[main] Strategy : K-Means++
[main] Dist mode: Block
[rank 0] Loaded 4943 x 500 feature matrix from 'features.csv'
[member4] Row balance: min=1235 max=1236 avg=1235.75 max/avg=1.000
[init] K-Means++ init complete.
[fcm_mpi] N=4943  F=500  C=10  P=4  m=2.0  eps=1e-05
[fcm_mpi] iter    1  delta=...  time(avg)=...s  comp=...s  comm=...s  imbalance=...
...
[fcm_mpi] Membership saved -> membership_mpi_kmeanspp.csv
[fcm_mpi] Centroids saved  -> centroids_mpi_kmeanspp.csv
[member4] viz_top_terms.csv saved.
[member4] viz_membership_sample.csv saved.
```

---

## Step 5: Run All Three Init Strategies

```bash
make run_random    # random init
make run           # K-Means++ (recommended)
make run_domain    # domain-guided (uses specialty labels)
```

Or manually:
```bash
mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 0 0 0
mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 1 0 0
mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 2 0 0
```

---

## Step 6: Clustering Quality Metrics (Silhouette, DB, ARI)

```bash
python3 evaluate_metrics.py \
    --membership membership_mpi_kmeanspp.csv \
    --centroids  centroids_mpi_kmeanspp.csv  \
    --features   features.csv                \
    --labels     specialty_labels.csv        \
    --sample     2000
```

This computes:
- **Silhouette coefficient** (sampled, ~30 seconds)
- **Davies-Bouldin index** (full dataset, fast)
- **Adjusted Rand Index** (vs specialty labels) — *this was missing before*
- **Per-specialty dominant ratio and purity**

---

## Step 7: SNOMED CT Concept Validation

```bash
# First run fcm_mpi to generate viz_top_terms.csv, then:
python3 snomed_validate.py \
    --top_terms  viz_top_terms.csv    \
    --feat_names feature_names.csv   \
    --output     cluster_snomed_map.csv
```

This maps cluster top-terms to canonical SNOMED CT concepts and codes.
Satisfies the "clinical concept normalisation to standard vocabulary" requirement.

---

## Step 8: Strong Scaling Experiments

```bash
make scaling
```

Or manually:
```bash
for P in 1 2 4 8; do
    echo "=== P=$P ==="
    mpirun --oversubscribe -np $P \
        ./fcm_mpi features.csv specialty_labels.csv 1 0 0
done
```

**What to look at:** The `[member4] Total compute time / communication time` lines
at the end of each run. Record `Avg time/iter` for speedup calculation:
```
speedup(P) = avg_iter(P=1) / avg_iter(P)
efficiency  = speedup / P × 100%
```

---

## Step 9: Distribution Strategy Comparison

```bash
make dist_compare NP=4
make dist_compare NP=8
```

Or manually:
```bash
# Block
mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 1 0 0
# Cyclic
mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 1 1 0
# Dynamic
mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 1 2 0
```

---

## Step 10: Communication Mode Comparison

```bash
make comm_compare NP=4
```

Or manually:
```bash
# Baseline (non-blocking numerator + blocking denominator)
mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 1 0 0
# Fully non-blocking (both reductions concurrent)
mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 1 0 1
```

---

## Step 11: Vary Cluster Count (k=5, 10, 20, 50)

```bash
make vary_k
```

This automatically recompiles with each N_CLUSTERS value and runs.
Or manually for each k, edit `fcm_mpi.h` line:
```c
#define N_CLUSTERS   10    ← change this to 5, 10, 20, or 50
```
Then:
```bash
make clean && make
mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 1 0 0
```

---

## Step 12: True Weak Scaling

```bash
# Generate proportionally sized datasets
python3 generate_weak_data.py --suite features.csv
# This creates: features_weak_P1.csv (N=1236)
#               features_weak_P2.csv (N=2472)
#               features.csv         (N=4943, P=4)
#               features_weak_P8.csv (N=9886, tiled with noise)

# Then run:
mpirun --oversubscribe -np 1 ./fcm_mpi features_weak_P1.csv specialty_labels.csv 1 0 0
mpirun --oversubscribe -np 2 ./fcm_mpi features_weak_P2.csv specialty_labels.csv 1 0 0
mpirun --oversubscribe -np 4 ./fcm_mpi features.csv           specialty_labels.csv 1 0 0
mpirun --oversubscribe -np 8 ./fcm_mpi features_weak_P8.csv specialty_labels.csv 1 0 0
```

**What to look at:** `Avg time/iter` should stay approximately constant
as P increases (each rank always has ~1236 documents).

---

## Step 13: Run Full Experiment Suite (Automated)

```bash
python3 run_scaling_experiments.py
# Runs everything and saves to scaling_results.csv
# Takes ~20-40 minutes depending on hardware
```

Skip slow parts:
```bash
python3 run_scaling_experiments.py --skip_weak --skip_vary_k
```

---

## Output Files Reference

| File | Contents | Used by |
|------|----------|---------|
| `membership_mpi_random.csv` | 4943×10 memberships | evaluate_metrics.py |
| `membership_mpi_kmeanspp.csv` | 4943×10 memberships | evaluate_metrics.py |
| `membership_mpi_domain.csv` | 4943×10 memberships | evaluate_metrics.py |
| `centroids_mpi_kmeanspp.csv` | 10×500 centroids | evaluate_metrics.py |
| `viz_top_terms.csv` | top-10 terms per cluster | snomed_validate.py |
| `viz_membership_sample.csv` | 120-doc sample | heatmap notebook |
| `cluster_snomed_map.csv` | SNOMED mapping | report |
| `scaling_results.csv` | all experiment timings | report tables |

---

## Troubleshooting

**"make: command not found"**
```bash
apt install -y make
```

**"mpirun: command not found"**
```bash
apt install -y openmpi-bin
```

**"Cannot open features.csv"**
→ Copy Member 2's data files into the same folder as the binary.

**"Open MPI has detected insufficient slots"**
→ Always use `--oversubscribe` when P > physical cores.

**Convergence in 1 iteration with K-Means++**
→ This can happen if centroids land exactly on data points.
  Switch to random init for timing benchmarks:
```bash
mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 0 0 0
```
  Random init produces more iterations and realistic timing.
  K-Means++ is better for clustering quality; random is better for timing tests.

**P=8 slower than P=4 on a 4-core machine**
→ This is expected (oversubscription). On a real cluster with 8 cores,
  P=8 would be faster. The `--oversubscribe` flag forces MPI to share cores.

---

## Expected Results Summary

| Experiment | Expected Value |
|---|---|
| Strong scaling speedup at P=4 | ~2.5–3.0× (73% efficiency) |
| Strong scaling speedup at P=8 (real cluster) | ~4.5–5.2× (65% efficiency) |
| Communication fraction | 13–20% per iteration |
| Rank imbalance (max/avg) | < 1.01 (block/cyclic) |
| Silhouette (sparse TF-IDF) | ~0.01–0.03 (expected, not a bug) |
| Davies-Bouldin | 5–15 |
| ARI vs specialty labels | 0.05–0.15 |
| Weighted purity | 0.45–0.55 |
| Cyclic vs block speedup | 12–15% faster |
| Non-blocking comm savings | 5–7% faster |
