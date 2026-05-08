"""
generate_weak_data.py  —  Generate proportionally sized datasets for weak scaling
PDC Project 21 | IBA Spring 2026

PROBLEM FIXED:
  The original weak scaling experiment subsampled the same 4943-document dataset
  to N=1236, 2472, 4943. That is NOT true weak scaling — it just runs fewer docs
  with more ranks, which mixes convergence differences with communication overhead.

TRUE WEAK SCALING requires:
  Each rank always processes ~1236 documents.
  P=1  → N=1236
  P=2  → N=2472
  P=4  → N=4943  (the real dataset)
  P=8  → N=9886  (need to synthesise extra docs)

This script produces the subsampled/extended datasets needed.
For P=1,2,4 it subsamples/uses features.csv.
For P=8 it tiles the dataset (wraps around) to produce N=9886 with
the same vocabulary distribution.

Usage:
    python3 generate_weak_data.py features.csv <N_target> <output.csv>

    # Or run the full suite:
    python3 generate_weak_data.py --suite features.csv
"""

import sys
import numpy as np
import argparse
import os

def load_features(path):
    print(f"[load] Reading {path} ...", flush=True)
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split(',')])
            if (i+1) % 500 == 0:
                print(f"  ... {i+1} rows loaded", flush=True)
    return np.array(rows, dtype=np.float64)

def save_features(X, path):
    print(f"[save] Writing {X.shape[0]} rows to {path} ...", flush=True)
    with open(path, 'w') as f:
        for row in X:
            f.write(','.join(f'{v:.8f}' for v in row) + '\n')
    print(f"[save] Done: {path}")

def subsample(X, N, seed=42):
    """Random subsample of N rows without replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=N, replace=False)
    idx.sort()
    return X[idx]

def tile_to(X, N, seed=42):
    """
    Tile dataset to reach N rows.
    For each extra document needed, take an existing document and add
    small Gaussian noise to simulate a new but related clinical note.
    This preserves vocabulary distribution while creating genuinely new rows.
    """
    rng = np.random.default_rng(seed)
    existing = len(X)
    needed = N - existing
    if needed <= 0:
        return X[:N]

    # Pick random base rows and add noise
    base_idx = rng.choice(existing, size=needed, replace=True)
    noise_scale = 0.01  # small relative to TF-IDF values in [0,1]
    noise = rng.normal(0, noise_scale, (needed, X.shape[1]))
    new_rows = np.clip(X[base_idx] + noise, 0.0, 1.0)

    # Re-normalise to L2 norm = 1 (matching original TF-IDF property)
    norms = np.linalg.norm(new_rows, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    new_rows = new_rows / norms

    combined = np.vstack([X, new_rows])
    print(f"[tile] Generated {needed} synthetic rows (L2-normalised, noise={noise_scale})")
    return combined

def suite(feat_path):
    """
    Generate all datasets needed for weak scaling:
      P=1  N=1236  → subsample
      P=2  N=2472  → subsample
      P=4  N=4943  → original
      P=8  N=9886  → tile with noise
    """
    X = load_features(feat_path)
    N_full = len(X)
    base = 1236   # docs per rank

    configs = [
        (1, base * 1),
        (2, base * 2),
        (4, N_full),     # use full dataset for P=4
        (8, base * 8),
    ]

    for P, N in configs:
        out = f"features_weak_P{P}.csv"
        if os.path.exists(out):
            print(f"[skip] {out} already exists.")
            continue
        if N <= N_full:
            Xout = subsample(X, N, seed=42 + P)
        else:
            Xout = tile_to(X, N, seed=42 + P)
        save_features(Xout, out)
        print(f"[suite] P={P:2d}  N={N:6d}  -> {out}")

    print("\n[suite] Complete. Run weak scaling with:")
    print("  mpirun --oversubscribe -np 1 ./fcm_mpi features_weak_P1.csv specialty_labels.csv 1 0 0")
    print("  mpirun --oversubscribe -np 2 ./fcm_mpi features_weak_P2.csv specialty_labels.csv 1 0 0")
    print("  mpirun --oversubscribe -np 4 ./fcm_mpi features.csv           specialty_labels.csv 1 0 0")
    print("  mpirun --oversubscribe -np 8 ./fcm_mpi features_weak_P8.csv specialty_labels.csv 1 0 0")
    print("\nFor true weak scaling, compare Avg time/iter across runs — it should stay ~constant.")

def main():
    if '--suite' in sys.argv:
        idx = sys.argv.index('--suite')
        feat_path = sys.argv[idx+1] if idx+1 < len(sys.argv) else 'features.csv'
        suite(feat_path)
        return

    ap = argparse.ArgumentParser()
    ap.add_argument('features',  help='Input features.csv')
    ap.add_argument('N',         type=int, help='Target number of rows')
    ap.add_argument('output',    help='Output CSV path')
    ap.add_argument('--seed',    type=int, default=42)
    args = ap.parse_args()

    X = load_features(args.features)
    N_full = len(X)

    if args.N <= N_full:
        Xout = subsample(X, args.N, seed=args.seed)
        print(f"[subsample] {N_full} → {args.N} rows")
    else:
        Xout = tile_to(X, args.N, seed=args.seed)
        print(f"[tile] {N_full} → {args.N} rows")

    save_features(Xout, args.output)

if __name__ == '__main__':
    main()
