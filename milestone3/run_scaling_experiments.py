"""
run_scaling_experiments.py  —  Automated scaling experiment suite
PDC Project 21 | Member 3 (Zain Khan) | IBA Spring 2026

Runs ALL required experiments and saves results to CSV:
  1. Strong scaling         (P=1,2,4,8 — fixed N)
  2. True weak scaling      (P=1,2,4,8 — N proportional, needs generate_weak_data.py)
  3. Distribution strategy  (block / cyclic / dynamic at P=4 and P=8)
  4. Communication mode     (baseline vs non-blocking at P=4)
  5. Vary cluster count     (k=5,10,20,50 at P=4)

Parses stdout of ./fcm_mpi to extract timing numbers, then
writes a clean results CSV and prints a formatted summary.

Usage:
    python3 run_scaling_experiments.py [--binary ./fcm_mpi] [--np_max 8]
"""

import subprocess
import re
import csv
import time
import os
import sys
import argparse

BINARY  = './fcm_mpi'
FEAT    = 'features.csv'
LABELS  = 'specialty_labels.csv'
MPIRUN  = 'mpirun'

def run(cmd, timeout=600):
    """Run a shell command, capture stdout, return (stdout, elapsed_s)."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"  [warn] Non-zero exit: {result.returncode}")
            print(f"  stderr: {result.stderr[:300]}")
        return result.stdout, elapsed
    except subprocess.TimeoutExpired:
        print(f"  [warn] Timed out after {timeout}s")
        return "", timeout

def parse_output(stdout):
    """Extract key metrics from fcm_mpi stdout."""
    data = {}

    # Total FCM time
    m = re.search(r'FCM total time:\s*([\d.]+)\s*s\s*\((\d+)\s*iters?\)', stdout)
    if m:
        data['total_s']    = float(m.group(1))
        data['iterations'] = int(m.group(2))

    # Average iter time (from last [fcm_mpi] iter line)
    iter_times = re.findall(r'time\(avg\)=([\d.]+)s', stdout)
    if iter_times:
        data['avg_iter_s'] = sum(float(x) for x in iter_times) / len(iter_times)

    # Compute / comm breakdown
    m = re.search(r'Total compute time.*?:([\d.]+)\s*s\s*\(([\d.]+)%\)', stdout)
    if m:
        data['compute_s'] = float(m.group(1))
        data['compute_pct'] = float(m.group(2))
    m = re.search(r'Total communication time.*?:([\d.]+)\s*s\s*\(([\d.]+)%\)', stdout)
    if m:
        data['comm_s']   = float(m.group(1))
        data['comm_pct'] = float(m.group(2))

    # Worst imbalance
    m = re.search(r'Worst iter imbalance.*?:([\d.]+)', stdout)
    if m:
        data['imbalance'] = float(m.group(1))

    # Final delta
    m = re.search(r'final delta.*?([\d.e+-]+)', stdout, re.IGNORECASE)
    if m:
        data['final_delta'] = float(m.group(1))

    # Converged?
    data['converged'] = 'Converged at iter' in stdout

    return data

def mpirun_cmd(np, feat, init=1, dist=0, comm=0):
    return [MPIRUN, '--oversubscribe', '-np', str(np),
            BINARY, feat, LABELS, str(init), str(dist), str(comm)]

def experiment_strong_scaling(np_max=8):
    print("\n" + "="*60)
    print("  EXPERIMENT 1: Strong Scaling (block, kmeanspp, baseline)")
    print("="*60)
    rows = []
    t1 = None
    for P in [1, 2, 4, 8]:
        if P > np_max:
            continue
        print(f"\n--- P={P} ---")
        stdout, wall = run(mpirun_cmd(P, FEAT))
        d = parse_output(stdout)
        if P == 1 and 'avg_iter_s' in d:
            t1 = d['avg_iter_s']
        speedup  = (t1 / d['avg_iter_s']) if t1 and 'avg_iter_s' in d else None
        eff      = (speedup / P * 100) if speedup else None
        row = {'experiment': 'strong_scaling', 'P': P,
               'feat_file': FEAT, 'init': 'kmeanspp',
               'dist': 'block', 'comm': 'baseline',
               **d,
               'speedup': speedup, 'efficiency_pct': eff,
               'wall_elapsed_s': wall}
        rows.append(row)
        print(f"  avg_iter={d.get('avg_iter_s','?'):.4f}s  "
              f"speedup={speedup:.2f}×  eff={eff:.1f}%"
              if speedup else f"  avg_iter={d.get('avg_iter_s','?')}")
    return rows

def experiment_weak_scaling():
    print("\n" + "="*60)
    print("  EXPERIMENT 2: Weak Scaling (true proportional)")
    print("="*60)

    weak_files = {
        1: 'features_weak_P1.csv',
        2: 'features_weak_P2.csv',
        4: FEAT,
        8: 'features_weak_P8.csv',
    }

    # Generate if needed
    for P, fpath in weak_files.items():
        if not os.path.exists(fpath) and fpath != FEAT:
            N = 1236 * P
            print(f"[generate] Creating {fpath} (N={N}) ...")
            subprocess.run([sys.executable, 'generate_weak_data.py',
                            FEAT, str(N), fpath], check=False)

    rows = []
    t1 = None
    for P in [1, 2, 4, 8]:
        fpath = weak_files[P]
        if not os.path.exists(fpath):
            print(f"[skip] {fpath} not found.")
            continue
        print(f"\n--- P={P}  N={1236*P} ---")
        stdout, wall = run(mpirun_cmd(P, fpath))
        d = parse_output(stdout)
        if P == 1 and 'avg_iter_s' in d:
            t1 = d['avg_iter_s']
        weak_eff = (t1 / d['avg_iter_s'] * 100) if t1 and 'avg_iter_s' in d else None
        row = {'experiment': 'weak_scaling', 'P': P,
               'N_docs': 1236*P, 'feat_file': fpath,
               **d,
               'weak_efficiency_pct': weak_eff,
               'wall_elapsed_s': wall}
        rows.append(row)
        print(f"  avg_iter={d.get('avg_iter_s','?'):.4f}s  "
              f"weak_eff={weak_eff:.1f}%" if weak_eff else "")
    return rows

def experiment_distribution(np_list=(4, 8)):
    print("\n" + "="*60)
    print("  EXPERIMENT 3: Distribution Strategy Comparison")
    print("="*60)
    dist_map = {0: 'block', 1: 'cyclic', 2: 'dynamic'}
    rows = []
    for P in np_list:
        for dist_id, dist_label in dist_map.items():
            print(f"\n--- P={P}  dist={dist_label} ---")
            stdout, wall = run(mpirun_cmd(P, FEAT, init=1, dist=dist_id, comm=0))
            d = parse_output(stdout)
            rows.append({'experiment': 'dist_compare', 'P': P,
                         'dist': dist_label, 'comm': 'baseline',
                         **d, 'wall_elapsed_s': wall})
            print(f"  total={d.get('total_s','?'):.4f}s  "
                  f"compute={d.get('compute_pct','?'):.1f}%  "
                  f"comm={d.get('comm_pct','?'):.1f}%")
    return rows

def experiment_comm_mode(np_list=(4,)):
    print("\n" + "="*60)
    print("  EXPERIMENT 4: Communication Mode Comparison")
    print("="*60)
    comm_map = {0: 'baseline', 1: 'nonblock'}
    rows = []
    for P in np_list:
        for comm_id, comm_label in comm_map.items():
            print(f"\n--- P={P}  comm={comm_label} ---")
            stdout, wall = run(mpirun_cmd(P, FEAT, init=1, dist=0, comm=comm_id))
            d = parse_output(stdout)
            rows.append({'experiment': 'comm_compare', 'P': P,
                         'dist': 'block', 'comm': comm_label,
                         **d, 'wall_elapsed_s': wall})
            print(f"  total={d.get('total_s','?'):.4f}s")
    return rows

def experiment_vary_k(P=4, k_values=(5, 10, 20, 50)):
    print("\n" + "="*60)
    print("  EXPERIMENT 5: Vary Cluster Count k")
    print("="*60)
    rows = []
    header_h = fcm_h = open('fcm_mpi.h').read()

    for k in k_values:
        print(f"\n--- k={k}  P={P} ---")
        # Temporarily patch header
        patched = re.sub(r'#define N_CLUSTERS\s+\d+',
                         f'#define N_CLUSTERS   {k}', header_h)
        with open('fcm_mpi_tmp.h', 'w') as f:
            f.write(patched)

        # Build with patched header
        build = subprocess.run(
            ['mpicc', '-O2', '-std=c99', '-include', 'fcm_mpi_tmp.h',
             '-o', f'fcm_mpi_k{k}', 'fcm_mpi.c', 'main_mpi.c', '-lm'],
            capture_output=True, text=True
        )
        if build.returncode != 0:
            print(f"  [warn] Build failed for k={k}: {build.stderr[:200]}")
            continue

        stdout, wall = run([MPIRUN, '--oversubscribe', '-np', str(P),
                            f'./fcm_mpi_k{k}', FEAT, LABELS, '1', '0', '0'])
        d = parse_output(stdout)
        rows.append({'experiment': 'vary_k', 'k': k, 'P': P, **d,
                     'wall_elapsed_s': wall})
        print(f"  k={k}  avg_iter={d.get('avg_iter_s','?')}s  "
              f"iters={d.get('iterations','?')}")
        os.remove(f'fcm_mpi_k{k}')

    os.remove('fcm_mpi_tmp.h')
    return rows

def save_all(rows, path='scaling_results.csv'):
    if not rows:
        print("[warn] No results to save.")
        return
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[save] All results -> {path}")

def print_strong_scaling_table(rows):
    sr = [r for r in rows if r.get('experiment') == 'strong_scaling']
    if not sr:
        return
    print("\n── Strong Scaling Summary ──────────────────────────")
    print(f"  {'P':>3}  {'Avg/iter(s)':>12}  {'Speedup':>8}  {'Efficiency':>10}  {'Converged':>9}")
    print(f"  {'-'*3}  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*9}")
    for r in sr:
        print(f"  {r['P']:>3}  {r.get('avg_iter_s',0):>12.4f}  "
              f"{r.get('speedup',0):>8.2f}×  "
              f"{r.get('efficiency_pct',0):>9.1f}%  "
              f"{'Yes' if r.get('converged') else 'No':>9}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binary',  default='./fcm_mpi')
    ap.add_argument('--np_max',  type=int, default=8)
    ap.add_argument('--skip_weak',   action='store_true')
    ap.add_argument('--skip_vary_k', action='store_true')
    args = ap.parse_args()

    global BINARY
    BINARY = args.binary

    if not os.path.exists(BINARY):
        print(f"[error] Binary {BINARY} not found. Run 'make' first.")
        sys.exit(1)
    if not os.path.exists(FEAT):
        print(f"[error] {FEAT} not found. Copy Member 2 data here.")
        sys.exit(1)

    all_rows = []
    all_rows += experiment_strong_scaling(np_max=args.np_max)
    if not args.skip_weak:
        all_rows += experiment_weak_scaling()
    all_rows += experiment_distribution()
    all_rows += experiment_comm_mode()
    if not args.skip_vary_k:
        all_rows += experiment_vary_k()

    save_all(all_rows)
    print_strong_scaling_table(all_rows)
    print("\n✅ All experiments complete.")
    print("   Results in scaling_results.csv")
    print("   Run: python3 evaluate_metrics.py  to compute Silhouette/DB/ARI")

if __name__ == '__main__':
    main()
