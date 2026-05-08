"""
evaluate_metrics.py  —  Clustering quality metrics for parallel FCM output
PDC Project 21 | Member 3 (Zain Khan) | IBA Spring 2026

Computes:
  - Silhouette coefficient  (sampled for speed)
  - Davies-Bouldin index
  - Adjusted Rand Index     (vs specialty labels)  ← WAS MISSING, NOW FIXED

Usage:
    python3 evaluate_metrics.py \
        --membership membership_mpi_kmeanspp.csv \
        --centroids  centroids_mpi_kmeanspp.csv  \
        --features   features.csv                \
        --labels     specialty_labels.csv         \
        --sample     2000
"""

import argparse
import numpy as np
import csv
from collections import defaultdict

# ─── Load helpers ─────────────────────────────────────────────────────────────

def load_csv_matrix(path, dtype=float):
    rows = []
    with open(path, newline='') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([dtype(x) for x in line.split(',')])
    return np.array(rows)


def load_labels(path):
    """Return list of string specialty labels, one per document."""
    labels = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # try common column names
            for col in ('medical_specialty', 'specialty', 'label'):
                if col in row:
                    labels.append(row[col].strip())
                    break
    return labels


def labels_to_int(str_labels):
    unique = sorted(set(str_labels))
    mapping = {v: i for i, v in enumerate(unique)}
    return np.array([mapping[l] for l in str_labels]), unique

# ─── Metrics ──────────────────────────────────────────────────────────────────

def hard_labels(U):
    """Argmax membership → hard cluster assignment."""
    return np.argmax(U, axis=1)


def silhouette(X, labels, sample=2000, seed=42):
    """
    Sampled Silhouette coefficient.
    Uses row-by-row distance computation to avoid N×N matrix.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = rng.choice(N, size=min(sample, N), replace=False)
    Xs = X[idx]
    Ls = labels[idx]

    scores = []
    unique_c = np.unique(Ls)
    if len(unique_c) < 2:
        print("[silhouette] Only 1 cluster — score undefined.")
        return 0.0

    for i in range(len(idx)):
        ci = Ls[i]
        xi = Xs[i]

        # a(i): mean distance to same-cluster points
        same = Xs[Ls == ci]
        if len(same) <= 1:
            a = 0.0
        else:
            dists = np.sqrt(((same - xi) ** 2).sum(axis=1))
            a = dists[dists > 0].mean() if dists[dists > 0].size > 0 else 0.0

        # b(i): min mean distance to any other cluster
        b = np.inf
        for cj in unique_c:
            if cj == ci:
                continue
            other = Xs[Ls == cj]
            if len(other) == 0:
                continue
            mean_d = np.sqrt(((other - xi) ** 2).sum(axis=1)).mean()
            if mean_d < b:
                b = mean_d

        if b == np.inf:
            scores.append(0.0)
        else:
            denom = max(a, b)
            scores.append((b - a) / denom if denom > 0 else 0.0)

    return float(np.mean(scores))


def davies_bouldin(X, centroids, labels):
    """Davies-Bouldin index (full dataset, cheap)."""
    C = centroids.shape[0]
    scatter = np.zeros(C)
    for j in range(C):
        members = X[labels == j]
        if len(members) == 0:
            scatter[j] = 0.0
        else:
            scatter[j] = np.sqrt(((members - centroids[j]) ** 2).sum(axis=1)).mean()

    db = 0.0
    for i in range(C):
        worst = -np.inf
        for j in range(C):
            if i == j:
                continue
            sep = np.sqrt(((centroids[i] - centroids[j]) ** 2).sum())
            if sep < 1e-12:
                r = (scatter[i] + scatter[j]) / 1e-12
            else:
                r = (scatter[i] + scatter[j]) / sep
            if r > worst:
                worst = r
        if worst > -np.inf:
            db += worst
    return db / C


def adjusted_rand_index(true_labels, pred_labels):
    """
    Adjusted Rand Index — compares predicted clusters with known specialty labels.
    This was listed in the project plan but never computed. Fixed here.
    """
    n = len(true_labels)
    assert len(pred_labels) == n

    # Contingency table
    classes  = np.unique(true_labels)
    clusters = np.unique(pred_labels)
    contingency = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    class_map   = {c: i for i, c in enumerate(classes)}
    cluster_map  = {c: i for i, c in enumerate(clusters)}
    for t, p in zip(true_labels, pred_labels):
        contingency[class_map[t], cluster_map[p]] += 1

    # Combinations
    def comb2(n): return n * (n - 1) // 2

    sum_comb_c = sum(comb2(contingency[i, j])
                     for i in range(len(classes))
                     for j in range(len(clusters)))
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    sum_comb_r = sum(comb2(x) for x in row_sums)
    sum_comb_k = sum(comb2(x) for x in col_sums)
    total_comb  = comb2(n)

    expected = sum_comb_r * sum_comb_k / total_comb if total_comb > 0 else 0
    max_idx   = (sum_comb_r + sum_comb_k) / 2

    if max_idx - expected < 1e-12:
        return 1.0 if sum_comb_c == expected else 0.0
    return (sum_comb_c - expected) / (max_idx - expected)


def purity(true_labels, pred_labels):
    """Weighted purity: fraction of docs in their dominant cluster per label."""
    label_cluster = defaultdict(lambda: defaultdict(int))
    label_total   = defaultdict(int)
    for t, p in zip(true_labels, pred_labels):
        label_cluster[t][p] += 1
        label_total[t] += 1

    macro_sum = 0.0
    weighted_sum = 0.0
    total = sum(label_total.values())
    for lbl, cluster_counts in label_cluster.items():
        dom_ratio = max(cluster_counts.values()) / label_total[lbl]
        macro_sum    += dom_ratio
        weighted_sum += dom_ratio * label_total[lbl]
    n_labels = len(label_cluster)
    macro   = macro_sum / n_labels if n_labels > 0 else 0.0
    weighted = weighted_sum / total if total > 0 else 0.0
    return macro, weighted


def membership_sanity(U):
    """Check rows sum to 1 and no NaN/Inf."""
    row_sums = U.sum(axis=1)
    max_dev  = np.abs(row_sums - 1.0).max()
    has_nan  = np.isnan(U).any()
    has_inf  = np.isinf(U).any()
    unique_rows = len(np.unique(U.round(6), axis=0))
    max_dom  = U.max(axis=1).max()
    print(f"[sanity] Membership rows     : {U.shape[0]}")
    print(f"[sanity] Unique rows (6dp)   : {unique_rows} ({100*unique_rows/U.shape[0]:.1f}%)")
    print(f"[sanity] Max deviation from 1: {max_dev:.2e}")
    print(f"[sanity] Max dominant value  : {max_dom:.4f}  (uniform baseline = {1/U.shape[1]:.4f})")
    print(f"[sanity] NaN present         : {has_nan}")
    print(f"[sanity] Inf present         : {has_inf}")
    n_active = len([j for j in range(U.shape[1]) if U.argmax(axis=1).tolist().count(j) > 0])
    print(f"[sanity] Active clusters     : {n_active}/{U.shape[1]}")
    return not has_nan and not has_inf and max_dev < 1e-6

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="FCM clustering quality metrics")
    ap.add_argument('--membership', default='membership_mpi_kmeanspp.csv')
    ap.add_argument('--centroids',  default='centroids_mpi_kmeanspp.csv')
    ap.add_argument('--features',   default='features.csv')
    ap.add_argument('--labels',     default='specialty_labels.csv')
    ap.add_argument('--sample',     type=int, default=2000,
                    help='Documents to sample for Silhouette (default 2000)')
    args = ap.parse_args()

    print("=" * 60)
    print("  FCM Clustering Quality Metrics")
    print("=" * 60)

    # Load data
    print(f"\n[load] Membership : {args.membership}")
    U = load_csv_matrix(args.membership)
    print(f"[load] Shape       : {U.shape}")

    print(f"[load] Centroids  : {args.centroids}")
    C_mat = load_csv_matrix(args.centroids)

    print(f"[load] Features   : {args.features}")
    X = load_csv_matrix(args.features)
    print(f"[load] Shape       : {X.shape}")

    # Derive hard labels
    pred = hard_labels(U)

    # Sanity checks
    print("\n─── Membership Sanity ───")
    membership_sanity(U)

    # Silhouette
    print(f"\n─── Silhouette Coefficient (sample={args.sample}) ───")
    sil = silhouette(X, pred, sample=args.sample)
    print(f"[metric] Silhouette : {sil:.4f}")
    if sil > 0.7:
        print("         → Strong cluster structure")
    elif sil > 0.5:
        print("         → Reasonable cluster structure")
    elif sil > 0.25:
        print("         → Weak cluster structure")
    else:
        print("         → No substantial geometric structure (expected for sparse TF-IDF)")

    # Davies-Bouldin
    print("\n─── Davies-Bouldin Index ───")
    db = davies_bouldin(X, C_mat, pred)
    print(f"[metric] Davies-Bouldin : {db:.4f}")
    print("         → Lower is better (0 = perfect)")

    # Cluster sizes
    print("\n─── Cluster Sizes ───")
    C = U.shape[1]
    for j in range(C):
        n = (pred == j).sum()
        print(f"  Cluster {j:2d}: {n:5d} ({100*n/len(pred):.1f}%)")

    # ARI and Purity (if labels available)
    try:
        str_labels = load_labels(args.labels)
        if len(str_labels) != len(pred):
            print(f"\n[warn] Label count ({len(str_labels)}) != doc count ({len(pred)}). "
                  f"Truncating to min.")
            n = min(len(str_labels), len(pred))
            str_labels = str_labels[:n]
            pred_trunc = pred[:n]
        else:
            pred_trunc = pred

        int_labels, unique_classes = labels_to_int(str_labels)
        print(f"\n─── Adjusted Rand Index (vs {len(unique_classes)} specialties) ───")
        ari = adjusted_rand_index(int_labels, pred_trunc)
        print(f"[metric] ARI : {ari:.4f}")
        print("         → 1.0=perfect, 0.0=random, <0=worse than random")
        print("         → NOTE: ARI with format-based labels (not disease labels) "
              "will be low; this is expected.")

        print("\n─── Purity vs Specialty Labels ───")
        macro, weighted = purity(str_labels, [str(p) for p in pred_trunc])
        print(f"[metric] Macro purity    : {macro:.4f}  (mean dominant ratio per specialty)")
        print(f"[metric] Weighted purity : {weighted:.4f}  (document-weighted)")
        C_random = 1.0 / C
        print(f"         → Random baseline: {C_random:.4f}")
        print(f"         → Improvement over random: {weighted/C_random:.1f}×")

        # Per-specialty breakdown (top 15)
        print("\n─── Per-Specialty Dominant Ratio (top 15 by doc count) ───")
        from collections import Counter
        label_counter = Counter(str_labels)
        top_specs = [sp for sp, _ in label_counter.most_common(15)]
        label_cluster_map = defaultdict(lambda: defaultdict(int))
        for lbl, p in zip(str_labels, [str(x) for x in pred_trunc]):
            label_cluster_map[lbl][p] += 1
        print(f"  {'Specialty':<40} {'Docs':>5} {'Dom Cluster':>11} {'Dom Ratio':>10}")
        print(f"  {'-'*40} {'-'*5} {'-'*11} {'-'*10}")
        for sp in top_specs:
            total_sp = label_counter[sp]
            cc = label_cluster_map[sp]
            dom_c = max(cc, key=cc.get)
            dom_r = cc[dom_c] / total_sp
            print(f"  {sp:<40} {total_sp:>5} {dom_c:>11} {dom_r:>10.3f}")

    except FileNotFoundError:
        print(f"\n[warn] Labels file '{args.labels}' not found — skipping ARI/purity.")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == '__main__':
    main()
