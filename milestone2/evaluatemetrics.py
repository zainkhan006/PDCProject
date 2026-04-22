#!/usr/bin/env python3
"""
evaluate_metrics.py — Member 3: Clustering Quality Metrics
Reads MPI output (membership + centroids + features) and computes:
  - Silhouette Coefficient
  - Davies-Bouldin Index
  - Cluster size distribution
  - Membership confidence statistics

Usage:
  python3 evaluate_metrics.py membership_mpi_kmeanspp.csv centroids_mpi_kmeanspp.csv features.csv
  python3 evaluate_metrics.py  # uses default filenames
"""

import sys
import numpy as np
import time

def load_csv(path):
    """Load a CSV of floats, skipping any header rows that aren't numeric."""
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            try:
                row = [float(x) for x in parts]
                rows.append(row)
            except ValueError:
                continue  # skip header
    return np.array(rows)

def derive_hard_labels(U):
    """Convert soft membership matrix to hard cluster assignments."""
    return np.argmax(U, axis=1)

def compute_silhouette(data, labels):
    """
    Compute mean Silhouette Coefficient (memory-efficient).
    Samples 2000 points to avoid O(N^2) memory explosion, then computes
    distances row-by-row instead of building the full NxN matrix.
    """
    N = len(labels)
    unique_clusters = np.unique(labels)
    C = len(unique_clusters)
    
    if C <= 1:
        print("[metrics] Only 1 cluster found — silhouette undefined, returning 0.0")
        return 0.0
    
    # Sample for tractability
    SAMPLE = min(N, 2000)
    if N > SAMPLE:
        print(f"[metrics] Sampling {SAMPLE}/{N} points for silhouette")
        idx = np.random.RandomState(42).choice(N, SAMPLE, replace=False)
        sample_data = data[idx]
        sample_labels = labels[idx]
    else:
        sample_data = data
        sample_labels = labels
        idx = np.arange(N)
    
    NS = len(sample_data)
    unique_clusters = np.unique(sample_labels)
    
    # Precompute cluster masks
    cluster_masks = {}
    for c in unique_clusters:
        cluster_masks[c] = (sample_labels == c)
    
    print(f"[metrics] Computing silhouette for {NS} points (row-by-row)...")
    sil_values = np.zeros(NS)
    
    for i in range(NS):
        # Compute distances from point i to ALL sampled points
        diffs = sample_data - sample_data[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        
        ci = sample_labels[i]
        mask_same = cluster_masks[ci].copy()
        mask_same[i] = False  # exclude self
        n_same = np.sum(mask_same)
        
        if n_same == 0:
            sil_values[i] = 0.0
            continue
        
        a_i = np.mean(dists[mask_same])
        
        b_i = np.inf
        for c in unique_clusters:
            if c == ci:
                continue
            mask_other = cluster_masks[c]
            n_other = np.sum(mask_other)
            if n_other == 0:
                continue
            mean_dist = np.mean(dists[mask_other])
            if mean_dist < b_i:
                b_i = mean_dist
        
        denom = max(a_i, b_i)
        sil_values[i] = (b_i - a_i) / denom if denom > 0 else 0.0
    
    return np.mean(sil_values)

def compute_davies_bouldin(data, labels, centroids):
    """
    Compute Davies-Bouldin Index.
    For each cluster i:
      S_i = mean distance from points in i to centroid_i
    For each pair (i,j):
      R_ij = (S_i + S_j) / dist(centroid_i, centroid_j)
    DB = (1/C) * sum of max_j(R_ij) for each i
    """
    unique_clusters = np.unique(labels)
    C_actual = len(unique_clusters)
    
    if C_actual <= 1:
        print("[metrics] Only 1 cluster — DB undefined, returning inf")
        return float('inf')
    
    # Compute scatter for each cluster
    S = np.zeros(C_actual)
    for idx, c in enumerate(unique_clusters):
        mask = (labels == c)
        n_c = np.sum(mask)
        if n_c == 0:
            S[idx] = 0.0
            continue
        dists = np.sqrt(np.sum((data[mask] - centroids[c])**2, axis=1))
        S[idx] = np.mean(dists)
    
    # Compute centroid distances
    M = np.zeros((C_actual, C_actual))
    for i in range(C_actual):
        for j in range(i+1, C_actual):
            d = np.sqrt(np.sum((centroids[unique_clusters[i]] - centroids[unique_clusters[j]])**2))
            M[i,j] = d
            M[j,i] = d
    
    # Compute DB index
    db_sum = 0.0
    for i in range(C_actual):
        max_R = 0.0
        for j in range(C_actual):
            if i == j:
                continue
            if M[i,j] < 1e-20:
                max_R = 1e10  # clusters overlap
                continue
            R_ij = (S[i] + S[j]) / M[i,j]
            if R_ij > max_R:
                max_R = R_ij
        db_sum += max_R
    
    return db_sum / C_actual

def membership_stats(U):
    """Analyze membership matrix confidence."""
    max_memberships = np.max(U, axis=1)
    hard_labels = np.argmax(U, axis=1)
    C = U.shape[1]
    
    print(f"\n{'='*60}")
    print("MEMBERSHIP CONFIDENCE ANALYSIS")
    print(f"{'='*60}")
    print(f"  Mean max membership:   {np.mean(max_memberships):.6f}")
    print(f"  Median max membership: {np.median(max_memberships):.6f}")
    print(f"  Min max membership:    {np.min(max_memberships):.6f}")
    print(f"  Max max membership:    {np.max(max_memberships):.6f}")
    print(f"  Uniform threshold:     {1.0/C:.6f} (= 1/C)")
    
    # Points with confident assignment (max > 2/C)
    confident = np.sum(max_memberships > 2.0/C)
    print(f"  Confident (>{2.0/C:.3f}):   {confident}/{len(max_memberships)} "
          f"({100*confident/len(max_memberships):.1f}%)")
    
    print(f"\n  Cluster sizes (hard assignment):")
    for c in range(C):
        count = np.sum(hard_labels == c)
        print(f"    Cluster {c}: {count:5d} ({100*count/len(hard_labels):.1f}%)")
    
    active = sum(1 for c in range(C) if np.sum(hard_labels == c) > 0)
    print(f"  Active clusters: {active}/{C}")

def main():
    # Parse arguments
    if len(sys.argv) >= 4:
        mem_path = sys.argv[1]
        cen_path = sys.argv[2]
        feat_path = sys.argv[3]
    else:
        mem_path = "membership_mpi_kmeanspp.csv"
        cen_path = "centroids_mpi_kmeanspp.csv"
        feat_path = "features.csv"
    
    print(f"[metrics] Loading membership: {mem_path}")
    U = load_csv(mem_path)
    print(f"[metrics]   Shape: {U.shape}")
    
    print(f"[metrics] Loading centroids: {cen_path}")
    centroids = load_csv(cen_path)
    print(f"[metrics]   Shape: {centroids.shape}")
    
    print(f"[metrics] Loading features: {feat_path}")
    data = load_csv(feat_path)
    print(f"[metrics]   Shape: {data.shape}")
    
    # Validate dimensions
    N, C = U.shape
    N2, F = data.shape
    C2, F2 = centroids.shape
    
    assert N == N2, f"Mismatch: membership has {N} rows but features has {N2}"
    assert C == C2, f"Mismatch: membership has {C} clusters but centroids has {C2}"
    assert F == F2, f"Mismatch: features has {F} cols but centroids has {F2}"
    
    print(f"[metrics] N={N}, F={F}, C={C}")
    
    # Membership analysis
    membership_stats(U)
    
    # Derive hard labels
    labels = derive_hard_labels(U)
    
    # Compute metrics
    print(f"\n{'='*60}")
    print("CLUSTERING QUALITY METRICS")
    print(f"{'='*60}")
    
    print("[metrics] Computing Silhouette Coefficient...")
    t0 = time.time()
    sil = compute_silhouette(data, labels)
    sil_time = time.time() - t0
    print(f"  Silhouette Coefficient: {sil:.6f}  (computed in {sil_time:.2f}s)")
    
    print("[metrics] Computing Davies-Bouldin Index...")
    t0 = time.time()
    db = compute_davies_bouldin(data, labels, centroids)
    db_time = time.time() - t0
    print(f"  Davies-Bouldin Index:   {db:.6f}  (computed in {db_time:.2f}s)")
    
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    if sil > 0.7:
        print(f"  Silhouette {sil:.3f}: STRONG cluster structure")
    elif sil > 0.5:
        print(f"  Silhouette {sil:.3f}: REASONABLE cluster structure")
    elif sil > 0.25:
        print(f"  Silhouette {sil:.3f}: WEAK cluster structure")
    else:
        print(f"  Silhouette {sil:.3f}: NO substantial structure detected")
    
    if db < 0.5:
        print(f"  DB Index {db:.3f}: EXCELLENT cluster separation")
    elif db < 1.0:
        print(f"  DB Index {db:.3f}: GOOD cluster separation")
    elif db < 2.0:
        print(f"  DB Index {db:.3f}: MODERATE cluster separation")
    else:
        print(f"  DB Index {db:.3f}: POOR cluster separation")

if __name__ == "__main__":
    main()