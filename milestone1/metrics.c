/* ============================================================
 * metrics.c  —  Clustering quality metrics
 *
 * Member 3's Milestone 1 responsibilities:
 *   ✓ Silhouette Coefficient  (Rousseeuw, 1987)
 *   ✓ Davies-Bouldin Index    (Davies & Bouldin, 1979)
 *   ✓ Hard assignment derivation from fuzzy membership
 *
 * Both metrics are "internal" validation — they evaluate
 * clustering quality using only the data and assignments,
 * with no ground-truth labels needed.
 *
 * References:
 *   [1] P. Rousseeuw, "Silhouettes: a Graphical Aid to the
 *       Interpretation and Validation of Cluster Analysis",
 *       J. Comput. Appl. Math., vol. 20, pp. 53-65, 1987.
 *
 *   [2] D. Davies & D. Bouldin, "A Cluster Separation Measure",
 *       IEEE Trans. PAMI, vol. 1, no. 2, pp. 224-227, 1979.
 *
 * Member 3 | IBA Karachi, Spring 2026
 * Project: Parallel Soft Clustering for Clinical Notes (OpenMPI)
 * ============================================================ */

#include "fcm.h"

/* ════════════════════════════════════════════════════════════
 * HARD ASSIGNMENT FROM FUZZY MEMBERSHIP
 * ════════════════════════════════════════════════════════════ */

/* ── Derive hard labels from membership matrix ────────────────
 *
 * For each point i, find: label_i = argmax_j U[i][j]
 *
 * FCM produces soft memberships (e.g., 0.6 to cluster A,
 * 0.3 to cluster B, 0.1 to cluster C).  Silhouette and
 * Davies-Bouldin are defined for hard (crisp) partitions,
 * so we assign each point to its highest-membership cluster.
 *
 * Returns a malloc'd int array of length N.  Caller must free.
 * ──────────────────────────────────────────────────────────── */
int *derive_hard_labels(const FCMModel *model) {
    int N = model->n_points;
    int C = model->n_clusters;

    int *labels = (int *)malloc(N * sizeof(int));
    if (!labels) {
        fprintf(stderr, "[metrics] malloc failed in derive_hard_labels\n");
        exit(1);
    }

    for (int i = 0; i < N; i++) {
        int best = 0;
        double best_val = model->U[i][0];
        for (int j = 1; j < C; j++) {
            if (model->U[i][j] > best_val) {
                best_val = model->U[i][j];
                best = j;
            }
        }
        labels[i] = best;
    }

    return labels;
}

/* ════════════════════════════════════════════════════════════
 * SILHOUETTE COEFFICIENT
 * ════════════════════════════════════════════════════════════
 *
 * For each point i assigned to cluster C_i:
 *
 *   a(i) = mean distance from i to all OTHER points in C_i
 *          (intra-cluster distance; measures cohesion)
 *
 *   b(i) = min over all clusters C_j (j ≠ label_i) of
 *          mean distance from i to all points in C_j
 *          (nearest-cluster distance; measures separation)
 *
 *                b(i) - a(i)
 *   s(i) = ─────────────────────
 *            max( a(i), b(i) )
 *
 * Special cases:
 *   - If cluster C_i has only 1 point: s(i) = 0
 *     (no intra-cluster distance to compute)
 *   - If a(i) = b(i) = 0: s(i) = 0
 *
 * Overall Silhouette = mean of s(i) over all N points.
 *
 * Range: [-1, +1]
 *   +1 = perfect separation
 *    0 = overlapping clusters
 *   -1 = wrong assignments
 *
 * Complexity: O(N²·F) — computes all pairwise distances.
 *             Fine for N=400 (Milestone 1); for larger N in
 *             Milestone 2, consider the simplified silhouette
 *             using centroid distances instead of all-pairs.
 *
 * MPI NOTE (Milestone 2): Pairwise distances can be
 * distributed by assigning each rank a slice of rows i.
 * Each rank needs access to all data (or at least to all
 * points in the relevant clusters) to compute b(i).
 * ════════════════════════════════════════════════════════════ */
double compute_silhouette(const FCMModel *model) {
    int N = model->n_points;
    int F = model->n_features;
    int C = model->n_clusters;

    /* Step 0: Derive hard labels from fuzzy membership */
    int *labels = derive_hard_labels(model);

    /* Step 1: Count cluster sizes */
    int *cluster_size = (int *)calloc(C, sizeof(int));
    if (!cluster_size) {
        fprintf(stderr, "[metrics] calloc failed\n");
        exit(1);
    }
    for (int i = 0; i < N; i++)
        cluster_size[labels[i]]++;

    /* Step 2: Compute silhouette for each point */
    double total_silhouette = 0.0;

    for (int i = 0; i < N; i++) {
        int my_cluster = labels[i];

        /* If this point is alone in its cluster, s(i) = 0 */
        if (cluster_size[my_cluster] <= 1) {
            /* s(i) = 0, contributes nothing */
            continue;
        }

        /* ── Compute a(i): mean distance to same-cluster points ── */
        double a_i = 0.0;
        int a_count = 0;

        /* ── Compute per-cluster mean distances for b(i) ── */
        /* sum_dist[j] = total distance from i to all points in cluster j */
        double *sum_dist = (double *)calloc(C, sizeof(double));
        int *count_dist = (int *)calloc(C, sizeof(int));
        if (!sum_dist || !count_dist) {
            fprintf(stderr, "[metrics] calloc failed\n");
            exit(1);
        }

        for (int k = 0; k < N; k++) {
            if (k == i) continue;  /* skip self */

            double dist = euclidean_distance(model->data[i], model->data[k], F);

            if (labels[k] == my_cluster) {
                /* Same cluster — contributes to a(i) */
                a_i += dist;
                a_count++;
            }

            /* Accumulate for all clusters (needed for b(i)) */
            sum_dist[labels[k]] += dist;
            count_dist[labels[k]]++;
        }

        /* Finalize a(i) */
        if (a_count > 0)
            a_i /= a_count;

        /* ── Compute b(i): min mean distance to other clusters ── */
        double b_i = DBL_MAX;
        for (int j = 0; j < C; j++) {
            if (j == my_cluster) continue;    /* skip own cluster */
            if (count_dist[j] == 0) continue; /* skip empty clusters */

            double mean_dist_j = sum_dist[j] / count_dist[j];
            if (mean_dist_j < b_i)
                b_i = mean_dist_j;
        }

        /* ── Compute s(i) ── */
        double s_i = 0.0;
        double denom = (a_i > b_i) ? a_i : b_i;  /* max(a_i, b_i) */

        if (denom > 1e-15)
            s_i = (b_i - a_i) / denom;
        /* else s_i = 0.0 (both distances are zero) */

        total_silhouette += s_i;

        free(sum_dist);
        free(count_dist);
    }

    /* Step 3: Average silhouette over all points */
    double avg_silhouette = total_silhouette / N;

    /* ── Print report ── */
    printf("\n[metrics] ── Silhouette Coefficient ──────────────\n");
    printf("[metrics] Average Silhouette Score : %.6f\n", avg_silhouette);
    printf("[metrics] Interpretation           : ");
    if (avg_silhouette > 0.70)
        printf("STRONG structure (> 0.70)\n");
    else if (avg_silhouette > 0.50)
        printf("REASONABLE structure (0.50 - 0.70)\n");
    else if (avg_silhouette > 0.25)
        printf("WEAK structure (0.25 - 0.50)\n");
    else
        printf("NO substantial structure (< 0.25)\n");

    printf("[metrics] Cluster sizes            : ");
    for (int j = 0; j < C; j++)
        printf("C%d=%d  ", j, cluster_size[j]);
    printf("\n");

    free(labels);
    free(cluster_size);

    return avg_silhouette;
}

/* ════════════════════════════════════════════════════════════
 * DAVIES-BOULDIN INDEX
 * ════════════════════════════════════════════════════════════
 *
 * For each cluster i:
 *
 *   S_i = (1/T_i) Σ_{x ∈ C_i}  dist(x, centroid_i)
 *         (average distance of points to their centroid;
 *          measures cluster "scatter" or "tightness")
 *
 * For each pair of clusters (i, j):
 *
 *   M_ij = dist(centroid_i, centroid_j)
 *          (distance between centroids; measures separation)
 *
 *   R_ij = (S_i + S_j) / M_ij
 *          (similarity ratio — high = bad, clusters overlap)
 *
 * For each cluster i:
 *
 *   D_i = max over all j ≠ i of R_ij
 *         (worst-case similarity for cluster i)
 *
 * Davies-Bouldin Index:
 *
 *   DB = (1/C) Σ_i D_i
 *
 * Range: [0, ∞)
 *   0    = perfect separation (unreachable in practice)
 *   lower = better clustering
 *
 * Complexity: O(N·C·F + C²·F)
 *   First term: computing S_i for all clusters
 *   Second term: computing all pairwise centroid distances
 *   Much cheaper than Silhouette (no N² term).
 *
 * MPI NOTE (Milestone 2): S_i computation is embarrassingly
 * parallel — each rank computes partial scatter for its local
 * data, then MPI_Allreduce sums them.  Centroid distances are
 * cheap (C is small) and can be computed on rank 0.
 * ════════════════════════════════════════════════════════════ */
double compute_davies_bouldin(const FCMModel *model) {
    int N = model->n_points;
    int F = model->n_features;
    int C = model->n_clusters;

    /* Step 0: Derive hard labels */
    int *labels = derive_hard_labels(model);

    /* Step 1: Count cluster sizes */
    int *cluster_size = (int *)calloc(C, sizeof(int));
    if (!cluster_size) {
        fprintf(stderr, "[metrics] calloc failed\n");
        exit(1);
    }
    for (int i = 0; i < N; i++)
        cluster_size[labels[i]]++;

    /* Step 2: Compute S_i (scatter) for each cluster
     * S_i = average distance of cluster i's points to centroid i
     * We use model->centroids which Member 1's code has already
     * computed during fcm_run().
     */
    double *scatter = (double *)calloc(C, sizeof(double));
    if (!scatter) {
        fprintf(stderr, "[metrics] calloc failed\n");
        exit(1);
    }

    for (int i = 0; i < N; i++) {
        int c = labels[i];
        scatter[c] += euclidean_distance(model->data[i], model->centroids[c], F);
    }

    for (int j = 0; j < C; j++) {
        if (cluster_size[j] > 0)
            scatter[j] /= cluster_size[j];
    }

    /* Step 3: Compute DB index */
    double db_sum = 0.0;
    int valid_clusters = 0;

    for (int i = 0; i < C; i++) {
        if (cluster_size[i] == 0) continue;  /* skip empty clusters */

        double max_R = 0.0;

        for (int j = 0; j < C; j++) {
            if (j == i) continue;
            if (cluster_size[j] == 0) continue;

            /* M_ij = distance between centroids i and j */
            double m_ij = euclidean_distance(model->centroids[i],
                                              model->centroids[j], F);

            /* Guard: if centroids coincide, R_ij would be infinite.
             * This shouldn't happen with proper initialization, but
             * guard against it anyway. */
            if (m_ij < 1e-15) {
                max_R = DBL_MAX;
                break;
            }

            /* R_ij = (S_i + S_j) / M_ij */
            double r_ij = (scatter[i] + scatter[j]) / m_ij;

            if (r_ij > max_R)
                max_R = r_ij;
        }

        db_sum += max_R;
        valid_clusters++;
    }

    double db_index = (valid_clusters > 0) ? db_sum / valid_clusters : 0.0;

    /* ── Print report ── */
    printf("\n[metrics] ── Davies-Bouldin Index ────────────────\n");
    printf("[metrics] DB Index                 : %.6f\n", db_index);
    printf("[metrics] Interpretation           : ");
    if (db_index < 0.5)
        printf("EXCELLENT separation (< 0.50)\n");
    else if (db_index < 1.0)
        printf("GOOD separation (0.50 - 1.00)\n");
    else if (db_index < 2.0)
        printf("MODERATE separation (1.00 - 2.00)\n");
    else
        printf("POOR separation (> 2.00)\n");

    printf("[metrics] Per-cluster scatter (S_i): ");
    for (int j = 0; j < C; j++) {
        if (cluster_size[j] > 0)
            printf("C%d=%.4f  ", j, scatter[j]);
        else
            printf("C%d=EMPTY  ", j);
    }
    printf("\n");

    free(labels);
    free(cluster_size);
    free(scatter);

    return db_index;
}

/* ════════════════════════════════════════════════════════════
 * COMBINED METRICS REPORT
 * ════════════════════════════════════════════════════════════ */

/* Convenience function that runs both metrics and prints a
 * combined comparison-ready summary.  Designed to be called
 * once after fcm_run() completes for each init strategy. */
void compute_all_metrics(const FCMModel *model, const char *strategy_name) {
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║  QUALITY METRICS: %-30s║\n", strategy_name);
    printf("╚══════════════════════════════════════════════════╝\n");

    double sil = compute_silhouette(model);
    double db  = compute_davies_bouldin(model);

    printf("\n[metrics] ── Combined Summary for %-16s ──\n", strategy_name);
    printf("[metrics] Silhouette  = %.6f  (higher is better, max=1)\n", sil);
    printf("[metrics] Davies-Bouldin = %.6f  (lower is better, min=0)\n", db);
    printf("[metrics] Iterations  = %d\n", model->iterations);
    printf("[metrics] Final delta = %.2e\n", model->final_delta);
}