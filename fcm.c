/* ============================================================
 * fcm.c  —  Fuzzy C-Means: full serial implementation
 *
 * Covers Member 1's Milestone 1 responsibilities:
 *   ✓ Serial FCM (membership update, centroid computation, convergence)
 *   ✓ Initialization: Random, K-Means++, Domain-guided
 *   ✓ M-step with weighted centroid update
 *   ✓ E-step with membership matrix update
 *   ✓ Frobenius-norm convergence detection
 *   ✓ CSV output for Member 3 (metrics/validation)
 *
 * NOTE ON MPI PARALLELISATION (Milestone 2 prep):
 *   The E-step (fcm_update_membership) is embarrassingly parallel —
 *   each rank will compute memberships for its local slice of data.
 *   The M-step (fcm_update_centroids) will use MPI_Allreduce to
 *   sum partial weighted vectors before dividing.
 *   Convergence will use MPI_Allreduce on the local Frobenius norms.
 *   This serial version is the direct baseline for that work.
 *
 * Member 1: Khansa Danish
 * IBA Karachi, Spring 2026
 * ============================================================ */

#include "fcm.h"

/* ════════════════════════════════════════════════════════════
 * LIFECYCLE
 * ════════════════════════════════════════════════════════════ */

/* Allocate and initialise an FCMModel.
   data[][] is allocated but left zeroed — caller fills it.    */
FCMModel *fcm_create(int n_points, int n_features, int n_clusters) {
    if (n_points <= 0 || n_features <= 0 || n_clusters <= 0) {
        fprintf(stderr, "[fcm] ERROR: invalid dimensions (%d, %d, %d)\n",
                n_points, n_features, n_clusters);
        return NULL;
    }
    FCMModel *m = (FCMModel *)malloc(sizeof(FCMModel));
    if (!m) { fprintf(stderr, "[fcm] ERROR: malloc failed\n"); exit(1); }

    m->n_points   = n_points;
    m->n_features = n_features;
    m->n_clusters = n_clusters;
    m->fuzziness  = FCM_M;
    m->iterations = 0;
    m->final_delta = 0.0;

    m->data      = alloc_matrix(n_points,   n_features);
    m->U         = alloc_matrix(n_points,   n_clusters);
    m->centroids = alloc_matrix(n_clusters, n_features);

    return m;
}

/* Free all heap memory owned by model */
void fcm_free(FCMModel *model) {
    if (!model) return;
    free_matrix(model->data,      model->n_points);
    free_matrix(model->U,         model->n_points);
    free_matrix(model->centroids, model->n_clusters);
    free(model);
}

/* ════════════════════════════════════════════════════════════
 * INITIALISATION STRATEGIES
 * ════════════════════════════════════════════════════════════ */

/* ── Strategy 1: Random ──────────────────────────────────────
 * Assign random membership values to every (point, cluster) pair,
 * then normalise each row so it sums to 1.
 * Simple, fast, but sensitive to random seed.                 */
void fcm_init_random(FCMModel *model) {
    srand((unsigned int)time(NULL));
    int N = model->n_points;
    int C = model->n_clusters;

    for (int i = 0; i < N; i++) {
        double row_sum = 0.0;
        for (int j = 0; j < C; j++) {
            model->U[i][j] = (double)rand() / RAND_MAX + 1e-9; /* avoid zero */
            row_sum += model->U[i][j];
        }
        for (int j = 0; j < C; j++)
            model->U[i][j] /= row_sum;
    }
    printf("[init] Random initialisation complete.\n");
}

/* ── Strategy 2: K-Means++ ───────────────────────────────────
 * Choose C cluster centres with distance-proportional probability.
 * This reduces bad initialisations and typically converges faster.
 *
 * Algorithm:
 *   1. Pick first centroid uniformly at random from data.
 *   2. For each subsequent centroid, compute D(x)² = min distance²
 *      from x to any already-chosen centroid.
 *   3. Sample the next centroid with probability ∝ D(x)².
 *   4. Translate chosen centroids into a fuzzy U matrix by setting
 *      each point's membership highest for its nearest centroid.  */
void fcm_init_kmeanspp(FCMModel *model) {
    srand((unsigned int)time(NULL));
    int N = model->n_points;
    int F = model->n_features;
    int C = model->n_clusters;

    /* Store indices of chosen seed data points */
    int *seed_idx = (int *)malloc(C * sizeof(int));
    double *min_dist2 = (double *)malloc(N * sizeof(double));
    if (!seed_idx || !min_dist2) { fprintf(stderr, "[init] malloc failed\n"); exit(1); }

    /* Step 1: first centroid uniformly at random */
    seed_idx[0] = rand() % N;

    for (int k = 1; k < C; k++) {
        /* Recompute min distance² from each point to nearest chosen centroid */
        double total = 0.0;
        for (int i = 0; i < N; i++) {
            double best = DBL_MAX;
            for (int s = 0; s < k; s++) {
                double d = euclidean_distance(model->data[i],
                                              model->data[seed_idx[s]], F);
                if (d * d < best) best = d * d;
            }
            min_dist2[i] = best;
            total += best;
        }

        /* Step 3: sample proportional to D²  */
        double r = ((double)rand() / RAND_MAX) * total;
        double cumsum = 0.0;
        seed_idx[k] = N - 1;  /* fallback */
        for (int i = 0; i < N; i++) {
            cumsum += min_dist2[i];
            if (cumsum >= r) { seed_idx[k] = i; break; }
        }
    }

    /* Step 4: set centroids from seed data points */
    for (int k = 0; k < C; k++)
        memcpy(model->centroids[k], model->data[seed_idx[k]], F * sizeof(double));

    /* Initialise U: full membership to nearest centroid, tiny share to others */
    for (int i = 0; i < N; i++) {
        int best_k = 0;
        double best_d = DBL_MAX;
        for (int k = 0; k < C; k++) {
            double d = euclidean_distance(model->data[i], model->centroids[k], F);
            if (d < best_d) { best_d = d; best_k = k; }
        }
        double row_sum = 0.0;
        for (int j = 0; j < C; j++) {
            model->U[i][j] = (j == best_k) ? 0.7 : (0.3 / (C - 1));
            row_sum += model->U[i][j];
        }
        for (int j = 0; j < C; j++)
            model->U[i][j] /= row_sum;   /* normalise */
    }

    free(seed_idx);
    free(min_dist2);
    printf("[init] K-Means++ initialisation complete.\n");
}

/* ── Strategy 3: Domain-guided ──────────────────────────────
 * When clinical category labels are available (e.g., from ICD codes
 * or a small labeled subset), we can seed centroids directly from
 * those categories.  This gives medically meaningful starting points.
 *
 * Parameters:
 *   domain_labels — integer array of length N.  domain_labels[i]
 *                   is the cluster index (0..C-1) for document i.
 *                   Pass NULL to fall back to random init.
 *
 * In Milestone 2, Member 2 may supply these from ICD code groupings.  */
void fcm_init_domain(FCMModel *model, int *domain_labels) {
    if (!domain_labels) {
        printf("[init] Domain labels NULL — falling back to random init.\n");
        fcm_init_random(model);
        return;
    }

    int N = model->n_points;
    int F = model->n_features;
    int C = model->n_clusters;

    /* Compute centroid for each domain class as mean of its members */
    int *counts = (int *)calloc(C, sizeof(int));
    double **sums = alloc_matrix(C, F);

    for (int i = 0; i < N; i++) {
        int c = domain_labels[i];
        if (c < 0 || c >= C) {
            fprintf(stderr, "[init] WARNING: domain_labels[%d]=%d out of range, skipping\n", i, c);
            continue;
        }
        counts[c]++;
        for (int f = 0; f < F; f++)
            sums[c][f] += model->data[i][f];
    }

    for (int j = 0; j < C; j++) {
        if (counts[j] == 0) {
            /* No samples for this class — fall back to random data point */
            int idx = rand() % N;
            memcpy(model->centroids[j], model->data[idx], F * sizeof(double));
            printf("[init] WARNING: domain class %d has 0 samples — random fallback.\n", j);
        } else {
            for (int f = 0; f < F; f++)
                model->centroids[j][f] = sums[j][f] / counts[j];
        }
    }

    /* Initialise U with hard membership to domain class + small noise */
    for (int i = 0; i < N; i++) {
        int dominant = domain_labels[i];
        if (dominant < 0 || dominant >= C) dominant = 0;
        double row_sum = 0.0;
        for (int j = 0; j < C; j++) {
            model->U[i][j] = (j == dominant) ? 0.8 : (0.2 / (C - 1));
            row_sum += model->U[i][j];
        }
        for (int j = 0; j < C; j++)
            model->U[i][j] /= row_sum;
    }

    free(counts);
    free_matrix(sums, C);
    printf("[init] Domain-guided initialisation complete.\n");
}

/* ════════════════════════════════════════════════════════════
 * CORE ALGORITHM — E-STEP & M-STEP
 * ════════════════════════════════════════════════════════════ */

/* ── M-Step: update centroids from current membership matrix ─
 *
 *        Σ_i  u_ij^m · x_i
 *  c_j = ──────────────────
 *           Σ_i  u_ij^m
 *
 * MPI NOTE: In Milestone 2, each rank will compute a PARTIAL sum
 * (numerator and denominator) over its local slice of data, then
 * MPI_Allreduce will sum those partials to get the global centroid.  */
void fcm_update_centroids(FCMModel *model) {
    int N = model->n_points;
    int F = model->n_features;
    int C = model->n_clusters;
    double m = model->fuzziness;

    for (int j = 0; j < C; j++) {
        double weight_total = 0.0;
        memset(model->centroids[j], 0, F * sizeof(double));

        for (int i = 0; i < N; i++) {
            double u_m = pow(model->U[i][j], m);   /* u_ij^m */
            weight_total += u_m;
            for (int f = 0; f < F; f++)
                model->centroids[j][f] += u_m * model->data[i][f];
        }

        if (weight_total < FCM_MIN_DIST) weight_total = FCM_MIN_DIST;
        for (int f = 0; f < F; f++)
            model->centroids[j][f] /= weight_total;
    }
}

/* ── E-Step: update membership matrix from current centroids ─
 *
 *                          1
 *  u_ij = ──────────────────────────────────────
 *          Σ_k  ( d(x_i, c_j) / d(x_i, c_k) )^(2/(m-1))
 *
 * Special case: if point i exactly equals centroid j (distance = 0),
 * assign full membership to j and 0 to all others.
 *
 * MPI NOTE: In Milestone 2, this entire loop will be distributed —
 * each rank iterates only over its local i values independently.     */
void fcm_update_membership(FCMModel *model) {
    int N = model->n_points;
    int F = model->n_features;
    int C = model->n_clusters;
    double exp = 2.0 / (model->fuzziness - 1.0);

    /* Reusable distance buffer to avoid re-malloc inside loop */
    double *dist = (double *)malloc(C * sizeof(double));
    if (!dist) { fprintf(stderr, "[fcm] malloc failed in E-step\n"); exit(1); }

    for (int i = 0; i < N; i++) {
        /* Compute distance from point i to every centroid */
        int zero_cluster = -1;
        for (int j = 0; j < C; j++) {
            dist[j] = euclidean_distance(model->data[i], model->centroids[j], F);
            if (dist[j] < FCM_MIN_DIST) { zero_cluster = j; }
        }

        /* Special case: point coincides with a centroid */
        if (zero_cluster >= 0) {
            for (int j = 0; j < C; j++)
                model->U[i][j] = (j == zero_cluster) ? 1.0 : 0.0;
            continue;
        }

        /* Normal case: compute membership via FCM formula */
        for (int j = 0; j < C; j++) {
            double sum = 0.0;
            for (int k = 0; k < C; k++) {
                double ratio = dist[j] / dist[k];
                sum += pow(ratio, exp);
            }
            model->U[i][j] = 1.0 / sum;
        }
    }
    free(dist);
}

/* ── Frobenius norm of (U_new - U_old) ──────────────────────
 * Used as convergence criterion.
 *
 * MPI NOTE: In Milestone 2, each rank computes a partial sum²,
 * then MPI_Allreduce sums them, and the square root is taken once.  */
double fcm_frobenius_delta(double **U_old, double **U_new, int n, int c) {
    double norm = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < c; j++) {
            double diff = U_new[i][j] - U_old[i][j];
            norm += diff * diff;
        }
    return sqrt(norm);
}

/* ════════════════════════════════════════════════════════════
 * MAIN LOOP
 * ════════════════════════════════════════════════════════════ */

/* Run FCM to convergence (or MAX_ITER).
 *
 * Sequence:
 *   1. Initialise U via chosen strategy.
 *   2. Compute initial centroids from U  (M-step).
 *   3. Loop:
 *        a. Save U_old
 *        b. E-step: update U from centroids
 *        c. M-step: update centroids from U
 *        d. Check ||U - U_old||_F < ε
 */
void fcm_run(FCMModel *model, InitStrategy strategy, int *domain_labels) {
    int N = model->n_points;
    int C = model->n_clusters;

    printf("\n[fcm] Starting FCM  |  N=%d  F=%d  C=%d  m=%.1f  ε=%.0e\n",
           N, model->n_features, C, model->fuzziness, FCM_EPSILON);

    /* ── 1. Initialise ── */
    switch (strategy) {
        case INIT_KMEANSPP: fcm_init_kmeanspp(model);               break;
        case INIT_DOMAIN:   fcm_init_domain(model, domain_labels);  break;
        default:            fcm_init_random(model);                  break;
    }

    /* ── 2. Initial centroids ── */
    fcm_update_centroids(model);

    /* ── 3. Allocate U_old buffer ── */
    double **U_old = alloc_matrix(N, C);

    /* ── 4. Iterate ── */
    for (int iter = 1; iter <= FCM_MAX_ITER; iter++) {
        copy_matrix(U_old, model->U, N, C);     /* save current U  */
        fcm_update_membership(model);            /* E-step          */
        fcm_update_centroids(model);             /* M-step          */

        double delta = fcm_frobenius_delta(U_old, model->U, N, C);

        if (iter % 10 == 0 || iter <= 5)
            printf("[fcm] iter %4d  |  delta = %.8f\n", iter, delta);

        if (delta < FCM_EPSILON) {
            printf("[fcm] ✓ Converged at iteration %d  (delta=%.2e)\n", iter, delta);
            model->iterations   = iter;
            model->final_delta  = delta;
            free_matrix(U_old, N);
            return;
        }
    }

    printf("[fcm] ⚠ Reached MAX_ITER=%d without full convergence.\n", FCM_MAX_ITER);
    model->iterations  = FCM_MAX_ITER;
    model->final_delta = fcm_frobenius_delta(U_old, model->U, N, C);
    free_matrix(U_old, N);
}

/* ════════════════════════════════════════════════════════════
 * OUTPUT
 * ════════════════════════════════════════════════════════════ */

/* Print a human-readable summary — dominant cluster per point */
void fcm_print_summary(const FCMModel *model) {
    int N = model->n_points;
    int C = model->n_clusters;

    printf("\n[fcm] ── Summary ──────────────────────────────────────\n");
    printf("[fcm] Iterations    : %d\n", model->iterations);
    printf("[fcm] Final delta   : %.2e\n", model->final_delta);

    /* Cluster size (count of dominant membership) */
    int *counts = (int *)calloc(C, sizeof(int));
    for (int i = 0; i < N; i++) {
        int best = 0;
        for (int j = 1; j < C; j++)
            if (model->U[i][j] > model->U[i][best]) best = j;
        counts[best]++;
    }
    printf("[fcm] Hard assignment distribution:\n");
    for (int j = 0; j < C; j++)
        printf("       Cluster %d : %d documents\n", j, counts[j]);

    /* Print membership for first 5 points */
    printf("[fcm] Sample memberships (first 5 points):\n");
    int show = (N < 5) ? N : 5;
    for (int i = 0; i < show; i++) {
        printf("  point %3d : ", i);
        for (int j = 0; j < C; j++)
            printf("C%d=%.3f  ", j, model->U[i][j]);
        printf("\n");
    }
    free(counts);
}

/* Save full membership matrix to CSV (for Member 3's metrics) */
void fcm_save_membership(const FCMModel *model, const char *filepath) {
    FILE *fp = fopen(filepath, "w");
    if (!fp) { fprintf(stderr, "[fcm] Cannot open %s for writing\n", filepath); return; }

    for (int i = 0; i < model->n_points; i++) {
        for (int j = 0; j < model->n_clusters; j++) {
            fprintf(fp, "%.12f", model->U[i][j]);
            if (j < model->n_clusters - 1) fputc(',', fp);
        }
        fputc('\n', fp);
    }
    fclose(fp);
    printf("[fcm] Membership matrix saved → %s\n", filepath);
}

/* Save centroid matrix to CSV */
void fcm_save_centroids(const FCMModel *model, const char *filepath) {
    FILE *fp = fopen(filepath, "w");
    if (!fp) { fprintf(stderr, "[fcm] Cannot open %s for writing\n", filepath); return; }

    for (int j = 0; j < model->n_clusters; j++) {
        for (int f = 0; f < model->n_features; f++) {
            fprintf(fp, "%.12f", model->centroids[j][f]);
            if (f < model->n_features - 1) fputc(',', fp);
        }
        fputc('\n', fp);
    }
    fclose(fp);
    printf("[fcm] Centroids saved → %s\n", filepath);
}
