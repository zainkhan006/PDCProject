/* ============================================================
 * normalize.c  —  Feature normalization, scaling & imbalance handling
 *
 * Member 3's Milestone 1 responsibilities:
 *   ✓ L2 row normalization
 *   ✓ Min-max feature scaling
 *   ✓ Z-score standardization
 *   ✓ IDF-based feature weighting (class imbalance)
 *   ✓ Feature variance analysis & imbalance reporting
 *
 * All functions operate on model->data[N][F] in-place.
 * Call BEFORE fcm_run() — normalization changes the feature
 * space, so it must happen before FCM computes distances.
 *
 * Member 3 | IBA Karachi, Spring 2026
 * Project: Parallel Soft Clustering for Clinical Notes (OpenMPI)
 * ============================================================ */

#include "fcm.h"

/* ════════════════════════════════════════════════════════════
 * NORMALIZATION & SCALING
 * ════════════════════════════════════════════════════════════ */

/* ── L2 Row Normalization ─────────────────────────────────────
 *
 * For each document (row) i:
 *   norm_i = sqrt( Σ_f  x_if² )
 *   x_if   = x_if / norm_i        (for all features f)
 *
 * This makes every document vector unit-length, removing the
 * effect of document length on Euclidean distance.  Standard
 * practice for TF-IDF vectors in information retrieval.
 *
 * Edge case: if a row is all zeros (norm = 0), we skip it
 * to avoid division by zero — the document has no features.
 *
 * MPI NOTE (Milestone 2): Each rank normalizes its local
 * slice independently — no communication needed.
 * ──────────────────────────────────────────────────────────── */
void normalize_l2(FCMModel *model) {
    int N = model->n_points;
    int F = model->n_features;
    int skipped = 0;

    for (int i = 0; i < N; i++) {
        /* Compute L2 norm of row i */
        double norm = 0.0;
        for (int f = 0; f < F; f++)
            norm += model->data[i][f] * model->data[i][f];
        norm = sqrt(norm);

        /* Guard: skip zero-vector documents */
        if (norm < 1e-15) {
            skipped++;
            continue;
        }

        /* Divide each element by the norm */
        for (int f = 0; f < F; f++)
            model->data[i][f] /= norm;
    }

    printf("[normalize] L2 row normalization complete. ");
    printf("(%d docs normalized, %d zero-vector docs skipped)\n", N - skipped, skipped);
}

/* ── Min-Max Feature Scaling ──────────────────────────────────
 *
 * For each feature (column) f:
 *   min_f = min over all docs i of x_if
 *   max_f = max over all docs i of x_if
 *   x_if  = (x_if - min_f) / (max_f - min_f)
 *
 * Maps every feature to the [0, 1] range.  This ensures no
 * single feature dominates the Euclidean distance by having
 * a larger numeric range than others.
 *
 * Edge case: if min_f == max_f (constant feature), the
 * denominator is zero.  We set all values to 0.0 — a
 * constant feature carries no discriminative information.
 *
 * MPI NOTE (Milestone 2): Requires MPI_Allreduce to compute
 * global min/max across ranks before local scaling.
 * ──────────────────────────────────────────────────────────── */
void normalize_minmax(FCMModel *model) {
    int N = model->n_points;
    int F = model->n_features;
    int constant_features = 0;

    for (int f = 0; f < F; f++) {
        /* Find min and max for this feature across all documents */
        double min_val = model->data[0][f];
        double max_val = model->data[0][f];

        for (int i = 1; i < N; i++) {
            if (model->data[i][f] < min_val) min_val = model->data[i][f];
            if (model->data[i][f] > max_val) max_val = model->data[i][f];
        }

        double range = max_val - min_val;

        if (range < 1e-15) {
            /* Constant feature — set to zero */
            for (int i = 0; i < N; i++)
                model->data[i][f] = 0.0;
            constant_features++;
        } else {
            /* Scale to [0, 1] */
            for (int i = 0; i < N; i++)
                model->data[i][f] = (model->data[i][f] - min_val) / range;
        }
    }

    printf("[normalize] Min-max scaling complete. ");
    printf("(%d features scaled, %d constant features zeroed)\n",
           F - constant_features, constant_features);
}

/* ── Z-Score Standardization ──────────────────────────────────
 *
 * For each feature (column) f:
 *   mean_f = (1/N) Σ_i  x_if
 *   std_f  = sqrt( (1/N) Σ_i  (x_if - mean_f)² )
 *   x_if   = (x_if - mean_f) / std_f
 *
 * Gives each feature mean=0 and standard deviation=1.
 * This is the strongest form of normalization and is
 * preferred when features have very different distributions.
 *
 * Edge case: if std_f == 0 (constant feature), set to 0.0.
 *
 * NOTE: After z-score, values are NO LONGER in [0,1] — they
 * can be negative.  This is fine for FCM (Euclidean distance
 * works on any real values), but it means you should NOT
 * apply both min-max and z-score — pick one.
 *
 * MPI NOTE (Milestone 2): Requires two MPI_Allreduce passes:
 * one for partial sums (to compute global mean), one for
 * partial squared-deviations (to compute global std).
 * ──────────────────────────────────────────────────────────── */
void normalize_zscore(FCMModel *model) {
    int N = model->n_points;
    int F = model->n_features;
    int constant_features = 0;

    for (int f = 0; f < F; f++) {
        /* Pass 1: compute mean */
        double sum = 0.0;
        for (int i = 0; i < N; i++)
            sum += model->data[i][f];
        double mean = sum / N;

        /* Pass 2: compute standard deviation */
        double sq_sum = 0.0;
        for (int i = 0; i < N; i++) {
            double diff = model->data[i][f] - mean;
            sq_sum += diff * diff;
        }
        double std = sqrt(sq_sum / N);

        if (std < 1e-15) {
            /* Constant feature — set to zero */
            for (int i = 0; i < N; i++)
                model->data[i][f] = 0.0;
            constant_features++;
        } else {
            /* Standardize */
            for (int i = 0; i < N; i++)
                model->data[i][f] = (model->data[i][f] - mean) / std;
        }
    }

    printf("[normalize] Z-score standardization complete. ");
    printf("(%d features standardized, %d constant features zeroed)\n",
           F - constant_features, constant_features);
}

/* ════════════════════════════════════════════════════════════
 * CLASS IMBALANCE HANDLING
 * ════════════════════════════════════════════════════════════ */

/* ── IDF Feature Weighting ────────────────────────────────────
 *
 * For each feature f:
 *   df_f   = number of documents where x_if > threshold
 *   idf_f  = log( N / (1 + df_f) )
 *   x_if   = x_if * idf_f           (for all documents i)
 *
 * This upweights rare clinical concepts (e.g., an uncommon
 * diagnosis that appears in few notes) and downweights
 * ubiquitous terms (e.g., "patient", "history").
 *
 * The +1 in the denominator prevents division by zero for
 * features that never appear (df = 0).
 *
 * Parameters:
 *   threshold — minimum value to consider a feature "present"
 *               in a document.  For TF-IDF data, 0.0 works
 *               (any nonzero weight counts).  For embeddings,
 *               a small positive threshold may be better.
 *
 * MPI NOTE (Milestone 2): Requires MPI_Allreduce to sum
 * local document-frequency counts into global df values.
 * ──────────────────────────────────────────────────────────── */
void weight_idf(FCMModel *model, double threshold) {
    int N = model->n_points;
    int F = model->n_features;

    int rare_count = 0;     /* features appearing in < 5% of docs */
    int common_count = 0;   /* features appearing in > 80% of docs */

    for (int f = 0; f < F; f++) {
        /* Count documents where feature f is present */
        int df = 0;
        for (int i = 0; i < N; i++) {
            if (model->data[i][f] > threshold)
                df++;
        }

        /* Compute IDF weight: log(N / (1 + df)) */
        double idf = log((double)N / (1.0 + df));

        /* Apply weight to all documents */
        for (int i = 0; i < N; i++)
            model->data[i][f] *= idf;

        /* Track imbalance statistics */
        double doc_frac = (double)df / N;
        if (doc_frac < 0.05) rare_count++;
        if (doc_frac > 0.80) common_count++;
    }

    printf("[imbalance] IDF weighting applied (threshold=%.4f).\n", threshold);
    printf("[imbalance] Rare features (df < 5%%): %d / %d\n", rare_count, F);
    printf("[imbalance] Common features (df > 80%%): %d / %d\n", common_count, F);
}

/* ── Feature Variance Analysis & Imbalance Report ─────────────
 *
 * Computes per-feature statistics and prints a diagnostic
 * report about the feature distribution.  Does NOT modify
 * data — this is a read-only analysis function.
 *
 * For each feature f:
 *   variance_f = (1/N) Σ_i (x_if - mean_f)²
 *   df_f       = count of documents where x_if > 0
 *
 * Reports:
 *   - Total / active / dead features
 *   - Min / max / mean variance
 *   - Number of near-zero-variance features
 *   - Number of rare features (low document frequency)
 *   - Top 5 highest-variance and lowest-variance features
 *
 * Parameters:
 *   var_threshold — features with variance below this are
 *                   flagged as "near-zero variance"
 * ──────────────────────────────────────────────────────────── */
void analyze_feature_imbalance(const FCMModel *model, double var_threshold) {
    int N = model->n_points;
    int F = model->n_features;

    /* Allocate arrays for per-feature statistics */
    double *variances = (double *)malloc(F * sizeof(double));
    int    *doc_freqs = (int *)malloc(F * sizeof(int));
    if (!variances || !doc_freqs) {
        fprintf(stderr, "[imbalance] malloc failed\n");
        return;
    }

    int dead_features = 0;      /* variance exactly 0 */
    int low_var_features = 0;   /* variance < threshold */
    int rare_features = 0;      /* df < 5% of N */

    double min_var = DBL_MAX, max_var = 0.0, sum_var = 0.0;

    for (int f = 0; f < F; f++) {
        /* Compute mean */
        double sum = 0.0;
        for (int i = 0; i < N; i++)
            sum += model->data[i][f];
        double mean = sum / N;

        /* Compute variance and document frequency */
        double sq_sum = 0.0;
        int df = 0;
        for (int i = 0; i < N; i++) {
            double diff = model->data[i][f] - mean;
            sq_sum += diff * diff;
            if (model->data[i][f] > 1e-10) df++;
        }

        double var = sq_sum / N;
        variances[f] = var;
        doc_freqs[f] = df;
        sum_var += var;

        if (var < 1e-15) dead_features++;
        if (var < var_threshold) low_var_features++;
        if ((double)df / N < 0.05) rare_features++;
        if (var < min_var) min_var = var;
        if (var > max_var) max_var = var;
    }

    /* ── Print report ── */
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║       FEATURE IMBALANCE ANALYSIS REPORT         ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  Total features          : %-21d ║\n", F);
    printf("║  Dead features (var≈0)   : %-21d ║\n", dead_features);
    printf("║  Low-variance (< %.1e): %-21d ║\n", var_threshold, low_var_features);
    printf("║  Active features         : %-21d ║\n", F - dead_features);
    printf("║  Rare features (df < 5%%) : %-21d ║\n", rare_features);
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  Variance: min=%.6f  max=%.6f           ║\n", min_var, max_var);
    printf("║  Variance: mean=%.6f                       ║\n", sum_var / F);
    printf("╠══════════════════════════════════════════════════╣\n");

    /* ── Top 5 highest-variance features ── */
    printf("║  Top 5 HIGHEST-variance features:               ║\n");
    for (int rank = 0; rank < 5 && rank < F; rank++) {
        int best_f = -1;
        double best_v = -1.0;
        for (int f = 0; f < F; f++) {
            if (variances[f] > best_v && variances[f] >= 0.0) {
                best_v = variances[f];
                best_f = f;
            }
        }
        if (best_f >= 0) {
            printf("║    Feature %3d : var=%.6f  df=%3d (%.1f%%)    ║\n",
                   best_f, variances[best_f], doc_freqs[best_f],
                   100.0 * doc_freqs[best_f] / N);
            variances[best_f] = -1.0;  /* mark as printed */
        }
    }

    /* Restore negated values for lowest-variance search */
    /* (Re-compute to avoid issues with the negation trick) */
    for (int f = 0; f < F; f++) {
        if (variances[f] < 0.0) {
            /* Recompute this feature's variance */
            double s = 0.0, sq = 0.0;
            for (int i = 0; i < N; i++) {
                s += model->data[i][f];
            }
            double m = s / N;
            for (int i = 0; i < N; i++) {
                double d = model->data[i][f] - m;
                sq += d * d;
            }
            variances[f] = sq / N;
        }
    }

    /* ── Top 5 lowest-variance features (non-dead) ── */
    printf("║  Top 5 LOWEST-variance features (non-dead):     ║\n");
    for (int rank = 0; rank < 5 && rank < F; rank++) {
        int best_f = -1;
        double best_v = DBL_MAX;
        for (int f = 0; f < F; f++) {
            if (variances[f] >= 1e-15 && variances[f] < best_v) {
                best_v = variances[f];
                best_f = f;
            }
        }
        if (best_f >= 0) {
            printf("║    Feature %3d : var=%.6f  df=%3d (%.1f%%)    ║\n",
                   best_f, variances[best_f], doc_freqs[best_f],
                   100.0 * doc_freqs[best_f] / N);
            variances[best_f] = DBL_MAX;  /* mark as printed */
        }
    }

    printf("╚══════════════════════════════════════════════════╝\n");

    free(variances);
    free(doc_freqs);
}