/* ============================================================
 * main.c  —  Entry point: run & compare all three init strategies
 *
 * Usage:
 *   ./fcm                     — runs all 3 strategies on dummy data
 *   ./fcm <csv_path> N F C    — runs on real CSV from Member 2
 *
 * Additional flags (Member 3):
 *   --norm=none|l2|minmax|zscore   (default: l2)
 *   --no-idf                       (disable IDF weighting)
 *
 * Example:
 *   ./fcm data/features.csv 500 100 4 --norm=minmax
 *
 * Member 1: Khansa Danish  (original structure)
 * Member 3:                (normalization, metrics, imbalance integration)
 * IBA Karachi, Spring 2026
 * ============================================================ */

#include "fcm.h"

/* ─── Normalization strategy enum (Member 3) ──────────────── */
typedef enum {
    NORM_NONE    = 0,
    NORM_L2      = 1,
    NORM_MINMAX  = 2,
    NORM_ZSCORE  = 3
} NormStrategy;

/* ── Dummy domain labels for 4 clinical groups ──────────────
 * Simulates what Member 2 would supply from ICD code groupings.
 * Group assignment cycles through 4 conditions.               */
static int *make_dummy_domain_labels(int N, int C) {
    int *labels = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
        labels[i] = i % C;
    return labels;
}

/* ── Apply chosen normalization (Member 3) ─────────────────── */
static void apply_normalization(FCMModel *model, NormStrategy norm) {
    switch (norm) {
        case NORM_L2:      normalize_l2(model);      break;
        case NORM_MINMAX:  normalize_minmax(model);   break;
        case NORM_ZSCORE:  normalize_zscore(model);   break;
        default:
            printf("[normalize] No normalization applied.\n");
            break;
    }
}

/* ── Run one strategy, compute metrics, return iteration count ── */
static int run_strategy(int N, int F, int C,
                         InitStrategy strategy,
                         const char *strategy_name,
                         int *domain_labels,
                         int use_csv, const char *csv_path,
                         NormStrategy norm, int use_idf)
{
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║  Strategy: %-37s║\n", strategy_name);
    printf("╚══════════════════════════════════════════════════╝\n");

    FCMModel *model = fcm_create(N, F, C);

    if (use_csv) {
        if (fcm_load_csv(model, csv_path) != 0) {
            fprintf(stderr, "Failed to load CSV. Exiting.\n");
            fcm_free(model);
            return -1;
        }
    } else {
        fcm_generate_clinical_dummy(model, 42);
    }

    /* ── Member 3: Preprocessing ── */
    if (use_idf)
        weight_idf(model, 0.0);    /* IDF weighting before normalization */
    apply_normalization(model, norm);

    /* ── Member 1: Run FCM ── */
    fcm_run(model, strategy, domain_labels);
    fcm_print_summary(model);

    /* ── Member 3: Quality Metrics ── */
    compute_all_metrics(model, strategy_name);

    /* Save outputs with strategy name in filename */
    char mem_path[128], cen_path[128];
    snprintf(mem_path, sizeof(mem_path), "membership_%s.csv",
             strategy == INIT_RANDOM ? "random" :
             strategy == INIT_KMEANSPP ? "kmeanspp" : "domain");
    snprintf(cen_path, sizeof(cen_path), "centroids_%s.csv",
             strategy == INIT_RANDOM ? "random" :
             strategy == INIT_KMEANSPP ? "kmeanspp" : "domain");

    fcm_save_membership(model, mem_path);
    fcm_save_centroids(model, cen_path);

    int iters = model->iterations;
    double delta = model->final_delta;
    fcm_free(model);

    printf("[main] %s → converged in %d iters, final delta=%.2e\n",
           strategy_name, iters, delta);
    return iters;
}

/* ══════════════════════════════════════════════════════════ */

int main(int argc, char *argv[]) {

    /* ── Configuration ── */
    int N = 400;    /* documents  — replace with Member 2's actual count */
    int F = 50;     /* features   — replace with Member 2's vector dimension */
    int C = 4;      /* clusters   — 4 clinical groups (cardio/resp/metabolic/neuro) */
    int use_csv  = 0;
    char csv_path[256] = "";

    /* ── Member 3: Preprocessing config ── */
    NormStrategy norm = NORM_L2;   /* default: L2 row normalization */
    int use_idf = 1;               /* default: apply IDF weighting  */

    /* ── Parse optional CLI arguments ── */
    if (argc >= 5 && argv[1][0] != '-') {
        strncpy(csv_path, argv[1], sizeof(csv_path) - 1);
        N = atoi(argv[2]);
        F = atoi(argv[3]);
        C = atoi(argv[4]);
        use_csv = 1;
        printf("[main] CSV mode: %s  N=%d  F=%d  C=%d\n", csv_path, N, F, C);
    } else {
        printf("[main] Dummy data mode: N=%d  F=%d  C=%d\n", N, F, C);
        printf("[main] To use real data: ./fcm <csv_path> <N> <F> <C>\n");
    }

    /* Parse optional flags: --norm=none|l2|minmax|zscore  --no-idf */
    for (int a = 1; a < argc; a++) {
        if (strncmp(argv[a], "--norm=", 7) == 0) {
            const char *val = argv[a] + 7;
            if (strcmp(val, "none") == 0)        norm = NORM_NONE;
            else if (strcmp(val, "l2") == 0)     norm = NORM_L2;
            else if (strcmp(val, "minmax") == 0)  norm = NORM_MINMAX;
            else if (strcmp(val, "zscore") == 0)  norm = NORM_ZSCORE;
            else fprintf(stderr, "[main] Unknown norm '%s', using L2\n", val);
        }
        if (strcmp(argv[a], "--no-idf") == 0)
            use_idf = 0;
    }

    printf("[main] Normalization: %s | IDF weighting: %s\n",
           norm == NORM_NONE ? "none" :
           norm == NORM_L2 ? "L2" :
           norm == NORM_MINMAX ? "min-max" : "z-score",
           use_idf ? "ON" : "OFF");

    /* ── Member 3: Feature imbalance analysis (run once on raw data) ── */
    {
        printf("\n[main] Running feature imbalance analysis on raw data...\n");
        FCMModel *analysis_model = fcm_create(N, F, C);
        if (use_csv) {
            fcm_load_csv(analysis_model, csv_path);
        } else {
            fcm_generate_clinical_dummy(analysis_model, 42);
        }
        analyze_feature_imbalance(analysis_model, 1e-4);
        fcm_free(analysis_model);
    }

    /* ── Domain labels (simulated; replace with ICD-based labels later) ── */
    int *domain_labels = make_dummy_domain_labels(N, C);

    /* ── Run all three strategies ── */
    int iters_random   = run_strategy(N, F, C, INIT_RANDOM,   "Random",
                                       NULL,          use_csv, csv_path, norm, use_idf);
    int iters_kpp      = run_strategy(N, F, C, INIT_KMEANSPP, "K-Means++",
                                       NULL,          use_csv, csv_path, norm, use_idf);
    int iters_domain   = run_strategy(N, F, C, INIT_DOMAIN,   "Domain-Guided",
                                       domain_labels, use_csv, csv_path, norm, use_idf);

    /* ── Final comparison table (Member 1 + Member 3) ── */
    printf("\n╔═══════════════════════════════════════════════════╗\n");
    printf("║          INITIALISATION STRATEGY COMPARISON       ║\n");
    printf("╠═══════════════╦═══════════════════════════════════╣\n");
    printf("║ Strategy      ║  Iterations to converge           ║\n");
    printf("╠═══════════════╬═══════════════════════════════════╣\n");
    printf("║ Random        ║  %-33d ║\n", iters_random);
    printf("║ K-Means++     ║  %-33d ║\n", iters_kpp);
    printf("║ Domain-Guided ║  %-33d ║\n", iters_domain);
    printf("╚═══════════════╩═══════════════════════════════════╝\n");
    printf("\n[main] Output files: membership_*.csv, centroids_*.csv\n");
    printf("[main] Quality metrics printed above for each strategy.\n");

    free(domain_labels);
    return 0;
}