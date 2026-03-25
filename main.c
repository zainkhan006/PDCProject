/* ============================================================
 * main.c  —  Entry point: run & compare all three init strategies
 *
 * Usage:
 *   ./fcm <csv_path> N F C    — runs all strategies on real CSV data
 *
 * Example:
 *   ./fcm data/features.csv 500 100 4
 *
 * Member 1: Khansa Danish
 * IBA Karachi, Spring 2026
 * ============================================================ */

#include "fcm.h"

/* ── Dummy domain labels for 4 clinical groups ──────────────
 * Simulates what Member 2 would supply from ICD code groupings.
 * Group assignment cycles through 4 conditions.               */
static int *make_dummy_domain_labels(int N, int C) {
    int *labels = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
        labels[i] = i % C;
    return labels;
}

/* ── Run one strategy, save outputs, return iteration count ── */
static int run_strategy(int N, int F, int C,
                         InitStrategy strategy,
                         const char *strategy_name,
                         int *domain_labels,
                         const char *csv_path)
{
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║  Strategy: %-37s║\n", strategy_name);
    printf("╚══════════════════════════════════════════════════╝\n");

    FCMModel *model = fcm_create(N, F, C);

    if (fcm_load_csv(model, csv_path) != 0) {
        fprintf(stderr, "Failed to load CSV. Exiting.\n");
        fcm_free(model);
        return -1;
    }

    fcm_run(model, strategy, domain_labels);
    fcm_print_summary(model);

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

    if (argc != 5) {
        fprintf(stderr, "Usage: %s <csv_path> <N> <F> <C>\n", argv[0]);
        return 1;
    }

    char csv_path[256];
    strncpy(csv_path, argv[1], sizeof(csv_path) - 1);
    csv_path[sizeof(csv_path) - 1] = '\0';

    int N = atoi(argv[2]);
    int F = atoi(argv[3]);
    int C = atoi(argv[4]);

    if (N <= 0 || F <= 0 || C <= 0) {
        fprintf(stderr, "[main] ERROR: N, F, C must be positive integers.\n");
        return 1;
    }

    printf("[main] CSV mode: %s  N=%d  F=%d  C=%d\n", csv_path, N, F, C);

    /* ── Domain labels (simulated; replace with ICD-based labels later) ── */
    int *domain_labels = make_dummy_domain_labels(N, C);

    /* ── Run all three strategies ── */
    int iters_random   = run_strategy(N, F, C, INIT_RANDOM,   "Random",        NULL,          csv_path);
    int iters_kpp      = run_strategy(N, F, C, INIT_KMEANSPP, "K-Means++",     NULL,          csv_path);
    int iters_domain   = run_strategy(N, F, C, INIT_DOMAIN,   "Domain-Guided", domain_labels, csv_path);

    /* ── Comparison table ── */
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
    printf("[main] Pass these to Member 3 for silhouette/DB-index metrics.\n");

    free(domain_labels);
    return 0;
}
