/* ============================================================
 * main_mpi.c  —  Entry point for parallel FCM
 *
 * Usage:
 *   mpirun -np <P> ./fcm_mpi <features.csv> <labels.csv> <strategy> <distribution> <comm_mode>
 *
 *   strategy:     0 = Random, 1 = K-Means++, 2 = Domain-guided
 *   distribution: 0 = Block, 1 = Cyclic, 2 = Dynamic (load-balanced)
 *   comm_mode:    0 = Baseline, 1 = Non-blocking optimized (experimental)
 *
 * Example (4 processes, K-Means++, dynamic):
 *   mpirun -np 4 ./fcm_mpi features.csv specialty_labels.csv 1 2
 *
 * Example (4 processes, K-Means++, cyclic):
 *   mpirun -np 4 ./fcm_mpi features.csv specialty_labels.csv 1 1
 *
 * Default (no args): runs on Member 2's files with K-Means++
 *
 * Member 1: Khansa Danish | IBA Karachi, Spring 2026
 * ============================================================ */

#include "fcm_mpi.h"

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* ── Parse arguments ── */
    char features_path[256] = "features.csv";
    char labels_path[256]   = "specialty_labels.csv";
    InitStrategy strategy   = INIT_KMEANSPP;   /* default: best strategy */
    DistStrategy dist_mode  = DIST_BLOCK;      /* default baseline mode  */
    CommStrategy comm_mode  = COMM_BASELINE;   /* default report-safe path */

    if (argc >= 3) {
        strncpy(features_path, argv[1], 255);
        strncpy(labels_path,   argv[2], 255);
    }
    if (argc >= 4) {
        int s = atoi(argv[3]);
        strategy = (s == 2) ? INIT_DOMAIN :
                   (s == 0) ? INIT_RANDOM : INIT_KMEANSPP;
    }
    if (argc >= 5) {
        int d = atoi(argv[4]);
        dist_mode = (d == 2) ? DIST_DYNAMIC :
                    (d == 1) ? DIST_CYCLIC : DIST_BLOCK;
    }
    if (argc >= 6) {
        int c = atoi(argv[5]);
        comm_mode = (c == 1) ? COMM_NONBLOCK_OPT : COMM_BASELINE;
    }

    if (rank == 0) {
        printf("╔══════════════════════════════════════════════════╗\n");
        printf("║  Parallel FCM — Clinical Notes (OpenMPI)        ║\n");
        printf("║  Member 1: Khansa Danish | IBA Spring 2026      ║\n");
        printf("╚══════════════════════════════════════════════════╝\n");
        printf("[main] Features : %s\n", features_path);
        printf("[main] Labels   : %s\n", labels_path);
        printf("[main] Strategy : %s\n",
               strategy == INIT_RANDOM   ? "Random" :
               strategy == INIT_KMEANSPP ? "K-Means++" : "Domain-Guided");
         printf("[main] Dist mode: %s\n",
             dist_mode == DIST_BLOCK ? "Block" :
             dist_mode == DIST_CYCLIC ? "Cyclic" : "Dynamic (load-balanced)");
        printf("[main] Comm mode: %s\n",
               comm_mode == COMM_NONBLOCK_OPT
                   ? "Non-blocking optimized (experimental)"
                   : "Baseline");
        printf("[main] N=%d  F=%d  C=%d\n", TOTAL_DOCS, N_FEATURES, N_CLUSTERS);
    }

    /* ── Create model ── */
    FCMMpiModel *model = fcm_mpi_create(TOTAL_DOCS, N_FEATURES, N_CLUSTERS,
                                        dist_mode, comm_mode);

    /* ── Load and scatter data ── */
    double t_load = MPI_Wtime();
    fcm_mpi_load_and_scatter(model, features_path);
    if (rank == 0)
        printf("[main] Data load+scatter: %.3f s\n", MPI_Wtime() - t_load);

    /* ── Load labels for domain-guided init (rank 0 only) ── */
    int *domain_labels = NULL;
    if (strategy == INIT_DOMAIN)
        domain_labels = fcm_mpi_load_labels(model, labels_path);

    /* ── Run FCM ── */
    double t_run = MPI_Wtime();
    fcm_mpi_run(model, strategy, domain_labels);
    double run_time = MPI_Wtime() - t_run;

    if (rank == 0)
        printf("[main] FCM total time: %.4f s  (%d iters)\n",
               run_time, model->iterations);

    /* ── Gather results and save ── */
    const char *strat_str = (strategy == INIT_RANDOM)   ? "random" :
                            (strategy == INIT_KMEANSPP) ? "kmeanspp" : "domain";
    char mem_path[128], cen_path[128];
    snprintf(mem_path, sizeof(mem_path), "membership_mpi_%s.csv",  strat_str);
    snprintf(cen_path, sizeof(cen_path), "centroids_mpi_%s.csv",   strat_str);

    fcm_mpi_gather_and_save(model, mem_path, cen_path, labels_path, "feature_names.csv");
    fcm_mpi_print_summary(model);
    fcm_mpi_print_timing(model);

    if (domain_labels) free(domain_labels);
    fcm_mpi_free(model);
    MPI_Finalize();
    return 0;
}
