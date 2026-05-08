/*
 * main_mpi.c  —  Entry point for parallel FCM
 * PDC Project 21, IBA Karachi Spring 2026
 *
 * Usage:
 *   mpirun --oversubscribe -np <P> ./fcm_mpi \
 *          <features.csv> <labels.csv> <init> <dist> <comm>
 *
 *   init:  0=random  1=kmeanspp  2=domain
 *   dist:  0=block   1=cyclic    2=dynamic
 *   comm:  0=baseline  1=nonblock
 */

#include "fcm_mpi.h"

static void print_banner(int rank)
{
    if (rank != 0) return;
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║  Parallel FCM — Clinical Notes (OpenMPI)        ║\n");
    printf("║  PDC Project 21 | IBA Spring 2026               ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
}

static const char *init_name(InitStrategy s)
{
    switch (s) {
        case INIT_KMEANSPP: return "K-Means++";
        case INIT_DOMAIN:   return "Domain-guided";
        default:            return "Random";
    }
}
static const char *dist_name(DistStrategy s)
{
    switch (s) {
        case DIST_CYCLIC:  return "Cyclic";
        case DIST_DYNAMIC: return "Dynamic (load-balanced)";
        default:           return "Block";
    }
}
static const char *comm_name(CommStrategy s)
{
    return (s == COMM_NONBLOCK) ? "Non-blocking" : "Baseline";
}

/* Build output filenames based on init strategy */
static void make_paths(InitStrategy init,
                       char *mem_path, char *cen_path, int sz)
{
    const char *tag = (init == INIT_KMEANSPP) ? "kmeanspp"
                    : (init == INIT_DOMAIN)   ? "domain"
                    :                            "random";
    snprintf(mem_path, sz, "membership_mpi_%s.csv", tag);
    snprintf(cen_path, sz, "centroids_mpi_%s.csv",  tag);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    print_banner(rank);

    /* ── Parse arguments ───────────────────────────────────────────── */
    const char *feat_path  = (argc > 1) ? argv[1] : "features.csv";
    const char *label_path = (argc > 2) ? argv[2] : "specialty_labels.csv";
    InitStrategy init = (argc > 3) ? (InitStrategy)atoi(argv[3]) : INIT_KMEANSPP;
    DistStrategy dist = (argc > 4) ? (DistStrategy)atoi(argv[4]) : DIST_BLOCK;
    CommStrategy comm = (argc > 5) ? (CommStrategy)atoi(argv[5]) : COMM_BASELINE;

    if (rank == 0) {
        printf("[main] Features : %s\n", feat_path);
        printf("[main] Labels   : %s\n", label_path);
        printf("[main] Strategy : %s\n", init_name(init));
        printf("[main] Dist mode: %s\n", dist_name(dist));
        printf("[main] Comm mode: %s\n", comm_name(comm));
        printf("[main] N=%d  F=%d  C=%d\n", TOTAL_DOCS, N_FEATURES, N_CLUSTERS);
    }

    /* ── Create model ─────────────────────────────────────────────── */
    FCMMpiModel *m = fcm_mpi_create(TOTAL_DOCS, N_FEATURES, N_CLUSTERS,
                                    dist, comm);

    /* ── Load and distribute data ─────────────────────────────────── */
    double t_io0 = MPI_Wtime();
    int *domain_labels = NULL;
    if (fcm_mpi_load_and_scatter(m, feat_path, label_path, &domain_labels) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double t_io1 = MPI_Wtime();
    if (rank == 0)
        printf("[main] Data load+scatter: %.3f s\n", t_io1 - t_io0);

    /* ── Run FCM ──────────────────────────────────────────────────── */
    double t_fcm0 = MPI_Wtime();
    fcm_mpi_run(m, init, domain_labels);
    double t_fcm1 = MPI_Wtime();

    if (rank == 0)
        printf("[main] FCM total time: %.4f s  (%d iters)\n",
               t_fcm1 - t_fcm0, m->iterations);

    /* ── Gather and save outputs ──────────────────────────────────── */
    char mem_path[128], cen_path[128];
    make_paths(init, mem_path, cen_path, sizeof(mem_path));
    fcm_mpi_gather_and_save(m, mem_path, cen_path, "feature_names.csv");

    /* ── Print summary and timing ─────────────────────────────────── */
    fcm_mpi_print_summary(m);
    fcm_mpi_print_timing(m);

    /* ── Cluster distribution (rank 0, from saved file) ──────────── */
    if (rank == 0) {
        /* Re-read membership to compute hard assignment counts */
        FILE *fp = fopen(mem_path, "r");
        if (fp) {
            int *counts = calloc(N_CLUSTERS, sizeof(int));
            char line[N_CLUSTERS * 25];
            int doc = 0;
            while (fgets(line, sizeof(line), fp) && doc < TOTAL_DOCS) {
                int best = 0; double best_v = -1.0;
                char *tok = strtok(line, ",\n");
                int j = 0;
                while (tok) {
                    double v = atof(tok);
                    if (v > best_v) { best_v = v; best = j; }
                    j++; tok = strtok(NULL, ",\n");
                }
                counts[best]++;
                doc++;
            }
            fclose(fp);
            printf("[fcm_mpi] Cluster distribution:\n");
            for (int j = 0; j < N_CLUSTERS; j++)
                printf("  Cluster %2d : %4d documents (%.1f%%)\n",
                       j, counts[j], 100.0 * counts[j] / TOTAL_DOCS);
            free(counts);
        }
    }

    free(domain_labels);
    fcm_mpi_free(m);
    MPI_Finalize();
    return 0;
}
