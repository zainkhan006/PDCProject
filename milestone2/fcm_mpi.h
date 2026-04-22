#ifndef FCM_MPI_H
#define FCM_MPI_H

/* ============================================================
 * fcm_mpi.h  —  Parallel Fuzzy C-Means with OpenMPI
 * Milestone 2: MPI-parallel implementation
 * Member 1: Khansa Danish
 * Project 21: Parallel Soft Clustering for Clinical Notes
 * IBA Karachi, Spring 2026
 * ============================================================ */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <time.h>

/* ─── Hyperparameters ──────────────────────────────────────── */
#define FCM_M          1.1      /* Fuzziness exponent                          */
#define FCM_EPSILON    1e-5     /* Convergence threshold (Frobenius norm of DU)*/
#define FCM_MAX_ITER   150      /* Maximum iterations                          */
#define FCM_MIN_DIST   1e-10    /* Guard against zero distance                 */

/* ─── Dataset dimensions ───────────────────────────────────── */
#define TOTAL_DOCS     4943
#define N_FEATURES     500
#define N_CLUSTERS     10

/* ─── Initialisation strategies ───────────────────────────── */
typedef enum {
    INIT_RANDOM   = 0,
    INIT_KMEANSPP = 1,
    INIT_DOMAIN   = 2
} InitStrategy;

/* ─── Per-rank local model ─────────────────────────────────── */
typedef struct {
    int  N, F, C;
    int  rank, n_procs;
    int  local_n, local_start;
    double *local_data;
    double *local_U;
    double *local_U_old;
    double *all_U;       /* rank 0 only */
    double *centroids;   /* all ranks   */
    int    iterations;
    double final_delta;
    double iter_times[FCM_MAX_ITER];
} FCMMpiModel;

/* ─── Function prototypes ──────────────────────────────────── */
FCMMpiModel *fcm_mpi_create(int N, int F, int C);
void         fcm_mpi_free(FCMMpiModel *m);

int   fcm_mpi_load_and_scatter(FCMMpiModel *m,
                                const char *features_csv,
                                const char *labels_csv);
int  *fcm_mpi_load_labels(FCMMpiModel *m, const char *labels_csv);

void fcm_mpi_init_random(FCMMpiModel *m);
void fcm_mpi_init_kmeanspp(FCMMpiModel *m);
void fcm_mpi_init_domain(FCMMpiModel *m, int *domain_labels);

void   fcm_mpi_run(FCMMpiModel *m, InitStrategy strategy, int *domain_labels);
void   fcm_mpi_update_centroids(FCMMpiModel *m);
void   fcm_mpi_update_membership(FCMMpiModel *m);
double fcm_mpi_convergence(FCMMpiModel *m);

void fcm_mpi_gather_and_save(FCMMpiModel *m,
                              const char *membership_path,
                              const char *centroids_path);
void fcm_mpi_print_summary(FCMMpiModel *m);
void fcm_mpi_print_timing(FCMMpiModel *m);

double  l2_distance(const double *a, const double *b, int len);
double *alloc_flat(int rows, int cols);
void    copy_flat(double *dst, const double *src, int n);

#endif /* FCM_MPI_H */
