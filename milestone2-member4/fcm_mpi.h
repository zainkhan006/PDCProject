#ifndef FCM_MPI_H
#define FCM_MPI_H

/* ============================================================
 * fcm_mpi.h  —  Parallel Fuzzy C-Means with OpenMPI
 * Milestone 2: MPI-parallel implementation
 * Member 1: Khansa Danish
 * Project 21: Parallel Soft Clustering for Clinical Notes
 * IBA Karachi, Spring 2026
 *
 * Data: MTSamples clinical notes (4943 docs, 500 TF-IDF features)
 *       Provided by Member 2 (Ali Hamza)
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
#define FCM_EPSILON    1e-5     /* Convergence threshold (Frobenius norm of ΔU)*/
#define FCM_MAX_ITER   150      /* Maximum iterations                          */
#define FCM_MIN_DIST   1e-10    /* Guard against zero distance                 */

/* ─── Dataset dimensions (from Member 2's output) ─────────── */
#define TOTAL_DOCS     4943     /* N: total clinical documents                 */
#define N_FEATURES     500      /* F: TF-IDF features per document             */
#define N_CLUSTERS     10       /* C: number of clusters to find               */

/* ─── Initialisation strategies ───────────────────────────── */
typedef enum {
    INIT_RANDOM   = 0,
    INIT_KMEANSPP = 1,
    INIT_DOMAIN   = 2
} InitStrategy;

/* ─── Data distribution strategy (Member 4) ───────────────── */
typedef enum {
    DIST_BLOCK  = 0,   /* contiguous row chunks per rank             */
    DIST_CYCLIC = 1,   /* strided rows (i % P == rank) per rank      */
    DIST_DYNAMIC = 2   /* runtime load-balanced repartitioning        */
} DistStrategy;

/* ─── Communication strategy (Member 4 + Member 1 optimization path) ── */
typedef enum {
    COMM_BASELINE = 0,      /* Member 1 reference collective pattern        */
    COMM_NONBLOCK_OPT = 1   /* experimental non-blocking overlap variants   */
} CommStrategy;

/* ─── Per-rank local model ─────────────────────────────────── */
typedef struct {
    /* Global dimensions */
    int  N;              /* total number of documents (global)        */
    int  F;              /* number of features                        */
    int  C;              /* number of clusters                        */

    /* MPI info */
    int  rank;           /* this process's rank                       */
    int  n_procs;        /* total number of MPI processes             */
    DistStrategy dist;   /* block or cyclic partitioning              */
    CommStrategy comm;   /* baseline or non-blocking communication    */

    /* Local data slice for this rank */
    int  local_n;        /* number of rows owned by this rank         */
    int  local_start;    /* global row index of first local row       */
    int *local_ids;      /* [local_n] global row id for each local row*/
    double *local_data;  /* [local_n x F] flat row-major array        */
    double *local_U;     /* [local_n x C] local membership slice      */
    double *local_U_old; /* [local_n x C] previous U for convergence  */

    /* Global arrays (only rank 0 holds full versions) */
    double *all_U;       /* [N x C] full membership matrix (rank 0)   */
    double *centroids;   /* [C x F] cluster centroids (all ranks)     */
    double *init_centroids; /* [C x F] initialization anchors          */

    /* Convergence tracking */
    int    iterations;
    double final_delta;
    double iter_times[FCM_MAX_ITER]; /* wall time per iteration       */
    double iter_imbalance[FCM_MAX_ITER]; /* max/avg iteration time ratio */
    double iter_compute_times[FCM_MAX_ITER]; /* avg compute time/iter     */
    double iter_comm_times[FCM_MAX_ITER];    /* avg communication/iter    */
    double curr_iter_compute;
    double curr_iter_comm;
} FCMMpiModel;

/* ─── Function prototypes ──────────────────────────────────── */

/* Lifecycle */
FCMMpiModel *fcm_mpi_create(int N, int F, int C,
                            DistStrategy dist, CommStrategy comm);
void         fcm_mpi_free(FCMMpiModel *m);

/* Data loading and distribution */
int  fcm_mpi_load_and_scatter(FCMMpiModel *m,
                              const char *features_csv);
int *fcm_mpi_load_labels(FCMMpiModel *m, const char *labels_csv);

/* Initialisation */
void fcm_mpi_init_random(FCMMpiModel *m);
void fcm_mpi_init_kmeanspp(FCMMpiModel *m);
void fcm_mpi_init_domain(FCMMpiModel *m, int *domain_labels);

/* Core parallel algorithm */
void   fcm_mpi_run(FCMMpiModel *m, InitStrategy strategy,
                   int *domain_labels);
void   fcm_mpi_update_centroids(FCMMpiModel *m);   /* M-step */
void   fcm_mpi_update_membership(FCMMpiModel *m);  /* E-step */
double fcm_mpi_convergence(FCMMpiModel *m);         /* global Frobenius norm */

/* Output */
void fcm_mpi_gather_and_save(FCMMpiModel *m,
                             const char *membership_path,
                             const char *centroids_path,
                             const char *labels_path,
                             const char *feature_names_path);
void fcm_mpi_print_summary(FCMMpiModel *m);
void fcm_mpi_print_timing(FCMMpiModel *m);

/* Utilities */
double  l2_distance(const double *a, const double *b, int len);
double *alloc_flat(int rows, int cols);
void    copy_flat(double *dst, const double *src, int n);

#endif /* FCM_MPI_H */
