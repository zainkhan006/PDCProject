#ifndef FCM_MPI_H
#define FCM_MPI_H

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ─── Tunable constants ──────────────────────────────────────────────── */
#define N_CLUSTERS   10
#define N_FEATURES   500      /* F: TF-IDF vector dimension               */
#define TOTAL_DOCS   4943     /* N: total documents in corpus             */
#define FCM_M        2.0      /* fuzziness exponent  (m > 1)              */
#define FCM_EPSILON  1e-5     /* convergence threshold (Frobenius norm)   */
#define FCM_MAX_ITER 150      /* maximum iterations before giving up      */
#define FCM_MIN_DIST 1e-10    /* guard against divide-by-zero in E-step   */

/* Dynamic load-balancing thresholds */
#define DYNAMIC_WARMUP_ITERS    3     /* skip rebalance for first N iters */
#define DYNAMIC_CHECK_INTERVAL  3     /* check imbalance every N iters    */
#define DYNAMIC_IMBALANCE_THR   1.08  /* rebalance if max/avg > threshold */

/* ─── Enumerations ───────────────────────────────────────────────────── */
typedef enum { INIT_RANDOM = 0, INIT_KMEANSPP = 1, INIT_DOMAIN = 2 } InitStrategy;
typedef enum { DIST_BLOCK  = 0, DIST_CYCLIC   = 1, DIST_DYNAMIC = 2 } DistStrategy;
typedef enum { COMM_BASELINE = 0, COMM_NONBLOCK = 1 }                   CommStrategy;

/* ─── Core model struct ──────────────────────────────────────────────── */
typedef struct {
    /* MPI topology */
    int rank, n_procs;

    /* Dimensions */
    int N, F, C;

    /* Data distribution */
    DistStrategy dist_mode;
    CommStrategy comm_mode;
    int  local_n;       /* rows owned by this rank                        */
    int  local_start;   /* global row index of first local row (block)    */
    int *local_ids;     /* global row index for each local row            */

    /* Local data slices (flat row-major) */
    double *local_data; /* local_n × F feature matrix                     */
    double *local_U;    /* local_n × C current membership matrix          */
    double *local_U_old;/* local_n × C previous membership (convergence)  */

    /* Global arrays (all ranks hold identical copies) */
    double *centroids;  /* C × F centroid matrix                          */

    /* Per-iteration timing (rank-averaged) */
    double iter_times[FCM_MAX_ITER];
    double iter_compute[FCM_MAX_ITER];
    double iter_comm[FCM_MAX_ITER];
    double iter_imbalance[FCM_MAX_ITER];

    /* Convergence state */
    int    iterations;
    double final_delta;
} FCMMpiModel;

/* ─── Function prototypes ────────────────────────────────────────────── */

/* Lifecycle */
FCMMpiModel *fcm_mpi_create(int N, int F, int C,
                             DistStrategy dist, CommStrategy comm);
void         fcm_mpi_free(FCMMpiModel *m);

/* Data I/O */
int  fcm_mpi_load_and_scatter(FCMMpiModel *m,
                               const char *feat_path,
                               const char *label_path,
                               int **domain_labels_out);

/* Algorithm */
void fcm_mpi_run(FCMMpiModel *m, InitStrategy strategy,
                 int *domain_labels);
void fcm_mpi_update_membership(FCMMpiModel *m);
void fcm_mpi_update_centroids(FCMMpiModel *m);
double fcm_mpi_convergence(FCMMpiModel *m);

/* Output */
void fcm_mpi_gather_and_save(FCMMpiModel *m,
                              const char *mem_path,
                              const char *cen_path,
                              const char *feat_names_path);
void fcm_mpi_print_summary(FCMMpiModel *m);
void fcm_mpi_print_timing(FCMMpiModel *m);

/* Helpers */
static inline double l2_distance(const double *a, const double *b, int len)
{
    double s = 0.0;
    for (int i = 0; i < len; i++) { double d = a[i]-b[i]; s += d*d; }
    return sqrt(s);
}

#endif /* FCM_MPI_H */
