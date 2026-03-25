#ifndef FCM_H
#define FCM_H

/* ============================================================
 * fcm.h  —  Fuzzy C-Means: data structures & function prototypes
 * Milestone 1: Serial implementation
 * Member 1: Khansa Danish
 * Project: Parallel Soft Clustering for Clinical Notes (OpenMPI)
 * IBA Karachi, Spring 2026
 * ============================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>

/* ─── Hyperparameters ──────────────────────────────────────── */
#define FCM_M          2.0      /* Fuzziness exponent (m > 1). m=2 is standard */
#define FCM_EPSILON    1e-5     /* Convergence threshold (Frobenius norm of ΔU) */
#define FCM_MAX_ITER   150      /* Maximum iterations before forced stop        */
#define FCM_MIN_DIST   1e-10    /* Guard against zero-distance (point == centroid) */

/* ─── Initialization strategy enum ────────────────────────── */
typedef enum {
    INIT_RANDOM  = 0,   /* Uniform random rows, row-normalized to sum to 1 */
    INIT_KMEANSPP = 1,  /* K-Means++ seeding (distance-proportional sampling) */
    INIT_DOMAIN  = 2    /* Domain-guided: seed from known clinical concept groups */
} InitStrategy;

/* ─── Core model struct ────────────────────────────────────── */
typedef struct {
    int     n_points;       /* N  — number of documents / data points          */
    int     n_features;     /* F  — dimensionality of each feature vector      */
    int     n_clusters;     /* C  — number of clusters                         */
    double  fuzziness;      /* m  — fuzziness exponent (copy of FCM_M)         */

    double **data;          /* [N x F] — feature matrix (TF-IDF or embeddings) */
    double **U;             /* [N x C] — membership matrix  (U[i][j] ∈ [0,1])  */
    double **centroids;     /* [C x F] — cluster centroid matrix                */

    int     iterations;     /* how many iterations ran until convergence        */
    double  final_delta;    /* final Frobenius norm (quality of convergence)    */
} FCMModel;

/* ─── fcm.c: lifecycle ─────────────────────────────────────── */
FCMModel *fcm_create(int n_points, int n_features, int n_clusters);
void      fcm_free(FCMModel *model);

/* ─── fcm.c: initialization strategies ────────────────────── */
void fcm_init_random(FCMModel *model);
void fcm_init_kmeanspp(FCMModel *model);
void fcm_init_domain(FCMModel *model, int *domain_labels);

/* ─── fcm.c: core algorithm ────────────────────────────────── */
void   fcm_run(FCMModel *model, InitStrategy strategy, int *domain_labels);
void   fcm_update_centroids(FCMModel *model);
void   fcm_update_membership(FCMModel *model);
double fcm_frobenius_delta(double **U_old, double **U_new, int n, int c);

/* ─── fcm.c: output ─────────────────────────────────────────── */
void fcm_print_summary(const FCMModel *model);
void fcm_save_membership(const FCMModel *model, const char *filepath);
void fcm_save_centroids(const FCMModel *model, const char *filepath);

/* ─── utils.c ───────────────────────────────────────────────── */
double  euclidean_distance(const double *a, const double *b, int len);
double **alloc_matrix(int rows, int cols);
void     free_matrix(double **mat, int rows);
void     copy_matrix(double **dst, double **src, int rows, int cols);

/* ─── data.c ────────────────────────────────────────────────── */
void fcm_generate_clinical_dummy(FCMModel *model, unsigned int seed);
int  fcm_load_csv(FCMModel *model, const char *filepath);

/* ─── normalize.c (Member 3) ────────────────────────────────── */
void normalize_l2(FCMModel *model);
void normalize_minmax(FCMModel *model);
void normalize_zscore(FCMModel *model);
void weight_idf(FCMModel *model, double threshold);
void analyze_feature_imbalance(const FCMModel *model, double var_threshold);

/* ─── metrics.c (Member 3) ──────────────────────────────────── */
int   *derive_hard_labels(const FCMModel *model);
double compute_silhouette(const FCMModel *model);
double compute_davies_bouldin(const FCMModel *model);
void   compute_all_metrics(const FCMModel *model, const char *strategy_name);

#endif /* FCM_H */