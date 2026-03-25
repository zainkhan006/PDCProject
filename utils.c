/* ============================================================
 * utils.c  —  Matrix helpers and distance functions
 * Member 1: Khansa Danish
 * ============================================================ */

#include "fcm.h"

/* ── Euclidean distance between two vectors of length `len` ── */
double euclidean_distance(const double *a, const double *b, int len) {
    double sum = 0.0;
    for (int f = 0; f < len; f++) {
        double diff = a[f] - b[f];
        sum += diff * diff;
    }
    return sqrt(sum);
}

/* ── Allocate a zero-initialised rows×cols matrix ─────────── */
double **alloc_matrix(int rows, int cols) {
    double **mat = (double **)malloc(rows * sizeof(double *));
    if (!mat) { fprintf(stderr, "alloc_matrix: malloc failed (rows)\n"); exit(1); }
    for (int i = 0; i < rows; i++) {
        mat[i] = (double *)calloc(cols, sizeof(double));
        if (!mat[i]) { fprintf(stderr, "alloc_matrix: calloc failed (row %d)\n", i); exit(1); }
    }
    return mat;
}

/* ── Free a 2-D matrix ────────────────────────────────────── */
void free_matrix(double **mat, int rows) {
    if (!mat) return;
    for (int i = 0; i < rows; i++) free(mat[i]);
    free(mat);
}

/* ── Deep copy src → dst (same dimensions) ───────────────── */
void copy_matrix(double **dst, double **src, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        memcpy(dst[i], src[i], cols * sizeof(double));
}

/* ── Load feature matrix from CSV into an existing model ─────── */
int fcm_load_csv(FCMModel *model, const char *filepath) {
    FILE *fp = fopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "[data] ERROR: cannot open '%s'\n", filepath);
        return -1;
    }

    int N = model->n_points;
    int F = model->n_features;
    char line[1 << 16];

    for (int i = 0; i < N; i++) {
        if (!fgets(line, sizeof(line), fp)) {
            fprintf(stderr, "[data] ERROR: unexpected EOF at row %d\n", i);
            fclose(fp);
            return -1;
        }
        char *token = strtok(line, ",\r\n");
        for (int f = 0; f < F; f++) {
            if (!token) {
                fprintf(stderr, "[data] ERROR: too few columns at row %d\n", i);
                fclose(fp);
                return -1;
            }
            model->data[i][f] = atof(token);
            token = strtok(NULL, ",\r\n");
        }
    }

    fclose(fp);
    printf("[data] Loaded %d x %d feature matrix from '%s'\n", N, F, filepath);
    return 0;
}
