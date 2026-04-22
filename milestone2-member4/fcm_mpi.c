#include "fcm_mpi.h"

/* ============================================================
 * fcm_mpi.c  -  Parallel Fuzzy C-Means (OpenMPI)
 *
 * Milestone 2 - Member 1: Khansa Danish | IBA Karachi, Spring 2026
 *
 * -- PARALLEL DESIGN ------------------------------------------
 *
 *  Data distribution : MPI_Scatterv, block partition of N docs.
 *
 *  E-step (membership update):
 *    Each rank computes local_U for its own rows independently.
 *    No communication needed - embarrassingly parallel.
 *
 *  M-step (centroid update):
 *    Each rank accumulates local partial sums (numerator & denominator).
 *    MPI_Iallreduce (non-blocking) on numerator [C*F doubles] is
 *    issued first; MPI_Allreduce on denominator [C doubles] runs
 *    concurrently so network latency overlaps.  MPI_Wait syncs after.
 *    All ranks compute identical centroids - no extra broadcast needed.
 *
 *  Convergence detection:
 *    Each rank computes its local Frobenius norm2.
 *    One MPI_Allreduce(SUM) -> global norm. No master bottleneck.
 *
 * -- ROOT CAUSE OF THE 0.166667 (UNIFORM MEMBERSHIP) BUG ----
 *
 *  The core failure was centroid collapse: all 6 centroids became
 *  bit-for-bit identical, so every E-step returned u_ij = 1/C = 0.1667.
 *
 *  WHY centroids collapsed (domain-guided init):
 *    The old init computed each centroid as the arithmetic mean of all
 *    L2-normalised document vectors in that domain group.  Clinical TF-IDF
 *    vectors share dominant terms ("patient", "history", "mg", ...) so all
 *    4943 unit vectors point within a narrow cone on the 500-D unit sphere.
 *    Their per-cluster averages are nearly co-linear; after re-normalising,
 *    every centroid points in the direction of the global mean -> identical.
 *
 *  FIX (domain init - the only critical fix):
 *    Replace averaged-mean centroids with MEDOID selection:
 *    for each cluster, find the document closest to that cluster's mean
 *    (in L2 distance) and use it as the centroid.  Medoids are real data
 *    points already on the unit sphere; documents from different specialties
 *    genuinely differ even after normalisation, so medoid centroids are
 *    well-separated and the E-step produces non-trivial memberships.
 *
 *  FIX (MPI collective correctness):
 *    The old code had a conditional if/else-if pattern around MPI_Gatherv
 *    that skipped the collective on rank 0 when domain_labels was NULL.
 *    MPI_Gatherv is a collective - every rank must call it.  Restructured
 *    so all ranks always participate, then rank 0 does its serial work.
 *
 *  OTHER FIXES (from previous iteration, kept):
 *    FIX C: CSV header/index-column auto-detection.
 *    FIX D: fcm_mpi_load_labels() declared in header.
 *    FIX E/F/G: fgets checks, unused-param suppression, wide name buffer.
 *
 * ------------------------------------------------------------
 *  - Member 4: Arham (built on Khansa baseline)
 *
 * Arham tasks and productive results vs Khansa baseline:
 * 1) Added multiple data distribution strategies (block/cyclic/dynamic)
 *    with runtime rebalancing support and local row-ID tracking.
 *    Result vs baseline: moved from static block-only partitioning to
 *    adaptive ownership across ranks when imbalance appears.
 *
 * 2) Added per-iteration load-imbalance instrumentation plus compute/
 *    communication timing breakdown arrays and summary reporting.
 *    Result vs baseline: now provides measurable bottleneck diagnostics
 *    (time(avg), comp(avg), comm(avg), max/avg imbalance).
 *
 * 3) Kept and integrated non-blocking communication path, while adding
 *    communication strategy selection for reproducible comparison runs.
 *    Result vs baseline: can compare communication behavior under the
 *    same algorithmic flow without changing the rest of the pipeline.
 *
 * 4) Extended output pipeline with visualization-ready artifacts:
 *    viz_membership_sample.csv, viz_top_terms.csv,
 *    viz_label_cluster_comparison.csv.
 *    Result vs baseline: direct notebook ingestion for Member 4 analysis
 *    and report figures/tables.
 *
 * 5) Increased membership export precision to 12 decimals and improved
 *    gather/remap handling for dynamic distribution.
 *    Result vs baseline: reduced apparent uniform-rounding artifacts and
 *    preserved finer membership variation in exported CSV files.
 *
 * 6) Added/kept sparse-row numerical safeguards in membership/centroid
 *    updates and improved feature-name ingestion (2-column schema).
 *    Result vs baseline: better stability on edge rows and real token
 *    names in top-term outputs instead of placeholder feature labels.
 * ============================================================ */

typedef struct {
    int *counts_rows;      /* rows per rank */
    int *displs_rows;      /* row-displacements */
    int *counts_items;     /* counts_rows * item_width */
    int *displs_items;     /* displs_rows * item_width */
} DistMeta;

#define DYNAMIC_IMBALANCE_THRESHOLD 1.08
#define DYNAMIC_MIN_INTERVAL 3
#define DYNAMIC_WARMUP_ITERS 3

static void block_decompose(int N, int n_procs, int rank,
                            int *local_start, int *local_n) {
    int base = N / n_procs;
    int rem  = N % n_procs;
    *local_start = rank * base + (rank < rem ? rank : rem);
    *local_n     = base + (rank < rem ? 1 : 0);
}

static int cyclic_count(int N, int n_procs, int rank) {
    if (rank >= N) return 0;
    return (N - 1 - rank) / n_procs + 1;
}

static int already_chosen_seed(const int *seed_idx, int k, int candidate) {
    for (int i = 0; i < k; i++) {
        if (seed_idx[i] == candidate) return 1;
    }
    return 0;
}

static DistMeta build_dist_meta(const FCMMpiModel *m, int item_width) {
    DistMeta meta;
    int P = m->n_procs;
    meta.counts_rows  = (int *)malloc((size_t)P * sizeof(int));
    meta.displs_rows  = (int *)malloc((size_t)P * sizeof(int));
    meta.counts_items = (int *)malloc((size_t)P * sizeof(int));
    meta.displs_items = (int *)malloc((size_t)P * sizeof(int));

    if (m->dist == DIST_DYNAMIC) {
        MPI_Allgather(&m->local_n, 1, MPI_INT, meta.counts_rows, 1, MPI_INT, MPI_COMM_WORLD);
    }

    int running_rows = 0;
    int running_items = 0;
    for (int r = 0; r < P; r++) {
        int rows = meta.counts_rows[r];
        if (m->dist != DIST_DYNAMIC) {
            int start_unused = 0;
            if (m->dist == DIST_BLOCK) {
                block_decompose(m->N, P, r, &start_unused, &rows);
            } else {
                rows = cyclic_count(m->N, P, r);
            }
            meta.counts_rows[r] = rows;
        }
        meta.displs_rows[r] = running_rows;
        meta.counts_items[r] = rows * item_width;
        meta.displs_items[r] = running_items;

        running_rows += rows;
        running_items += rows * item_width;
    }
    return meta;
}

static void free_dist_meta(DistMeta *meta) {
    free(meta->counts_rows);
    free(meta->displs_rows);
    free(meta->counts_items);
    free(meta->displs_items);
    meta->counts_rows = NULL;
    meta->displs_rows = NULL;
    meta->counts_items = NULL;
    meta->displs_items = NULL;
}

static double *pack_rows_by_rank(const FCMMpiModel *m, const double *global_data,
                                 int item_width, const DistMeta *meta) {
    double *packed = alloc_flat(m->N, item_width);
    int P = m->n_procs;

    if (m->dist != DIST_CYCLIC) {
        copy_flat(packed, global_data, m->N * item_width);
        return packed;
    }

    for (int r = 0; r < P; r++) {
        int out_row = meta->displs_rows[r];
        for (int g = r; g < m->N; g += P) {
            memcpy(packed + (size_t)out_row * item_width,
                   global_data + (size_t)g * item_width,
                   (size_t)item_width * sizeof(double));
            out_row++;
        }
    }
    return packed;
}

static int *pack_int_rows_by_rank(const FCMMpiModel *m, const int *global_labels,
                                  const DistMeta *meta) {
    int *packed = (int *)calloc((size_t)m->N, sizeof(int));
    int P = m->n_procs;

    if (m->dist != DIST_CYCLIC) {
        memcpy(packed, global_labels, (size_t)m->N * sizeof(int));
        return packed;
    }

    for (int r = 0; r < P; r++) {
        int out_row = meta->displs_rows[r];
        for (int g = r; g < m->N; g += P) {
            packed[out_row++] = global_labels[g];
        }
    }
    return packed;
}

static void unpack_rows_to_global(const FCMMpiModel *m, const double *packed,
                                  double *global_data, int item_width,
                                  const DistMeta *meta) {
    int P = m->n_procs;
    if (m->dist != DIST_CYCLIC) {
        copy_flat(global_data, packed, m->N * item_width);
        return;
    }

    for (int r = 0; r < P; r++) {
        int in_row = meta->displs_rows[r];
        for (int g = r; g < m->N; g += P) {
            memcpy(global_data + (size_t)g * item_width,
                   packed + (size_t)in_row * item_width,
                   (size_t)item_width * sizeof(double));
            in_row++;
        }
    }
}

static void normalise_rows(double *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double norm2 = 0.0;
        for (int f = 0; f < cols; f++) {
            double v = mat[i * cols + f];
            norm2 += v * v;
        }
        double norm = sqrt(norm2);
        if (norm < 1e-12) continue;
        for (int f = 0; f < cols; f++) {
            mat[i * cols + f] /= norm;
        }
    }
}

static int csv_has_header(FILE *fp) {
    long pos = ftell(fp);
    int ch;
    while ((ch = fgetc(fp)) != EOF) {
        if (ch != ' ' && ch != '\t' &&
            ch != 0xef && ch != 0xbb && ch != 0xbf) break;
    }
    fseek(fp, pos, SEEK_SET);
    return (ch != EOF && (ch == '"' || ch == '\'' ||
                          (ch >= 'A' && ch <= 'Z') ||
                          (ch >= 'a' && ch <= 'z') ||
                          ch == '_'));
}

static int csv_has_index_col(const char *line) {
    const char *p = line;
    while (*p == ' ') p++;
    if (*p == '"' || *p == '\'') return 0;
    int has_dot = 0, has_e = 0, ndigits = 0;
    while (*p && *p != ',' && *p != '\n' && *p != '\r') {
        if (*p == '.') has_dot = 1;
        else if (*p == 'e' || *p == 'E') has_e = 1;
        else if (*p >= '0' && *p <= '9') ndigits++;
        p++;
    }
    return (ndigits > 0 && !has_dot && !has_e);
}

double l2_distance(const double *a, const double *b, int len) {
    double s = 0.0;
    for (int i = 0; i < len; i++) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return sqrt(s);
}

double *alloc_flat(int rows, int cols) {
    double *p = (double *)calloc((size_t)rows * cols, sizeof(double));
    if (!p) {
        fprintf(stderr, "[fcm_mpi] calloc failed (%d x %d)\n", rows, cols);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return p;
}

void copy_flat(double *dst, const double *src, int n) {
    memcpy(dst, src, (size_t)n * sizeof(double));
}

static void resize_local_buffers(FCMMpiModel *m, int new_local_n) {
    if (new_local_n == m->local_n) return;

    int *new_ids = (int *)realloc(m->local_ids, (size_t)new_local_n * sizeof(int));
    double *new_data = (double *)realloc(m->local_data, (size_t)new_local_n * m->F * sizeof(double));
    double *new_u = (double *)realloc(m->local_U, (size_t)new_local_n * m->C * sizeof(double));
    double *new_u_old = (double *)realloc(m->local_U_old, (size_t)new_local_n * m->C * sizeof(double));

    if (!new_ids || !new_data || !new_u || !new_u_old) {
        fprintf(stderr, "[fcm_mpi] realloc failed during dynamic rebalance\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    m->local_ids = new_ids;
    m->local_data = new_data;
    m->local_U = new_u;
    m->local_U_old = new_u_old;
    m->local_n = new_local_n;
}

static double fcm_mpi_dynamic_rebalance(FCMMpiModel *m, double iter_local, int iter) {
    if (m->dist != DIST_DYNAMIC) return 0.0;

    double t0 = MPI_Wtime();
    int P = m->n_procs;

    int *curr_counts = (int *)malloc((size_t)P * sizeof(int));
    double *iter_times = (double *)malloc((size_t)P * sizeof(double));
    int *target_counts = (int *)malloc((size_t)P * sizeof(int));
    if (!curr_counts || !iter_times || !target_counts) {
        fprintf(stderr, "[fcm_mpi] malloc failed during dynamic rebalance\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Allgather(&m->local_n, 1, MPI_INT, curr_counts, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&iter_local, 1, MPI_DOUBLE, iter_times, 1, MPI_DOUBLE, MPI_COMM_WORLD);

    int *target_displs = (int *)malloc((size_t)P * sizeof(int));
    if (!target_displs) {
        fprintf(stderr, "[fcm_mpi] malloc failed during dynamic rebalance\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (m->rank == 0) {
        double *throughput = (double *)malloc((size_t)P * sizeof(double));
        if (!throughput) {
            fprintf(stderr, "[fcm_mpi] malloc failed during dynamic rebalance\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        double sum_t = 0.0;
        for (int r = 0; r < P; r++) {
            double denom = iter_times[r] > 1e-12 ? iter_times[r] : 1e-12;
            throughput[r] = (double)curr_counts[r] / denom;
            sum_t += throughput[r];
        }

        int min_rows = (m->N >= P) ? 1 : 0;
        int remaining = m->N - min_rows * P;
        int assigned = 0;

        for (int r = 0; r < P; r++) {
            int bonus = (sum_t > 0.0) ? (int)((remaining * throughput[r]) / sum_t) : (remaining / P);
            target_counts[r] = min_rows + bonus;
            assigned += target_counts[r];
        }

        while (assigned < m->N) {
            int best = 0;
            for (int r = 1; r < P; r++) {
                if (throughput[r] > throughput[best]) best = r;
            }
            target_counts[best]++;
            assigned++;
        }
        while (assigned > m->N) {
            int worst = 0;
            for (int r = 1; r < P; r++) {
                if (throughput[r] < throughput[worst]) worst = r;
            }
            if (target_counts[worst] > min_rows) {
                target_counts[worst]--;
                assigned--;
            } else {
                break;
            }
        }

        free(throughput);
    }

    MPI_Bcast(target_counts, P, MPI_INT, 0, MPI_COMM_WORLD);

    target_displs[0] = 0;
    for (int r = 1; r < P; r++) {
        target_displs[r] = target_displs[r - 1] + target_counts[r - 1];
    }

    DistMeta meta_data = build_dist_meta(m, m->F);
    DistMeta meta_u = build_dist_meta(m, m->C);
    DistMeta meta_ids = build_dist_meta(m, 1);

    double *packed_data = (m->rank == 0) ? alloc_flat(m->N, m->F) : NULL;
    double *packed_u = (m->rank == 0) ? alloc_flat(m->N, m->C) : NULL;
    int *packed_ids = (m->rank == 0) ? (int *)malloc((size_t)m->N * sizeof(int)) : NULL;

    MPI_Gatherv(m->local_data, m->local_n * m->F, MPI_DOUBLE,
                packed_data, meta_data.counts_items, meta_data.displs_items, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(m->local_U, m->local_n * m->C, MPI_DOUBLE,
                packed_u, meta_u.counts_items, meta_u.displs_items, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(m->local_ids, m->local_n, MPI_INT,
                packed_ids, meta_ids.counts_rows, meta_ids.displs_rows, MPI_INT,
                0, MPI_COMM_WORLD);

    double *out_data = NULL;
    double *out_u = NULL;
    int *out_ids = NULL;

    if (m->rank == 0) {
        double *global_data = alloc_flat(m->N, m->F);
        double *global_u = alloc_flat(m->N, m->C);

        for (int i = 0; i < m->N; i++) {
            int gid = packed_ids[i];
            if (gid < 0 || gid >= m->N) continue;
            memcpy(global_data + (size_t)gid * m->F,
                   packed_data + (size_t)i * m->F,
                   (size_t)m->F * sizeof(double));
            memcpy(global_u + (size_t)gid * m->C,
                   packed_u + (size_t)i * m->C,
                   (size_t)m->C * sizeof(double));
        }

        out_data = alloc_flat(m->N, m->F);
        out_u = alloc_flat(m->N, m->C);
        out_ids = (int *)malloc((size_t)m->N * sizeof(int));

        int cursor = 0;
        for (int r = 0; r < P; r++) {
            int start = target_displs[r];
            for (int i = 0; i < target_counts[r]; i++) {
                int gid = cursor++;
                int out_row = start + i;
                out_ids[out_row] = gid;
                memcpy(out_data + (size_t)out_row * m->F,
                       global_data + (size_t)gid * m->F,
                       (size_t)m->F * sizeof(double));
                memcpy(out_u + (size_t)out_row * m->C,
                       global_u + (size_t)gid * m->C,
                       (size_t)m->C * sizeof(double));
            }
        }

        free(global_data);
        free(global_u);
    }

    int new_local_n = 0;
    MPI_Scatter(target_counts, 1, MPI_INT, &new_local_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    resize_local_buffers(m, new_local_n);

    int *send_counts_data = NULL, *send_displs_data = NULL;
    int *send_counts_u = NULL, *send_displs_u = NULL;
    int *send_counts_ids = NULL, *send_displs_ids = NULL;

    if (m->rank == 0) {
        send_counts_data = (int *)malloc((size_t)P * sizeof(int));
        send_displs_data = (int *)malloc((size_t)P * sizeof(int));
        send_counts_u = (int *)malloc((size_t)P * sizeof(int));
        send_displs_u = (int *)malloc((size_t)P * sizeof(int));
        send_counts_ids = (int *)malloc((size_t)P * sizeof(int));
        send_displs_ids = (int *)malloc((size_t)P * sizeof(int));

        int run_rows = 0;
        int run_data = 0;
        int run_u = 0;
        for (int r = 0; r < P; r++) {
            send_counts_ids[r] = target_counts[r];
            send_displs_ids[r] = run_rows;
            send_counts_data[r] = target_counts[r] * m->F;
            send_displs_data[r] = run_data;
            send_counts_u[r] = target_counts[r] * m->C;
            send_displs_u[r] = run_u;
            run_rows += target_counts[r];
            run_data += target_counts[r] * m->F;
            run_u += target_counts[r] * m->C;
        }
    }

    MPI_Scatterv(out_data, send_counts_data, send_displs_data, MPI_DOUBLE,
                 m->local_data, m->local_n * m->F, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(out_u, send_counts_u, send_displs_u, MPI_DOUBLE,
                 m->local_U, m->local_n * m->C, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(out_ids, send_counts_ids, send_displs_ids, MPI_INT,
                 m->local_ids, m->local_n, MPI_INT, 0, MPI_COMM_WORLD);

    copy_flat(m->local_U_old, m->local_U, m->local_n * m->C);
    m->local_start = (m->local_n > 0) ? m->local_ids[0] : -1;

    if (m->rank == 0) {
        printf("[member4] Dynamic rebalance iter %d: rows redistributed by measured throughput\n", iter);
    }

    free(send_counts_data);
    free(send_displs_data);
    free(send_counts_u);
    free(send_displs_u);
    free(send_counts_ids);
    free(send_displs_ids);

    free(out_data);
    free(out_u);
    free(out_ids);
    free(packed_data);
    free(packed_u);
    free(packed_ids);

    free_dist_meta(&meta_data);
    free_dist_meta(&meta_u);
    free_dist_meta(&meta_ids);

    free(curr_counts);
    free(iter_times);
    free(target_counts);
    free(target_displs);

    return MPI_Wtime() - t0;
}

FCMMpiModel *fcm_mpi_create(int N, int F, int C,
                            DistStrategy dist, CommStrategy comm) {
    FCMMpiModel *m = (FCMMpiModel *)calloc(1, sizeof(FCMMpiModel));
    if (!m) {
        fprintf(stderr, "calloc FCMMpiModel failed\n");
        exit(1);
    }

    m->N = N;
    m->F = F;
    m->C = C;
    m->dist = dist;
    m->comm = comm;

    MPI_Comm_rank(MPI_COMM_WORLD, &m->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &m->n_procs);

    if (dist == DIST_CYCLIC) {
        m->local_start = -1;
        m->local_n = cyclic_count(N, m->n_procs, m->rank);
    } else {
        block_decompose(N, m->n_procs, m->rank, &m->local_start, &m->local_n);
    }

    m->local_ids = (int *)malloc((size_t)m->local_n * sizeof(int));
    for (int i = 0; i < m->local_n; i++) {
        m->local_ids[i] = (dist == DIST_CYCLIC)
            ? (m->rank + i * m->n_procs)
            : (m->local_start + i);
    }

    m->local_data  = alloc_flat(m->local_n, F);
    m->local_U     = alloc_flat(m->local_n, C);
    m->local_U_old = alloc_flat(m->local_n, C);
    m->centroids   = alloc_flat(C, F);
    m->init_centroids = alloc_flat(C, F);
    m->all_U       = (m->rank == 0) ? alloc_flat(N, C) : NULL;

    return m;
}

void fcm_mpi_free(FCMMpiModel *m) {
    if (!m) return;
    free(m->local_ids);
    free(m->local_data);
    free(m->local_U);
    free(m->local_U_old);
    free(m->centroids);
    free(m->init_centroids);
    free(m->all_U);
    free(m);
}

int fcm_mpi_load_and_scatter(FCMMpiModel *m, const char *features_csv) {
    DistMeta meta = build_dist_meta(m, m->F);
    double *global_data = NULL;
    double *packed_data = NULL;

    if (m->rank == 0) {
        global_data = alloc_flat(m->N, m->F);

        FILE *fp = fopen(features_csv, "r");
        if (!fp) {
            fprintf(stderr, "[rank 0] ERROR: cannot open '%s'\n", features_csv);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        char line[1 << 20];
        int skip_index = 0;

        if (csv_has_header(fp)) {
            if (!fgets(line, sizeof(line), fp)) {
                fprintf(stderr, "[rank 0] features.csv appears empty\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            printf("[rank 0] Skipped header row in '%s'\n", features_csv);
        }

        long data_start = ftell(fp);
        if (fgets(line, sizeof(line), fp)) {
            skip_index = csv_has_index_col(line);
            if (skip_index) {
                printf("[rank 0] Will skip row-index column in '%s'\n", features_csv);
            }
        }
        fseek(fp, data_start, SEEK_SET);

        for (int i = 0; i < m->N; i++) {
            if (!fgets(line, sizeof(line), fp)) {
                fprintf(stderr, "[rank 0] Unexpected EOF at row %d\n", i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            char *tok = strtok(line, ",\n\r");
            if (skip_index) tok = strtok(NULL, ",\n\r");
            for (int f = 0; f < m->F; f++) {
                global_data[i * m->F + f] = tok ? atof(tok) : 0.0;
                tok = strtok(NULL, ",\n\r");
            }
        }
        fclose(fp);

        /* Keep feature vectors on unit sphere to stabilize distance geometry. */
        normalise_rows(global_data, m->N, m->F);

        packed_data = pack_rows_by_rank(m, global_data, m->F, &meta);

        printf("[rank 0] Loaded %d x %d feature matrix from '%s'\n",
               m->N, m->F, features_csv);
         printf("[rank 0] Distribution mode: %s\n",
             (m->dist == DIST_BLOCK) ? "block" :
             (m->dist == DIST_CYCLIC) ? "cyclic" : "dynamic(load-balanced)");
    }

    MPI_Scatterv(packed_data,
                 meta.counts_items,
                 meta.displs_items,
                 MPI_DOUBLE,
                 m->local_data,
                 m->local_n * m->F,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    int local_rows = m->local_n;
    int global_min = 0, global_max = 0, global_sum_rows = 0;
    MPI_Allreduce(&local_rows, &global_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_rows, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_rows, &global_sum_rows, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (m->rank == 0) {
        double avg = (double)global_sum_rows / m->n_procs;
        double ratio = (avg > 0.0) ? (global_max / avg) : 0.0;
        printf("[member4] Row balance: min=%d max=%d avg=%.2f max/avg=%.3f\n",
               global_min, global_max, avg, ratio);
    }

    if (m->rank == 0) {
        free(global_data);
        free(packed_data);
    }
    free_dist_meta(&meta);
    return 0;
}

int *fcm_mpi_load_labels(FCMMpiModel *m, const char *labels_csv) {
    int *labels = NULL;
    if (m->rank == 0) {
        labels = (int *)malloc((size_t)m->N * sizeof(int));
        FILE *fp = fopen(labels_csv, "r");
        if (!fp) {
            fprintf(stderr, "[rank 0] Cannot open labels file '%s'\n", labels_csv);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        char line[1024];
        if (!fgets(line, sizeof(line), fp)) {
            fprintf(stderr, "[rank 0] Labels file is empty: '%s'\n", labels_csv);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        char known[128][256];
        int n_known = 0;

        for (int i = 0; i < m->N; i++) {
            if (!fgets(line, sizeof(line), fp)) {
                fprintf(stderr, "[rank 0] Labels EOF at row %d\n", i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            line[strcspn(line, "\r\n")] = '\0';

            char *s = line;
            while (*s == ' ' || *s == '\t') s++;

            int id = -1;
            for (int k = 0; k < n_known; k++) {
                if (strcmp(known[k], s) == 0) {
                    id = k;
                    break;
                }
            }
            if (id == -1) {
                if (n_known >= 128) {
                    id = 127;
                } else {
                    strncpy(known[n_known], s, 255);
                    known[n_known][255] = '\0';
                    id = n_known;
                    n_known++;
                }
            }
            labels[i] = id % m->C;
        }
        fclose(fp);
        printf("[rank 0] Loaded %d labels (%d unique, mapped to %d clusters)\n",
               m->N, n_known, m->C);
    }
    return labels;
}

void fcm_mpi_init_random(FCMMpiModel *m) {
    DistMeta meta = build_dist_meta(m, m->C);
    double *global_U = NULL;
    double *packed_U = NULL;

    if (m->rank == 0) {
        srand((unsigned int)time(NULL));
        global_U = alloc_flat(m->N, m->C);
        for (int i = 0; i < m->N; i++) {
            double sum = 0.0;
            for (int j = 0; j < m->C; j++) {
                global_U[i * m->C + j] = (double)rand() / RAND_MAX + 1e-9;
                sum += global_U[i * m->C + j];
            }
            for (int j = 0; j < m->C; j++) global_U[i * m->C + j] /= sum;
        }
        packed_U = pack_rows_by_rank(m, global_U, m->C, &meta);
    }

    MPI_Scatterv(packed_U,
                 meta.counts_items,
                 meta.displs_items,
                 MPI_DOUBLE,
                 m->local_U,
                 m->local_n * m->C,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    if (m->rank == 0) {
        free(global_U);
        free(packed_U);
        printf("[init] Random init complete.\n");
    }
    free_dist_meta(&meta);
}

static void init_compute_u(FCMMpiModel *m) {
    double expn = 2.0 / (FCM_M - 1.0);
    double *dist = (double *)malloc((size_t)m->C * sizeof(double));

    for (int i = 0; i < m->local_n; i++) {
        int zero_k = -1;
        for (int c = 0; c < m->C; c++) {
            dist[c] = l2_distance(m->local_data + (size_t)i * m->F,
                                  m->centroids + (size_t)c * m->F,
                                  m->F);
            if (dist[c] < FCM_MIN_DIST) zero_k = c;
        }

        if (zero_k >= 0) {
            for (int c = 0; c < m->C; c++) {
                m->local_U[i * m->C + c] = (c == zero_k) ? 1.0 : 0.0;
            }
            continue;
        }

        for (int c = 0; c < m->C; c++) {
            double sum = 0.0;
            for (int k = 0; k < m->C; k++) {
                sum += pow(dist[c] / dist[k], expn);
            }
            m->local_U[i * m->C + c] = 1.0 / sum;
        }
    }

    free(dist);
}

void fcm_mpi_init_kmeanspp(FCMMpiModel *m) {
    DistMeta meta = build_dist_meta(m, m->F);
    double *packed_data = (m->rank == 0) ? alloc_flat(m->N, m->F) : NULL;
    double *global_data = (m->rank == 0) ? alloc_flat(m->N, m->F) : NULL;

    MPI_Gatherv(m->local_data,
                m->local_n * m->F,
                MPI_DOUBLE,
                packed_data,
                meta.counts_items,
                meta.displs_items,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    if (m->rank == 0) {
        unpack_rows_to_global(m, packed_data, global_data, m->F, &meta);
        srand((unsigned int)time(NULL));

        /* Member 1 K-Means++: random first seed, D² proportional sampling. */
        int *seed = (int *)malloc((size_t)m->C * sizeof(int));
        double *md2 = (double *)malloc((size_t)m->N * sizeof(double));

        seed[0] = rand() % m->N;
        for (int k = 1; k < m->C; k++) {
            double total = 0.0;
            for (int i = 0; i < m->N; i++) {
                double best = DBL_MAX;
                for (int s = 0; s < k; s++) {
                    double d = l2_distance(global_data + (size_t)i * m->F,
                                           global_data + (size_t)seed[s] * m->F,
                                           m->F);
                    if (d * d < best) best = d * d;
                }
                md2[i] = best;
                total += best;
            }
            double r2 = ((double)rand() / RAND_MAX) * total;
            double cum = 0.0;
            seed[k] = 0;
            for (int i = 0; i < m->N; i++) {
                cum += md2[i];
                if (cum >= r2) { seed[k] = i; break; }
            }
        }
        for (int k = 0; k < m->C; k++) {
            memcpy(m->centroids + (size_t)k * m->F,
                   global_data + (size_t)seed[k] * m->F,
                   (size_t)m->F * sizeof(double));
        }
        free(seed);
        free(md2);
        free(packed_data);
        free(global_data);
        printf("[init] K-Means++ init complete.\n");
    }

    MPI_Bcast(m->centroids, m->C * m->F, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free_dist_meta(&meta);
}

void fcm_mpi_init_domain(FCMMpiModel *m, int *domain_labels) {
    DistMeta meta_data = build_dist_meta(m, m->F);
    DistMeta meta_label = build_dist_meta(m, 1);

    double *packed_data = (m->rank == 0) ? alloc_flat(m->N, m->F) : NULL;
    double *global_data = (m->rank == 0) ? alloc_flat(m->N, m->F) : NULL;

    MPI_Gatherv(m->local_data,
                m->local_n * m->F,
                MPI_DOUBLE,
                packed_data,
                meta_data.counts_items,
                meta_data.displs_items,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    if (m->rank == 0) {
        unpack_rows_to_global(m, packed_data, global_data, m->F, &meta_data);

        double *sums = alloc_flat(m->C, m->F);
        int *cnts = (int *)calloc((size_t)m->C, sizeof(int));

        for (int i = 0; i < m->N; i++) {
            int c = domain_labels[i];
            if (c < 0 || c >= m->C) c = 0;
            cnts[c]++;
            for (int f = 0; f < m->F; f++) {
                sums[c * m->F + f] += global_data[i * m->F + f];
            }
        }

        for (int c = 0; c < m->C; c++) {
            if (cnts[c] > 0) {
                for (int f = 0; f < m->F; f++) {
                    sums[c * m->F + f] /= cnts[c];
                }
            }
        }

        /* Medoid centroids prevent mean-vector collapse into near-identical directions. */
        for (int c = 0; c < m->C; c++) {
            int best_idx = -1;
            double best_d = DBL_MAX;

            if (cnts[c] == 0) {
                best_idx = rand() % m->N;
            } else {
                for (int i = 0; i < m->N; i++) {
                    int dc = domain_labels[i];
                    if (dc < 0 || dc >= m->C) dc = 0;
                    if (dc != c) continue;

                    double d = l2_distance(global_data + (size_t)i * m->F,
                                           sums + (size_t)c * m->F,
                                           m->F);
                    if (d < best_d) {
                        best_d = d;
                        best_idx = i;
                    }
                }
            }

            if (best_idx < 0) best_idx = rand() % m->N;
            memcpy(m->centroids + (size_t)c * m->F,
                   global_data + (size_t)best_idx * m->F,
                   (size_t)m->F * sizeof(double));
        }

        free(sums);
        free(cnts);
        free(packed_data);
        free(global_data);
        printf("[init] Domain-guided init complete (medoid selection).\n");
    }

    MPI_Bcast(m->centroids, m->C * m->F, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int *local_labels = (int *)malloc((size_t)m->local_n * sizeof(int));
    int *packed_labels = NULL;
    if (m->rank == 0) packed_labels = pack_int_rows_by_rank(m, domain_labels, &meta_label);

    MPI_Scatterv(packed_labels,
                 meta_label.counts_rows,
                 meta_label.displs_rows,
                 MPI_INT,
                 local_labels,
                 m->local_n,
                 MPI_INT,
                 0,
                 MPI_COMM_WORLD);

    for (int i = 0; i < m->local_n; i++) {
        int dom = local_labels[i];
        if (dom < 0 || dom >= m->C) dom = 0;
        double rs = 0.0;
        for (int j = 0; j < m->C; j++) {
            m->local_U[i * m->C + j] = (j == dom) ? 0.7 : 0.3 / (m->C - 1);
            rs += m->local_U[i * m->C + j];
        }
        for (int j = 0; j < m->C; j++) {
            m->local_U[i * m->C + j] /= rs;
        }
    }

    free(local_labels);
    if (m->rank == 0) free(packed_labels);
    free_dist_meta(&meta_data);
    free_dist_meta(&meta_label);
}

void fcm_mpi_update_membership(FCMMpiModel *m) {
    double exp = 2.0 / (FCM_M - 1.0);
    double *dist = (double *)malloc((size_t)m->C * sizeof(double));

    for (int i = 0; i < m->local_n; i++) {
        double row_norm2 = 0.0;
        for (int f = 0; f < m->F; f++) {
            double v = m->local_data[(size_t)i * m->F + f];
            row_norm2 += v * v;
        }

        if (row_norm2 < 1e-20) {
            double u = 1.0 / (double)m->C;
            for (int c = 0; c < m->C; c++) {
                m->local_U[i * m->C + c] = u;
            }
            continue;
        }

        int zero_cluster = -1;

        for (int c = 0; c < m->C; c++) {
            dist[c] = l2_distance(m->local_data + (size_t)i * m->F,
                                  m->centroids + (size_t)c * m->F,
                                  m->F);
            if (dist[c] < FCM_MIN_DIST) zero_cluster = c;
        }

        if (zero_cluster >= 0) {
            for (int c = 0; c < m->C; c++) {
                m->local_U[i * m->C + c] = (c == zero_cluster) ? 1.0 : 0.0;
            }
            continue;
        }

        for (int c = 0; c < m->C; c++) {
            double sum = 0.0;
            for (int k = 0; k < m->C; k++) {
                double ratio = dist[c] / dist[k];
                sum += pow(ratio, exp);
            }
            m->local_U[i * m->C + c] = 1.0 / sum;
        }
    }

    free(dist);
}

void fcm_mpi_update_centroids(FCMMpiModel *m) {
    double t_comp0 = MPI_Wtime();
    double *local_num  = alloc_flat(m->C, m->F);
    double *global_num = alloc_flat(m->C, m->F);
    double *local_den  = (double *)calloc((size_t)m->C, sizeof(double));
    double *global_den = (double *)calloc((size_t)m->C, sizeof(double));

    for (int i = 0; i < m->local_n; i++) {
        double row_norm2 = 0.0;
        for (int f = 0; f < m->F; f++) {
            double v = m->local_data[(size_t)i * m->F + f];
            row_norm2 += v * v;
        }
        if (row_norm2 < 1e-20) continue;

        for (int c = 0; c < m->C; c++) {
            double u_m = pow(m->local_U[i * m->C + c], FCM_M);
            local_den[c] += u_m;
            for (int f = 0; f < m->F; f++) {
                local_num[c * m->F + f] += u_m * m->local_data[i * m->F + f];
            }
        }
    }
    m->curr_iter_compute += (MPI_Wtime() - t_comp0);

    double t_comm0 = MPI_Wtime();
    if (m->comm == COMM_NONBLOCK_OPT) {
        MPI_Request reqs[2];
        MPI_Iallreduce(local_num, global_num, m->C * m->F,
                       MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &reqs[0]);
        MPI_Iallreduce(local_den, global_den, m->C,
                       MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    } else {
        /* Member 1 reference pattern for reproducible baseline results. */
        MPI_Request req;
        MPI_Iallreduce(local_num, global_num, m->C * m->F,
                       MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &req);
        MPI_Allreduce(local_den, global_den, m->C,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
    m->curr_iter_comm += (MPI_Wtime() - t_comm0);

    t_comp0 = MPI_Wtime();
    for (int c = 0; c < m->C; c++) {
        double denom = (global_den[c] < FCM_MIN_DIST) ? FCM_MIN_DIST : global_den[c];
        for (int f = 0; f < m->F; f++) {
            m->centroids[c * m->F + f] = global_num[c * m->F + f] / denom;
        }
    }

    normalise_rows(m->centroids, m->C, m->F);  /* keep centroids on unit sphere (Member 1 fix) */

    m->curr_iter_compute += (MPI_Wtime() - t_comp0);

    free(local_num);
    free(global_num);
    free(local_den);
    free(global_den);
}

double fcm_mpi_convergence(FCMMpiModel *m) {
    double t_comp0 = MPI_Wtime();
    double local_norm2 = 0.0;
    for (int i = 0; i < m->local_n; i++) {
        for (int c = 0; c < m->C; c++) {
            double d = m->local_U[i * m->C + c] - m->local_U_old[i * m->C + c];
            local_norm2 += d * d;
        }
    }
    m->curr_iter_compute += (MPI_Wtime() - t_comp0);

    double global_norm2 = 0.0;
    double t_comm0 = MPI_Wtime();
    if (m->comm == COMM_NONBLOCK_OPT) {
        MPI_Request req;
        MPI_Iallreduce(&local_norm2, &global_norm2, 1,
                       MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    } else {
        MPI_Allreduce(&local_norm2, &global_norm2, 1,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    m->curr_iter_comm += (MPI_Wtime() - t_comm0);

    t_comp0 = MPI_Wtime();
    double delta = sqrt(global_norm2);
    m->curr_iter_compute += (MPI_Wtime() - t_comp0);
    return delta;
}

void fcm_mpi_run(FCMMpiModel *m, InitStrategy strategy, int *domain_labels) {
    if (m->rank == 0) {
        printf("\n[fcm_mpi] N=%d  F=%d  C=%d  P=%d  m=%.1f  eps=%.0e\n",
               m->N, m->F, m->C, m->n_procs, FCM_M, FCM_EPSILON);
    }

    switch (strategy) {
        case INIT_KMEANSPP:
            fcm_mpi_init_kmeanspp(m);
            break;
        case INIT_DOMAIN:
            fcm_mpi_init_domain(m, domain_labels);
            break;
        default:
            fcm_mpi_init_random(m);
            break;
    }

    int last_rebalance_iter = 0;

    for (int iter = 1; iter <= FCM_MAX_ITER; iter++) {
        double t0 = MPI_Wtime();
        m->curr_iter_compute = 0.0;
        m->curr_iter_comm = 0.0;

        copy_flat(m->local_U_old, m->local_U, m->local_n * m->C);
        double t_comp0 = MPI_Wtime();
        fcm_mpi_update_membership(m);
        m->curr_iter_compute += (MPI_Wtime() - t_comp0);
        fcm_mpi_update_centroids(m);
        double delta = fcm_mpi_convergence(m);

        double iter_local = MPI_Wtime() - t0;

        double t_min = 0.0, t_max = 0.0, t_sum = 0.0;
        MPI_Allreduce(&iter_local, &t_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&iter_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&iter_local, &t_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double t_avg = t_sum / m->n_procs;
        double imbalance = (t_avg > 0.0) ? (t_max / t_avg) : 0.0;

        if (m->dist == DIST_DYNAMIC) {
            int enough_warmup = (iter >= DYNAMIC_WARMUP_ITERS);
            int enough_gap = ((iter - last_rebalance_iter) >= DYNAMIC_MIN_INTERVAL);
            if (enough_warmup && enough_gap && imbalance > DYNAMIC_IMBALANCE_THRESHOLD) {
                double t_rebal = fcm_mpi_dynamic_rebalance(m, iter_local, iter);
                m->curr_iter_comm += t_rebal;
                iter_local += t_rebal;

                MPI_Allreduce(&iter_local, &t_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
                MPI_Allreduce(&iter_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                MPI_Allreduce(&iter_local, &t_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                t_avg = t_sum / m->n_procs;
                imbalance = (t_avg > 0.0) ? (t_max / t_avg) : 0.0;
                last_rebalance_iter = iter;
            }
        }

        m->iter_times[iter - 1] = iter_local;

        double compute_local = m->curr_iter_compute;
        double comm_local = m->curr_iter_comm;
        double compute_sum = 0.0, comm_sum = 0.0;
        MPI_Allreduce(&compute_local, &compute_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&comm_local, &comm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        m->iter_compute_times[iter - 1] = compute_sum / m->n_procs;
        m->iter_comm_times[iter - 1] = comm_sum / m->n_procs;

        m->iter_imbalance[iter - 1] = imbalance;

        if (m->rank == 0 && (iter <= 5 || iter % 10 == 0)) {
             printf("[fcm_mpi] iter %4d  delta=%.8f  time(avg)=%.4fs  comp(avg)=%.4fs  comm(avg)=%.4fs  imbalance(max/avg)=%.3f\n",
                 iter, delta, t_avg, m->iter_compute_times[iter - 1],
                 m->iter_comm_times[iter - 1], imbalance);
        }

        if (delta < FCM_EPSILON) {
            m->iterations = iter;
            m->final_delta = delta;
            if (m->rank == 0) {
                printf("[fcm_mpi] Converged at iter %d (delta=%.2e)\n", iter, delta);
            }
            fcm_mpi_update_membership(m);
            return;
        }
    }

    m->iterations = FCM_MAX_ITER;
    m->final_delta = fcm_mpi_convergence(m);
    if (m->rank == 0) {
        printf("[fcm_mpi] Reached MAX_ITER=%d\n", FCM_MAX_ITER);
    }
}

static int load_feature_names(const char *path, int F, char names[][128]) {
    FILE *fp = fopen(path, "r");
    if (!fp) return 0;

    char line[8192];
    int count = 0;
    int first_row = 1;
    while (fgets(line, sizeof(line), fp) && count < F) {
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') continue;

        char tmp[8192];
        strncpy(tmp, line, sizeof(tmp) - 1);
        tmp[sizeof(tmp) - 1] = '\0';

        char *col1 = strtok(tmp, ",");
        char *col2 = strtok(NULL, "\r\n");

        if (first_row) {
            first_row = 0;
            if (col1 && col2 && strcmp(col1, "feature_index") == 0 && strcmp(col2, "feature_name") == 0) {
                continue;
            }
        }

        const char *src = NULL;
        if (col2 && col2[0] != '\0') src = col2;
        else if (col1 && col1[0] != '\0') src = col1;
        else continue;

        strncpy(names[count], src, 127);
        names[count][127] = '\0';
        count++;
    }
    fclose(fp);
    return count;
}

static void write_membership_heatmap_sample(const FCMMpiModel *m, const char *path, int sample_n) {
    FILE *fp = fopen(path, "w");
    if (!fp) return;

    fprintf(fp, "doc_id");
    for (int c = 0; c < m->C; c++) fprintf(fp, ",cluster_%d", c);
    fprintf(fp, "\n");

    int lim = (sample_n < m->N) ? sample_n : m->N;
    for (int i = 0; i < lim; i++) {
        fprintf(fp, "%d", i);
        for (int c = 0; c < m->C; c++) {
            fprintf(fp, ",%.12f", m->all_U[i * m->C + c]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

static void write_top_terms(const FCMMpiModel *m, const char *path, const char *feature_names_path, int top_k) {
    char names[N_FEATURES][128];
    int names_loaded = load_feature_names(feature_names_path, m->F, names);

    FILE *fp = fopen(path, "w");
    if (!fp) return;
    fprintf(fp, "cluster,rank,feature_index,feature_name,weight\n");

    for (int c = 0; c < m->C; c++) {
        int used[32];
        int cap = (top_k < 32) ? top_k : 32;
        for (int i = 0; i < cap; i++) used[i] = -1;

        for (int r = 0; r < cap; r++) {
            int best_f = -1;
            double best_w = -DBL_MAX;
            for (int f = 0; f < m->F; f++) {
                int already = 0;
                for (int u = 0; u < r; u++) {
                    if (used[u] == f) {
                        already = 1;
                        break;
                    }
                }
                if (already) continue;

                double w = m->centroids[c * m->F + f];
                if (w > best_w) {
                    best_w = w;
                    best_f = f;
                }
            }

            if (best_f < 0) break;
            used[r] = best_f;
            if (names_loaded > best_f) {
                fprintf(fp, "%d,%d,%d,%s,%.6f\n", c, r + 1, best_f, names[best_f], best_w);
            } else {
                fprintf(fp, "%d,%d,%d,f_%d,%.6f\n", c, r + 1, best_f, best_f, best_w);
            }
        }
    }

    fclose(fp);
}

static void write_label_cluster_comparison(const FCMMpiModel *m,
                                           const char *labels_path,
                                           const char *out_path) {
    FILE *fp = fopen(labels_path, "r");
    if (!fp) return;

    char label_names[128][256];
    int label_ids[TOTAL_DOCS];
    int label_count = 0;
    char line[1024];

    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return;
    }

    for (int i = 0; i < m->N; i++) {
        if (!fgets(line, sizeof(line), fp)) {
            label_ids[i] = 0;
            continue;
        }
        line[strcspn(line, "\r\n")] = '\0';
        char *s = line;
        while (*s == ' ' || *s == '\t') s++;

        int id = -1;
        for (int k = 0; k < label_count; k++) {
            if (strcmp(label_names[k], s) == 0) {
                id = k;
                break;
            }
        }
        if (id == -1) {
            if (label_count < 128) {
                strncpy(label_names[label_count], s, 255);
                label_names[label_count][255] = '\0';
                id = label_count;
                label_count++;
            } else {
                id = 127;
            }
        }
        label_ids[i] = id;
    }
    fclose(fp);

    int *matrix = (int *)calloc((size_t)label_count * m->C, sizeof(int));
    if (!matrix) return;

    for (int i = 0; i < m->N; i++) {
        int pred = 0;
        for (int c = 1; c < m->C; c++) {
            if (m->all_U[i * m->C + c] > m->all_U[i * m->C + pred]) pred = c;
        }
        int lid = label_ids[i];
        if (lid >= 0 && lid < label_count) {
            matrix[lid * m->C + pred]++;
        }
    }

    FILE *fo = fopen(out_path, "w");
    if (!fo) {
        free(matrix);
        return;
    }

    fprintf(fo, "label");
    for (int c = 0; c < m->C; c++) fprintf(fo, ",cluster_%d", c);
    fprintf(fo, ",dominant_cluster,dominant_ratio\n");

    for (int lid = 0; lid < label_count; lid++) {
        int total = 0;
        int best_c = 0;
        int best_v = -1;
        for (int c = 0; c < m->C; c++) {
            int v = matrix[lid * m->C + c];
            total += v;
            if (v > best_v) {
                best_v = v;
                best_c = c;
            }
        }

        fprintf(fo, "%s", label_names[lid]);
        for (int c = 0; c < m->C; c++) {
            fprintf(fo, ",%d", matrix[lid * m->C + c]);
        }
        fprintf(fo, ",%d,%.6f\n", best_c, total > 0 ? (double)best_v / total : 0.0);
    }

    fclose(fo);
    free(matrix);
}

void fcm_mpi_gather_and_save(FCMMpiModel *m,
                             const char *membership_path,
                             const char *centroids_path,
                             const char *labels_path,
                             const char *feature_names_path) {
    DistMeta meta_u = build_dist_meta(m, m->C);
    double *packed_U = (m->rank == 0) ? alloc_flat(m->N, m->C) : NULL;

    MPI_Gatherv(m->local_U,
                m->local_n * m->C,
                MPI_DOUBLE,
                packed_U,
                meta_u.counts_items,
                meta_u.displs_items,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    int *packed_ids = (m->rank == 0 && m->dist == DIST_DYNAMIC)
        ? (int *)malloc((size_t)m->N * sizeof(int))
        : NULL;
    if (m->dist == DIST_DYNAMIC) {
        DistMeta meta_ids = build_dist_meta(m, 1);
        MPI_Gatherv(m->local_ids,
                    m->local_n,
                    MPI_INT,
                    packed_ids,
                    meta_ids.counts_rows,
                    meta_ids.displs_rows,
                    MPI_INT,
                    0,
                    MPI_COMM_WORLD);
        free_dist_meta(&meta_ids);
    }

    if (m->rank == 0) {
        if (m->dist == DIST_DYNAMIC) {
            memset(m->all_U, 0, (size_t)m->N * m->C * sizeof(double));
            for (int i = 0; i < m->N; i++) {
                int gid = packed_ids[i];
                if (gid < 0 || gid >= m->N) continue;
                memcpy(m->all_U + (size_t)gid * m->C,
                       packed_U + (size_t)i * m->C,
                       (size_t)m->C * sizeof(double));
            }
        } else {
            unpack_rows_to_global(m, packed_U, m->all_U, m->C, &meta_u);
        }

        FILE *fm = fopen(membership_path, "w");
        if (fm) {
            for (int i = 0; i < m->N; i++) {
                for (int c = 0; c < m->C; c++) {
                    fprintf(fm, "%.12f", m->all_U[i * m->C + c]);
                    if (c < m->C - 1) fputc(',', fm);
                }
                fputc('\n', fm);
            }
            fclose(fm);
            printf("[fcm_mpi] Membership saved -> %s\n", membership_path);
        }

        FILE *fc = fopen(centroids_path, "w");
        if (fc) {
            for (int c = 0; c < m->C; c++) {
                for (int f = 0; f < m->F; f++) {
                    fprintf(fc, "%.6f", m->centroids[c * m->F + f]);
                    if (f < m->F - 1) fputc(',', fc);
                }
                fputc('\n', fc);
            }
            fclose(fc);
            printf("[fcm_mpi] Centroids saved -> %s\n", centroids_path);
        }

        write_membership_heatmap_sample(m, "viz_membership_sample.csv", 120);
        write_top_terms(m, "viz_top_terms.csv", feature_names_path, 10);
        write_label_cluster_comparison(m, labels_path, "viz_label_cluster_comparison.csv");
        printf("[member4] Visualization artifacts saved -> viz_membership_sample.csv, viz_top_terms.csv, viz_label_cluster_comparison.csv\n");

        free(packed_U);
        free(packed_ids);
    }

    free_dist_meta(&meta_u);
}

void fcm_mpi_print_summary(FCMMpiModel *m) {
    if (m->rank != 0) return;

    int *counts = (int *)calloc((size_t)m->C, sizeof(int));
    for (int i = 0; i < m->N; i++) {
        int best = 0;
        for (int c = 1; c < m->C; c++) {
            if (m->all_U[i * m->C + c] > m->all_U[i * m->C + best]) best = c;
        }
        counts[best]++;
    }

    printf("\n[fcm_mpi] -- Summary -------------------------------\n");
    printf("[fcm_mpi] Iterations : %d\n", m->iterations);
    printf("[fcm_mpi] Final delta: %.2e\n", m->final_delta);
    printf("[fcm_mpi] Cluster distribution:\n");
    for (int c = 0; c < m->C; c++) {
        printf("  Cluster %2d : %4d documents (%.1f%%)\n",
               c, counts[c], 100.0 * counts[c] / m->N);
    }

    free(counts);
}

void fcm_mpi_print_timing(FCMMpiModel *m) {
    if (m->rank != 0) return;

    double total = 0.0;
    double total_compute = 0.0;
    double total_comm = 0.0;
    double max_imbalance = 0.0;
    for (int i = 0; i < m->iterations; i++) {
        total += m->iter_times[i];
        total_compute += m->iter_compute_times[i];
        total_comm += m->iter_comm_times[i];
        if (m->iter_imbalance[i] > max_imbalance) max_imbalance = m->iter_imbalance[i];
    }

    printf("[fcm_mpi] Total wall time : %.4f s\n", total);
    printf("[fcm_mpi] Avg time/iter   : %.4f s\n", total / m->iterations);
    printf("[member4] Total compute time (avg across ranks): %.4f s (%.1f%%)\n",
           total_compute, (total > 0.0) ? (100.0 * total_compute / total) : 0.0);
    printf("[member4] Total communication time (avg across ranks): %.4f s (%.1f%%)\n",
           total_comm, (total > 0.0) ? (100.0 * total_comm / total) : 0.0);
    printf("[member4] Worst iter imbalance (max/avg): %.3f\n", max_imbalance);
}
