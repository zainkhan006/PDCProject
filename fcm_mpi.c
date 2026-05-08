/*
 * fcm_mpi.c  —  Parallel Fuzzy C-Means for Clinical Notes
 * PDC Project 21, IBA Karachi, Spring 2026
 *
 * Members:
 *   Khansa Danish  — Core algorithm & MPI parallelisation
 *   Arham Jumshaid — Data distribution, load balancing, visualisation
 */

#define _POSIX_C_SOURCE 200809L  /* enables strdup */
#include "fcm_mpi.h"

/* ═══════════════════════════════════════════════════════════════════════
 * INTERNAL HELPERS
 * ═══════════════════════════════════════════════════════════════════════ */

static void copy_flat(double *dst, const double *src, int n)
{
    memcpy(dst, src, n * sizeof(double));
}

static double *alloc_zeros(int n)
{
    double *p = calloc(n, sizeof(double));
    if (!p) { fprintf(stderr, "OOM: alloc_zeros(%d)\n", n); MPI_Abort(MPI_COMM_WORLD, 1); }
    return p;
}

static int *alloc_int_zeros(int n)
{
    int *p = calloc(n, sizeof(int));
    if (!p) { fprintf(stderr, "OOM: alloc_int_zeros(%d)\n", n); MPI_Abort(MPI_COMM_WORLD, 1); }
    return p;
}

/* ─── Row-count helpers for cyclic distribution ─────────────────────── */
static int cyclic_count(int N, int P, int rank)
{
    if (rank >= N) return 0;
    return (N - 1 - rank) / P + 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════ */

FCMMpiModel *fcm_mpi_create(int N, int F, int C,
                             DistStrategy dist, CommStrategy comm)
{
    FCMMpiModel *m = calloc(1, sizeof(FCMMpiModel));
    if (!m) { fprintf(stderr, "OOM: FCMMpiModel\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    MPI_Comm_rank(MPI_COMM_WORLD, &m->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &m->n_procs);

    m->N = N; m->F = F; m->C = C;
    m->dist_mode = dist;
    m->comm_mode = comm;

    /* ── Compute local row count ──────────────────────────────────────── */
    if (dist == DIST_CYCLIC) {
        m->local_n     = cyclic_count(N, m->n_procs, m->rank);
        m->local_start = m->rank;              /* first global index      */
    } else {
        /* Block (also initial state for Dynamic) */
        int base  = N / m->n_procs;
        int extra = N % m->n_procs;
        m->local_n     = base + (m->rank < extra ? 1 : 0);
        m->local_start = m->rank * base + (m->rank < extra ? m->rank : extra);
    }

    /* ── Row-ID array (which global doc each local row corresponds to) ── */
    m->local_ids = alloc_int_zeros(m->local_n);
    for (int i = 0; i < m->local_n; i++) {
        if (dist == DIST_CYCLIC)
            m->local_ids[i] = m->rank + i * m->n_procs;
        else
            m->local_ids[i] = m->local_start + i;
    }

    /* ── Allocate local data buffers ────────────────────────────────────*/
    m->local_data  = alloc_zeros(m->local_n * F);
    m->local_U     = alloc_zeros(m->local_n * C);
    m->local_U_old = alloc_zeros(m->local_n * C);
    m->centroids   = alloc_zeros(C * F);

    /* ── Print balance info (rank 0 only) ───────────────────────────────*/
    int *all_counts = NULL;
    if (m->rank == 0) all_counts = alloc_int_zeros(m->n_procs);
    MPI_Gather(&m->local_n, 1, MPI_INT,
               all_counts,  1, MPI_INT, 0, MPI_COMM_WORLD);
    if (m->rank == 0) {
        int mn = all_counts[0], mx = all_counts[0];
        double avg = 0.0;
        for (int r = 0; r < m->n_procs; r++) {
            if (all_counts[r] < mn) mn = all_counts[r];
            if (all_counts[r] > mx) mx = all_counts[r];
            avg += all_counts[r];
        }
        avg /= m->n_procs;
        printf("[member4] Row balance: min=%d max=%d avg=%.2f max/avg=%.3f\n",
               mn, mx, avg, mx / avg);
        free(all_counts);
    }

    return m;
}

void fcm_mpi_free(FCMMpiModel *m)
{
    if (!m) return;
    free(m->local_data);
    free(m->local_U);
    free(m->local_U_old);
    free(m->centroids);
    free(m->local_ids);
    free(m);
}

/* ═══════════════════════════════════════════════════════════════════════
 * DATA LOADING AND SCATTERING
 * ═══════════════════════════════════════════════════════════════════════ */

int fcm_mpi_load_and_scatter(FCMMpiModel *m,
                              const char *feat_path,
                              const char *label_path,
                              int **domain_labels_out)
{
    int N = m->N, F = m->F, P = m->n_procs;
    double *full_data = NULL;
    int    *full_labels = NULL;

    /* ── Rank 0 reads the full feature matrix ───────────────────────── */
    if (m->rank == 0) {
        full_data = alloc_zeros(N * F);
        FILE *fp = fopen(feat_path, "r");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", feat_path); return -1; }
        for (int i = 0; i < N; i++)
            for (int f = 0; f < F; f++) {
                if (fscanf(fp, "%lf,", &full_data[i*F + f]) != 1) {
                    /* try no-comma at end of row */
                    fscanf(fp, "%lf", &full_data[i*F + f]);
                }
            }
        fclose(fp);
        printf("[rank 0] Loaded %d x %d feature matrix from '%s'\n", N, F, feat_path);
    }

    /* ── Rank 0 reads specialty labels ─────────────────────────────── */
    if (m->rank == 0 && label_path) {
        full_labels = alloc_int_zeros(N);
        FILE *fp = fopen(label_path, "r");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", label_path); return -1; }
        char line[256];
        /* skip header */
        fgets(line, sizeof(line), fp);
        for (int i = 0; i < N; i++) {
            if (!fgets(line, sizeof(line), fp)) break;
            /* hash specialty string to cluster index */
            unsigned int h = 5381;
            for (char *c = line; *c && *c != '\n'; c++)
                h = h * 31 + (unsigned char)*c;
            full_labels[i] = (int)(h % m->C);
        }
        fclose(fp);
        *domain_labels_out = full_labels;
    } else {
        *domain_labels_out = NULL;
    }

    /* ── Build scatter counts and displacements ─────────────────────── */
    int *counts = alloc_int_zeros(P);
    int *displs = alloc_int_zeros(P);

    if (m->dist_mode == DIST_CYCLIC) {
        /* For cyclic: rank 0 packs each rank's rows from full_data */
        /* We gather local_n from all ranks first */
        MPI_Allgather(&m->local_n, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
        displs[0] = 0;
        for (int r = 1; r < P; r++) displs[r] = displs[r-1] + counts[r-1];

        /* Re-pack full_data into cyclic order on rank 0 */
        if (m->rank == 0) {
            double *packed = alloc_zeros(N * F);
            int *pos = alloc_int_zeros(P);   /* write cursor per rank */
            displs[0] = 0;
            for (int r = 1; r < P; r++) displs[r] = displs[r-1] + counts[r-1] * F;
            for (int i = 0; i < N; i++) {
                int r = i % P;
                int dst = displs[r] + pos[r] * F;
                memcpy(packed + dst, full_data + i*F, F * sizeof(double));
                pos[r]++;
            }
            free(full_data);
            full_data = packed;
            free(pos);
        }
        /* Fix counts to be in doubles for Scatterv */
        for (int r = 0; r < P; r++) counts[r] *= F;
        if (m->rank != 0) {
            displs[0] = 0;   /* recalc on non-root (filled by Allgather above but needs F-scaling) */
        }
        /* Recalc displs in doubles */
        if (m->rank == 0) { /* already done above for packed */ }
        else {
            /* root displs already correct; non-roots only use their own slice */
        }
        MPI_Scatterv(full_data, counts, displs,
                     MPI_DOUBLE,
                     m->local_data, m->local_n * F, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
    } else {
        /* Block distribution */
        int base  = N / P;
        int extra = N % P;
        for (int r = 0; r < P; r++) {
            counts[r] = (base + (r < extra ? 1 : 0)) * F;
            displs[r] = (r * base + (r < extra ? r : extra)) * F;
        }
        MPI_Scatterv(full_data, counts, displs,
                     MPI_DOUBLE,
                     m->local_data, m->local_n * F, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
    }

    if (m->rank == 0) free(full_data);
    free(counts);
    free(displs);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * INITIALISATION STRATEGIES
 * ═══════════════════════════════════════════════════════════════════════ */

/* --- Random: rank 0 picks C distinct data points as centroids --------- */
static void init_random(FCMMpiModel *m)
{
    int C = m->C, F = m->F, N = m->N;

    if (m->rank == 0) {
        /* Gather all data to rank 0 */
        double *all_data = alloc_zeros(N * F);

        int *counts = alloc_int_zeros(m->n_procs);
        int *displs = alloc_int_zeros(m->n_procs);
        MPI_Gather(&m->local_n, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
        displs[0] = 0;
        for (int r = 1; r < m->n_procs; r++)
            displs[r] = displs[r-1] + counts[r-1];
        for (int r = 0; r < m->n_procs; r++) counts[r] *= F;
        for (int r = 0; r < m->n_procs; r++) displs[r] *= F;

        MPI_Gatherv(m->local_data, m->local_n * F, MPI_DOUBLE,
                    all_data, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* Pick C distinct random row indices */
        int *chosen = alloc_int_zeros(C);
        srand((unsigned)time(NULL) + 42);
        for (int j = 0; j < C; j++) {
            int idx;
            int ok;
            do {
                ok = 1;
                idx = rand() % N;
                for (int k = 0; k < j; k++)
                    if (chosen[k] == idx) { ok = 0; break; }
            } while (!ok);
            chosen[j] = idx;
            memcpy(m->centroids + j*F, all_data + idx*F, F * sizeof(double));
        }

        free(all_data); free(chosen); free(counts); free(displs);
    } else {
        /* Non-root: still participate in Gatherv */
        int dummy_n = 0;
        MPI_Gather(&m->local_n, 1, MPI_INT, NULL, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(m->local_data, m->local_n * F, MPI_DOUBLE,
                    NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        (void)dummy_n;
    }

    /* Broadcast chosen centroids */
    MPI_Bcast(m->centroids, C * F, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Each rank initialises its local U with soft assignments */
    for (int i = 0; i < m->local_n; i++) {
        double *xi = m->local_data + i * F;
        int nearest = 0;
        double best = l2_distance(xi, m->centroids, F);
        for (int j = 1; j < C; j++) {
            double d = l2_distance(xi, m->centroids + j*F, F);
            if (d < best) { best = d; nearest = j; }
        }
        for (int j = 0; j < C; j++)
            m->local_U[i*C + j] = (j == nearest) ? 0.7 : (0.3 / (C-1));
    }
    if (m->rank == 0) printf("[init] Random init complete.\n");
}

/* --- K-Means++: distance-proportional seeding ------------------------- */
static void init_kmeanspp(FCMMpiModel *m)
{
    int C = m->C, F = m->F, N = m->N, P = m->n_procs;

    /* Gather all data to rank 0 for seeding */
    double *all_data = NULL;
    int *counts = alloc_int_zeros(P);
    int *displs = alloc_int_zeros(P);

    MPI_Gather(&m->local_n, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (m->rank == 0) {
        all_data = alloc_zeros(N * F);
        displs[0] = 0;
        for (int r = 1; r < P; r++)
            displs[r] = displs[r-1] + counts[r-1];
        int *cnt_f = alloc_int_zeros(P);
        int *dsp_f = alloc_int_zeros(P);
        for (int r = 0; r < P; r++) { cnt_f[r] = counts[r]*F; dsp_f[r] = displs[r]*F; }

        MPI_Gatherv(m->local_data, m->local_n*F, MPI_DOUBLE,
                    all_data, cnt_f, dsp_f, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* K-Means++ seeding */
        srand((unsigned)time(NULL) + 7);
        int first = rand() % N;
        memcpy(m->centroids, all_data + first*F, F * sizeof(double));

        double *dist2 = malloc(N * sizeof(double));
        for (int j = 1; j < C; j++) {
            /* compute min distance squared to any chosen centroid */
            double total = 0.0;
            for (int i = 0; i < N; i++) {
                double best = 1e300;
                for (int k = 0; k < j; k++) {
                    double d = l2_distance(all_data + i*F, m->centroids + k*F, F);
                    if (d < best) best = d;
                }
                dist2[i] = best * best;
                total += dist2[i];
            }
            /* sample proportional to dist2 */
            double r = ((double)rand() / RAND_MAX) * total;
            double acc = 0.0;
            int chosen = N - 1;
            for (int i = 0; i < N; i++) {
                acc += dist2[i];
                if (acc >= r) { chosen = i; break; }
            }
            memcpy(m->centroids + j*F, all_data + chosen*F, F * sizeof(double));
        }

        free(dist2); free(all_data); free(cnt_f); free(dsp_f);
    } else {
        MPI_Gatherv(m->local_data, m->local_n*F, MPI_DOUBLE,
                    NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    free(counts); free(displs);

    /* Broadcast centroids */
    MPI_Bcast(m->centroids, C * F, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Soft initialise U: nearest centroid gets 0.7 */
    for (int i = 0; i < m->local_n; i++) {
        double *xi = m->local_data + i * F;
        int nearest = 0;
        double best = l2_distance(xi, m->centroids, F);
        for (int j = 1; j < C; j++) {
            double d = l2_distance(xi, m->centroids + j*F, F);
            if (d < best) { best = d; nearest = j; }
        }
        for (int j = 0; j < C; j++)
            m->local_U[i*C + j] = (j == nearest) ? 0.7 : (0.3 / (C-1));
    }
    if (m->rank == 0) printf("[init] K-Means++ init complete.\n");
}

/* --- Domain-guided: class-mean centroids from specialty labels --------- */
static void init_domain(FCMMpiModel *m, int *domain_labels)
{
    int C = m->C, F = m->F, N = m->N, P = m->n_procs;

    /* Broadcast labels from rank 0 */
    int *labels = alloc_int_zeros(N);
    if (m->rank == 0 && domain_labels)
        memcpy(labels, domain_labels, N * sizeof(int));
    MPI_Bcast(labels, N, MPI_INT, 0, MPI_COMM_WORLD);

    /* Gather all data to rank 0 */
    double *all_data = NULL;
    int *counts = alloc_int_zeros(P);
    int *displs = alloc_int_zeros(P);
    MPI_Gather(&m->local_n, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (m->rank == 0) {
        all_data = alloc_zeros(N * F);
        displs[0] = 0;
        for (int r = 1; r < P; r++) displs[r] = displs[r-1] + counts[r-1];
        int *cnt_f = alloc_int_zeros(P);
        int *dsp_f = alloc_int_zeros(P);
        for (int r = 0; r < P; r++) { cnt_f[r] = counts[r]*F; dsp_f[r] = displs[r]*F; }
        MPI_Gatherv(m->local_data, m->local_n*F, MPI_DOUBLE,
                    all_data, cnt_f, dsp_f, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* Compute class-mean centroids */
        double *sums = alloc_zeros(C * F);
        int    *cnt  = alloc_int_zeros(C);
        for (int i = 0; i < N; i++) {
            int j = labels[i] % C;
            cnt[j]++;
            for (int f = 0; f < F; f++)
                sums[j*F + f] += all_data[i*F + f];
        }
        for (int j = 0; j < C; j++) {
            double n = cnt[j] > 0 ? cnt[j] : 1.0;
            for (int f = 0; f < F; f++)
                m->centroids[j*F + f] = sums[j*F + f] / n;
        }
        free(sums); free(cnt); free(all_data); free(cnt_f); free(dsp_f);
    } else {
        MPI_Gatherv(m->local_data, m->local_n*F, MPI_DOUBLE,
                    NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    free(counts); free(displs);
    MPI_Bcast(m->centroids, C * F, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Soft U based on label */
    for (int i = 0; i < m->local_n; i++) {
        int gidx = m->local_ids[i];
        int dom  = labels[gidx] % C;
        for (int j = 0; j < C; j++)
            m->local_U[i*C + j] = (j == dom) ? 0.8 : (0.2 / (C-1));
    }
    free(labels);
    if (m->rank == 0) printf("[init] Domain-guided init complete.\n");
}

/* ═══════════════════════════════════════════════════════════════════════
 * E-STEP  (embarrassingly parallel — zero communication)
 * ═══════════════════════════════════════════════════════════════════════ */
void fcm_mpi_update_membership(FCMMpiModel *m)
{
    int ln = m->local_n, F = m->F, C = m->C;
    double exp_ = 2.0 / (FCM_M - 1.0);
    double *dist = malloc(C * sizeof(double));

    for (int i = 0; i < ln; i++) {
        double *xi = m->local_data + i * F;

        /* compute distances to all centroids */
        int zero_c = -1;
        for (int j = 0; j < C; j++) {
            dist[j] = l2_distance(xi, m->centroids + j*F, F);
            if (dist[j] < FCM_MIN_DIST) zero_c = j;
        }

        if (zero_c >= 0) {
            /* point coincides with centroid j → membership = 1 there */
            for (int j = 0; j < C; j++)
                m->local_U[i*C + j] = (j == zero_c) ? 1.0 : 0.0;
            continue;
        }

        for (int j = 0; j < C; j++) {
            double sum = 0.0;
            for (int k = 0; k < C; k++)
                sum += pow(dist[j] / dist[k], exp_);
            m->local_U[i*C + j] = 1.0 / sum;
        }
    }
    free(dist);
    /* Zero MPI calls — embarrassingly parallel */
}

/* ═══════════════════════════════════════════════════════════════════════
 * M-STEP  (two MPI_Allreduce calls)
 * ═══════════════════════════════════════════════════════════════════════ */
void fcm_mpi_update_centroids(FCMMpiModel *m)
{
    int ln = m->local_n, F = m->F, C = m->C;

    double *local_num  = alloc_zeros(C * F);
    double *global_num = alloc_zeros(C * F);
    double *local_den  = alloc_zeros(C);
    double *global_den = alloc_zeros(C);

    /* Partial sums over local rows */
    for (int i = 0; i < ln; i++) {
        for (int j = 0; j < C; j++) {
            double u_m = pow(m->local_U[i*C + j], FCM_M);
            local_den[j] += u_m;
            for (int f = 0; f < F; f++)
                local_num[j*F + f] += u_m * m->local_data[i*F + f];
        }
    }

    /* Global reduction */
    if (m->comm_mode == COMM_NONBLOCK) {
        /* Issue both reductions concurrently — latencies overlap */
        MPI_Request reqs[2];
        MPI_Iallreduce(local_num, global_num, C*F,
                       MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &reqs[0]);
        MPI_Iallreduce(local_den, global_den, C,
                       MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &reqs[1]);
        /* ── Overlap: compute convergence norm while transfers are in flight ── */
        /* (convergence is computed separately in fcm_mpi_convergence,          */
        /*  but any rank-local arithmetic could go here in a more advanced impl) */
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    } else {
        /* Baseline: non-blocking numerator, blocking denominator, then wait */
        MPI_Request req;
        MPI_Iallreduce(local_num, global_num, C*F,
                       MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &req);
        MPI_Allreduce(local_den, global_den, C,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }

    /* Update centroids (all ranks identical after Allreduce) */
    for (int j = 0; j < C; j++) {
        double denom = global_den[j] > FCM_MIN_DIST ? global_den[j] : FCM_MIN_DIST;
        for (int f = 0; f < F; f++)
            m->centroids[j*F + f] = global_num[j*F + f] / denom;
    }

    free(local_num); free(global_num); free(local_den); free(global_den);
}

/* ═══════════════════════════════════════════════════════════════════════
 * CONVERGENCE DETECTION  (one MPI_Allreduce)
 * ═══════════════════════════════════════════════════════════════════════ */
double fcm_mpi_convergence(FCMMpiModel *m)
{
    double local_norm2 = 0.0;
    int ln = m->local_n, C = m->C;
    for (int i = 0; i < ln; i++)
        for (int j = 0; j < C; j++) {
            double d = m->local_U[i*C+j] - m->local_U_old[i*C+j];
            local_norm2 += d * d;
        }

    double global_norm2 = 0.0;
    MPI_Allreduce(&local_norm2, &global_norm2, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(global_norm2);  /* identical on all ranks */
}

/* ═══════════════════════════════════════════════════════════════════════
 * DYNAMIC LOAD REBALANCING
 * ═══════════════════════════════════════════════════════════════════════ */
static void dynamic_rebalance(FCMMpiModel *m, double iter_time)
{
    int P = m->n_procs, N = m->N, F = m->F, C = m->C;

    double *all_times  = (m->rank == 0) ? malloc(P * sizeof(double)) : NULL;
    int    *all_counts = (m->rank == 0) ? malloc(P * sizeof(int))    : NULL;

    MPI_Gather(&iter_time,  1, MPI_DOUBLE, all_times,  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&m->local_n, 1, MPI_INT,   all_counts, 1, MPI_INT,    0, MPI_COMM_WORLD);

    /* Check if rebalancing is needed */
    int do_rebalance = 0;
    if (m->rank == 0) {
        double avg = 0.0, mx = 0.0;
        for (int r = 0; r < P; r++) {
            if (all_times[r] > mx) mx = all_times[r];
            avg += all_times[r];
        }
        avg /= P;
        if (mx / avg > DYNAMIC_IMBALANCE_THR) do_rebalance = 1;
    }
    MPI_Bcast(&do_rebalance, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!do_rebalance) {
        if (m->rank == 0) { free(all_times); free(all_counts); }
        return;
    }

    /* Gather all data to rank 0 */
    int *displs = alloc_int_zeros(P);
    int *cnt_f  = alloc_int_zeros(P);
    MPI_Gather(&m->local_n, 1, MPI_INT, all_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double *all_data = NULL;
    double *all_U    = NULL;
    if (m->rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < P; r++) displs[r] = displs[r-1] + all_counts[r-1];
        for (int r = 0; r < P; r++) cnt_f[r] = all_counts[r] * F;
        int *dsp_f = alloc_int_zeros(P);
        for (int r = 0; r < P; r++) dsp_f[r] = displs[r] * F;

        all_data = alloc_zeros(N * F);
        all_U    = alloc_zeros(N * C);

        int *cnt_c = alloc_int_zeros(P);
        int *dsp_c = alloc_int_zeros(P);
        for (int r = 0; r < P; r++) { cnt_c[r] = all_counts[r]*C; dsp_c[r] = displs[r]*C; }

        MPI_Gatherv(m->local_data, m->local_n*F, MPI_DOUBLE,
                    all_data, cnt_f, dsp_f, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(m->local_U, m->local_n*C, MPI_DOUBLE,
                    all_U, cnt_c, dsp_c, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        free(dsp_f); free(cnt_c); free(dsp_c);
    } else {
        MPI_Gatherv(m->local_data, m->local_n*F, MPI_DOUBLE,
                    NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(m->local_U, m->local_n*C, MPI_DOUBLE,
                    NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    /* Compute new distribution: throughput-proportional */
    int *new_counts = alloc_int_zeros(P);
    if (m->rank == 0) {
        double *tput = malloc(P * sizeof(double));
        double total_t = 0.0;
        for (int r = 0; r < P; r++) {
            tput[r] = all_counts[r] / (all_times[r] > 1e-12 ? all_times[r] : 1e-12);
            total_t += tput[r];
        }
        int min_rows = 1;
        int remaining = N - min_rows * P;
        for (int r = 0; r < P; r++)
            new_counts[r] = min_rows + (int)(remaining * tput[r] / total_t);
        /* fix rounding */
        int used = 0;
        for (int r = 0; r < P; r++) used += new_counts[r];
        new_counts[P-1] += N - used;
        free(tput);
    }
    MPI_Bcast(new_counts, P, MPI_INT, 0, MPI_COMM_WORLD);

    /* Update local_n and reallocate if needed */
    int new_local_n = new_counts[m->rank];
    if (new_local_n != m->local_n) {
        m->local_data  = realloc(m->local_data,  new_local_n * F * sizeof(double));
        m->local_U     = realloc(m->local_U,     new_local_n * C * sizeof(double));
        m->local_U_old = realloc(m->local_U_old, new_local_n * C * sizeof(double));
        m->local_ids   = realloc(m->local_ids,   new_local_n * sizeof(int));
    }
    m->local_n = new_local_n;

    /* Scatter new data */
    if (m->rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < P; r++) displs[r] = displs[r-1] + new_counts[r-1];
    }
    int *new_cnt_f = alloc_int_zeros(P);
    int *new_dsp_f = alloc_int_zeros(P);
    int *new_cnt_c = alloc_int_zeros(P);
    int *new_dsp_c = alloc_int_zeros(P);
    for (int r = 0; r < P; r++) {
        new_cnt_f[r] = new_counts[r] * F;
        new_dsp_f[r] = (m->rank == 0) ? displs[r] * F : 0;
        new_cnt_c[r] = new_counts[r] * C;
        new_dsp_c[r] = (m->rank == 0) ? displs[r] * C : 0;
    }
    /* Need to broadcast displs for non-root ranks to compute dsp */
    MPI_Bcast(displs, P, MPI_INT, 0, MPI_COMM_WORLD);
    for (int r = 0; r < P; r++) {
        new_dsp_f[r] = displs[r] * F;
        new_dsp_c[r] = displs[r] * C;
    }

    MPI_Scatterv(all_data, new_cnt_f, new_dsp_f, MPI_DOUBLE,
                 m->local_data, m->local_n*F, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(all_U,    new_cnt_c, new_dsp_c, MPI_DOUBLE,
                 m->local_U,   m->local_n*C, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Update local_ids */
    int start = displs[m->rank];
    for (int i = 0; i < m->local_n; i++)
        m->local_ids[i] = start + i;
    m->local_start = start;

    if (m->rank == 0) {
        free(all_data); free(all_U);
        free(all_times); free(all_counts);
    }
    free(displs); free(new_counts);
    free(cnt_f); free(new_cnt_f); free(new_dsp_f);
    free(new_cnt_c); free(new_dsp_c);
}

/* ═══════════════════════════════════════════════════════════════════════
 * MAIN PARALLEL FCM LOOP
 * ═══════════════════════════════════════════════════════════════════════ */
void fcm_mpi_run(FCMMpiModel *m, InitStrategy strategy, int *domain_labels)
{
    int C = m->C;

    /* 1. Initialise */
    switch (strategy) {
        case INIT_KMEANSPP: init_kmeanspp(m);             break;
        case INIT_DOMAIN:   init_domain(m, domain_labels); break;
        default:            init_random(m);                break;
    }

    /* 2. Initial M-step to get centroids from U */
    fcm_mpi_update_centroids(m);

    if (m->rank == 0)
        printf("[fcm_mpi] N=%d  F=%d  C=%d  P=%d  m=%.1f  eps=%.0e\n",
               m->N, m->F, C, m->n_procs, FCM_M, FCM_EPSILON);

    /* 3. Main iteration loop */
    for (int iter = 1; iter <= FCM_MAX_ITER; iter++) {
        double t_start = MPI_Wtime();

        /* Save old U for convergence */
        copy_flat(m->local_U_old, m->local_U, m->local_n * C);

        /* E-step: embarrassingly parallel, zero comm */
        double t_comp0 = MPI_Wtime();
        fcm_mpi_update_membership(m);
        double t_comp1 = MPI_Wtime();

        /* M-step: 2 Allreduce calls */
        double t_comm0 = MPI_Wtime();
        fcm_mpi_update_centroids(m);
        double t_comm1 = MPI_Wtime();

        /* Convergence: 1 Allreduce */
        double delta = fcm_mpi_convergence(m);
        double t_end = MPI_Wtime();

        /* Per-rank timing → reduce to averages */
        double local_comp = (t_comp1 - t_comp0);
        double local_comm = (t_comm1 - t_comm0) + (t_end - t_comm1);
        double local_iter = t_end - t_start;

        double avg_comp = 0.0, avg_comm = 0.0, avg_iter = 0.0, max_iter = 0.0;
        MPI_Allreduce(&local_comp, &avg_comp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_comm, &avg_comm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_iter, &avg_iter, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_iter, &max_iter, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        avg_comp /= m->n_procs;
        avg_comm /= m->n_procs;
        avg_iter /= m->n_procs;

        m->iter_times[iter-1]    = avg_iter;
        m->iter_compute[iter-1]  = avg_comp;
        m->iter_comm[iter-1]     = avg_comm;
        m->iter_imbalance[iter-1]= max_iter / (avg_iter > 1e-12 ? avg_iter : 1e-12);

        if (m->rank == 0)
            printf("[fcm_mpi] iter %4d  delta=%.8f  time(avg)=%.4fs"
                   "  comp(avg)=%.4fs  comm(avg)=%.4fs  imbalance(max/avg)=%.3f\n",
                   iter, delta, avg_iter, avg_comp, avg_comm,
                   m->iter_imbalance[iter-1]);

        /* Dynamic rebalance check */
        if (m->dist_mode == DIST_DYNAMIC &&
            iter > DYNAMIC_WARMUP_ITERS &&
            iter % DYNAMIC_CHECK_INTERVAL == 0)
        {
            dynamic_rebalance(m, local_iter);
        }

        if (delta < FCM_EPSILON) {
            if (m->rank == 0)
                printf("[fcm_mpi] Converged at iter %d (delta=%.2e)\n", iter, delta);
            m->iterations   = iter;
            m->final_delta  = delta;
            /* Final E-step so saved memberships match final centroids */
            fcm_mpi_update_membership(m);
            return;
        }
    }

    if (m->rank == 0)
        printf("[fcm_mpi] Reached MAX_ITER=%d without convergence.\n", FCM_MAX_ITER);
    m->iterations  = FCM_MAX_ITER;
    /* Final E-step before gather */
    fcm_mpi_update_membership(m);
}

/* ═══════════════════════════════════════════════════════════════════════
 * OUTPUT: GATHER AND SAVE
 * ═══════════════════════════════════════════════════════════════════════ */
void fcm_mpi_gather_and_save(FCMMpiModel *m,
                              const char *mem_path,
                              const char *cen_path,
                              const char *feat_names_path)
{
    int N = m->N, F = m->F, C = m->C, P = m->n_procs;

    /* Build gather counts */
    int *counts = alloc_int_zeros(P);
    int *displs = alloc_int_zeros(P);
    MPI_Gather(&m->local_n, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double *full_U = NULL;
    if (m->rank == 0) {
        full_U    = alloc_zeros(N * C);
        displs[0] = 0;
        for (int r = 1; r < P; r++) displs[r] = displs[r-1] + counts[r-1];
        int *cnt_c = alloc_int_zeros(P);
        int *dsp_c = alloc_int_zeros(P);
        for (int r = 0; r < P; r++) { cnt_c[r] = counts[r]*C; dsp_c[r] = displs[r]*C; }
        MPI_Gatherv(m->local_U, m->local_n*C, MPI_DOUBLE,
                    full_U, cnt_c, dsp_c, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        free(cnt_c); free(dsp_c);
    } else {
        MPI_Gatherv(m->local_U, m->local_n*C, MPI_DOUBLE,
                    NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (m->rank != 0) { free(counts); free(displs); return; }

    /* ── Write membership CSV ─────────────────────────────────────────── */
    FILE *fp = fopen(mem_path, "w");
    if (!fp) { fprintf(stderr, "Cannot write %s\n", mem_path); }
    else {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < C; j++) {
                fprintf(fp, "%.12f", full_U[i*C + j]);
                if (j < C-1) fputc(',', fp);
            }
            fputc('\n', fp);
        }
        fclose(fp);
        printf("[fcm_mpi] Membership saved -> %s\n", mem_path);
    }

    /* ── Write centroids CSV ──────────────────────────────────────────── */
    fp = fopen(cen_path, "w");
    if (!fp) { fprintf(stderr, "Cannot write %s\n", cen_path); }
    else {
        for (int j = 0; j < C; j++) {
            for (int f = 0; f < F; f++) {
                fprintf(fp, "%.12f", m->centroids[j*F + f]);
                if (f < F-1) fputc(',', fp);
            }
            fputc('\n', fp);
        }
        fclose(fp);
        printf("[fcm_mpi] Centroids saved -> %s\n", cen_path);
    }

    /* ── Write viz_top_terms.csv ──────────────────────────────────────── */
    /* Load feature names if available */
    char **feat_names = NULL;
    if (feat_names_path) {
        feat_names = malloc(F * sizeof(char *));
        for (int f = 0; f < F; f++) feat_names[f] = NULL;
        FILE *fnp = fopen(feat_names_path, "r");
        if (fnp) {
            char line[512];
            fgets(line, sizeof(line), fnp); /* skip header */
            while (fgets(line, sizeof(line), fnp)) {
                int idx; char name[400];
                if (sscanf(line, "%d,%399s", &idx, name) == 2 && idx < F) {
                    feat_names[idx] = strdup(name);
                }
            }
            fclose(fnp);
        }
    }

    fp = fopen("viz_top_terms.csv", "w");
    if (fp) {
        fprintf(fp, "cluster,rank,feature_index,feature_name,weight\n");
        int TOP = 10;
        for (int j = 0; j < C; j++) {
            /* find top-10 features by centroid weight */
            int   *top_idx = malloc(TOP * sizeof(int));
            double *top_w  = malloc(TOP * sizeof(double));
            for (int t = 0; t < TOP; t++) { top_idx[t] = -1; top_w[t] = -1.0; }
            for (int f = 0; f < F; f++) {
                double w = m->centroids[j*F + f];
                /* Only insert if w is larger than the smallest stored value */
                if (w <= top_w[TOP-1]) continue;
                /* Find insertion position in descending order */
                int pos = TOP - 1;
                while (pos > 0 && w > top_w[pos-1]) pos--;
                /* Shift everything from pos..TOP-2 down one slot */
                for (int t = TOP-1; t > pos; t--) {
                    top_idx[t] = top_idx[t-1];
                    top_w[t]   = top_w[t-1];
                }
                top_idx[pos] = f;
                top_w[pos]   = w;
            }
            for (int t = 0; t < TOP; t++) {
                if (top_idx[t] < 0) continue;
                int fi = top_idx[t];
                const char *name = (feat_names && feat_names[fi])
                                   ? feat_names[fi] : "unknown";
                fprintf(fp, "%d,%d,%d,%s,%.6f\n", j, t+1, fi, name, top_w[t]);
            }
            free(top_idx); free(top_w);
        }
        fclose(fp);
        printf("[member4] viz_top_terms.csv saved.\n");
    }

    /* ── Write viz_membership_sample.csv (first 120 docs) ───────────── */
    fp = fopen("viz_membership_sample.csv", "w");
    if (fp) {
        int sample = N < 120 ? N : 120;
        fprintf(fp, "doc_id");
        for (int j = 0; j < C; j++) fprintf(fp, ",cluster_%d", j);
        fputc('\n', fp);
        for (int i = 0; i < sample; i++) {
            fprintf(fp, "%d", i);
            for (int j = 0; j < C; j++)
                fprintf(fp, ",%.12f", full_U[i*C + j]);
            fputc('\n', fp);
        }
        fclose(fp);
        printf("[member4] viz_membership_sample.csv saved.\n");
    }

    if (feat_names) {
        for (int f = 0; f < F; f++) if (feat_names[f]) free(feat_names[f]);
        free(feat_names);
    }
    free(full_U); free(counts); free(displs);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SUMMARY AND TIMING REPORT
 * ═══════════════════════════════════════════════════════════════════════ */
void fcm_mpi_print_summary(FCMMpiModel *m)
{
    if (m->rank != 0) return;

    /* Need full U to compute hard assignments */
    /* Summarise from already-saved file is fine; here we recompute from centroids */
    printf("\n[fcm_mpi] -- Summary -------------------------------\n");
    printf("[fcm_mpi] Iterations : %d\n", m->iterations);
    printf("[fcm_mpi] Final delta: %.2e\n", m->final_delta);

    /* Cluster sizes from local_U (gathered) — we print total from timing */
    double total_time = 0.0;
    for (int i = 0; i < m->iterations; i++) total_time += m->iter_times[i];
    double avg_iter = m->iterations > 0 ? total_time / m->iterations : 0.0;
    printf("[fcm_mpi] Total wall time : %.4f s\n", total_time);
    printf("[fcm_mpi] Avg time/iter   : %.4f s\n", avg_iter);
}

void fcm_mpi_print_timing(FCMMpiModel *m)
{
    if (m->rank != 0) return;

    double tot_comp = 0.0, tot_comm = 0.0, worst_imb = 1.0;
    for (int i = 0; i < m->iterations; i++) {
        tot_comp += m->iter_compute[i];
        tot_comm += m->iter_comm[i];
        if (m->iter_imbalance[i] > worst_imb) worst_imb = m->iter_imbalance[i];
    }
    double total = tot_comp + tot_comm;
    if (total < 1e-12) total = 1e-12;
    printf("[member4] Total compute time (avg across ranks): %.4f s (%.1f%%)\n",
           tot_comp, 100.0 * tot_comp / total);
    printf("[member4] Total communication time (avg across ranks): %.4f s (%.1f%%)\n",
           tot_comm, 100.0 * tot_comm / total);
    printf("[member4] Worst iter imbalance (max/avg): %.3f\n", worst_imb);
}