/* ============================================================
 * fcm_mpi.c  —  Parallel Fuzzy C-Means (OpenMPI)
 *
 * Milestone 2 — Member 1: Khansa Danish | IBA Karachi, Spring 2026
 *
 * ── PARALLEL DESIGN ──────────────────────────────────────────
 *
 *  Data distribution : MPI_Scatterv, block partition of N docs.
 *
 *  E-step (membership update):
 *    Each rank computes local_U for its own rows independently.
 *    No communication needed — embarrassingly parallel.
 *
 *  M-step (centroid update):
 *    Each rank accumulates local partial sums (numerator & denominator).
 *    MPI_Iallreduce (non-blocking) on numerator [C*F doubles] is
 *    issued first; MPI_Allreduce on denominator [C doubles] runs
 *    concurrently so network latency overlaps.  MPI_Wait syncs after.
 *    All ranks compute identical centroids — no extra broadcast needed.
 *
 *  Convergence detection:
 *    Each rank computes its local Frobenius norm².
 *    One MPI_Allreduce(SUM) -> global norm. No master bottleneck.
 *
 * ── ROOT CAUSE OF THE 0.166667 (UNIFORM MEMBERSHIP) BUG ────
 *
 *  The core failure was centroid collapse: all 6 centroids became
 *  bit-for-bit identical, so every E-step returned u_ij = 1/C = 0.1667.
 *
 *  WHY centroids collapsed (domain-guided init):
 *    The old init computed each centroid as the arithmetic mean of all
 *    L2-normalised document vectors in that domain group.  Clinical TF-IDF
 *    vectors share dominant terms ("patient", "history", "mg", …) so all
 *    4943 unit vectors point within a narrow cone on the 500-D unit sphere.
 *    Their per-cluster averages are nearly co-linear; after re-normalising,
 *    every centroid points in the direction of the global mean → identical.
 *
 *  FIX (domain init — the only critical fix):
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
 *    MPI_Gatherv is a collective — every rank must call it.  Restructured
 *    so all ranks always participate, then rank 0 does its serial work.
 *
 *  OTHER FIXES (from previous iteration, kept):
 *    FIX C: CSV header/index-column auto-detection.
 *    FIX D: fcm_mpi_load_labels() declared in header.
 *    FIX E/F/G: fgets checks, unused-param suppression, wide name buffer.
 *
 * ============================================================ */

#include "fcm_mpi.h"

/* ════════════════════════════════════════════════════════════
 * UTILITIES
 * ════════════════════════════════════════════════════════════ */

double l2_distance(const double *a, const double *b, int len) {
    double s = 0.0;
    for (int f = 0; f < len; f++) {
        double d = a[f] - b[f];
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

static void block_decompose(int N, int n_procs, int rank,
                             int *local_start, int *local_n) {
    int base = N / n_procs;
    int rem  = N % n_procs;
    *local_start = rank * base + (rank < rem ? rank : rem);
    *local_n     = base + (rank < rem ? 1 : 0);
}

/* FIX B: L2-normalise each row of a [rows x cols] matrix, in-place. */
static void normalise_rows(double *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double norm = 0.0;
        for (int f = 0; f < cols; f++)
            norm += mat[i * cols + f] * mat[i * cols + f];
        norm = sqrt(norm);
        if (norm < 1e-12) continue;
        for (int f = 0; f < cols; f++)
            mat[i * cols + f] /= norm;
    }
}

/* ════════════════════════════════════════════════════════════
 * LIFECYCLE
 * ════════════════════════════════════════════════════════════ */

FCMMpiModel *fcm_mpi_create(int N, int F, int C) {
    FCMMpiModel *m = (FCMMpiModel *)calloc(1, sizeof(FCMMpiModel));
    if (!m) { fprintf(stderr, "calloc FCMMpiModel failed\n"); exit(1); }

    m->N = N; m->F = F; m->C = C;
    MPI_Comm_rank(MPI_COMM_WORLD, &m->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &m->n_procs);
    block_decompose(N, m->n_procs, m->rank, &m->local_start, &m->local_n);

    m->local_data  = alloc_flat(m->local_n, F);
    m->local_U     = alloc_flat(m->local_n, C);
    m->local_U_old = alloc_flat(m->local_n, C);
    m->centroids   = alloc_flat(C, F);
    m->all_U       = NULL;
    if (m->rank == 0)
        m->all_U = alloc_flat(N, C);
    return m;
}

void fcm_mpi_free(FCMMpiModel *m) {
    if (!m) return;
    free(m->local_data);
    free(m->local_U);
    free(m->local_U_old);
    free(m->centroids);
    if (m->all_U) free(m->all_U);
    free(m);
}

/* ════════════════════════════════════════════════════════════
 * DATA LOADING & DISTRIBUTION  (FIX B + FIX C)
 * ════════════════════════════════════════════════════════════ */

/* FIX C: return 1 if first non-space char of the file is alphabetic
 * or a quote — meaning it's a text header row, not numeric data.    */
static int csv_has_header(FILE *fp) {
    long pos = ftell(fp);
    int  ch;
    /* skip BOM and whitespace */
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

/* FIX C: return 1 if the first token on this line is a plain integer
 * (no dot, no exponent) — indicating a pandas row-index column.    */
static int csv_has_index_col(const char *line) {
    const char *p = line;
    while (*p == ' ') p++;
    if (*p == '"' || *p == '\'') return 0;
    int has_dot = 0, has_e = 0, ndigits = 0;
    while (*p && *p != ',' && *p != '\n' && *p != '\r') {
        if (*p == '.')             has_dot = 1;
        else if (*p=='e'||*p=='E') has_e   = 1;
        else if (*p>='0'&&*p<='9') ndigits++;
        p++;
    }
    return (ndigits > 0 && !has_dot && !has_e);
}

int fcm_mpi_load_and_scatter(FCMMpiModel *m,
                              const char *features_csv,
                              const char *labels_csv) {
    (void)labels_csv;   /* FIX F */
    int N = m->N, F = m->F, P = m->n_procs;

    int *sendcounts = (int *)malloc(P * sizeof(int));
    int *displs     = (int *)malloc(P * sizeof(int));
    for (int r = 0; r < P; r++) {
        int ls, ln;
        block_decompose(N, P, r, &ls, &ln);
        sendcounts[r] = ln * F;
        displs[r]     = ls * F;
    }

    double *global_data = NULL;
    if (m->rank == 0) {
        global_data = alloc_flat(N, F);
        FILE *fp = fopen(features_csv, "r");
        if (!fp) {
            fprintf(stderr, "[rank 0] ERROR: cannot open '%s'\n", features_csv);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        char line[1 << 20];
        int  skip_index = 0;

        /* FIX C: skip header row if present */
        if (csv_has_header(fp)) {
            if (!fgets(line, sizeof(line), fp)) {
                fprintf(stderr, "[rank 0] features.csv appears empty\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            printf("[rank 0] Skipped header row in '%s'\n", features_csv);
        }

        /* FIX C: detect index column from first data line */
        long data_start = ftell(fp);
        if (fgets(line, sizeof(line), fp)) {
            skip_index = csv_has_index_col(line);
            if (skip_index)
                printf("[rank 0] Will skip row-index column in '%s'\n", features_csv);
        }
        fseek(fp, data_start, SEEK_SET);

        /* Read N data rows */
        for (int i = 0; i < N; i++) {
            if (!fgets(line, sizeof(line), fp)) {
                fprintf(stderr, "[rank 0] Unexpected EOF at row %d\n", i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            char *tok = strtok(line, ",\n\r");
            if (skip_index) tok = strtok(NULL, ",\n\r");  /* skip index token */
            for (int f = 0; f < F; f++) {
                global_data[i * F + f] = tok ? atof(tok) : 0.0;
                tok = strtok(NULL, ",\n\r");
            }
        }
        fclose(fp);

        /* FIX B: L2-normalise all rows so L2 distance = angular distance */
        normalise_rows(global_data, N, F);
        printf("[rank 0] Loaded & L2-normalised %d x %d matrix from '%s'\n",
               N, F, features_csv);
    }

    MPI_Scatterv(global_data,   sendcounts, displs, MPI_DOUBLE,
                 m->local_data, m->local_n * F,     MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    if (m->rank == 0) free(global_data);
    free(sendcounts); free(displs);
    return 0;
}

/* ════════════════════════════════════════════════════════════
 * LABEL LOADING
 * ════════════════════════════════════════════════════════════ */
int *fcm_mpi_load_labels(FCMMpiModel *m, const char *labels_csv) {
    int *labels = NULL;
    if (m->rank == 0) {
        labels = (int *)malloc(m->N * sizeof(int));
        FILE *fp = fopen(labels_csv, "r");
        if (!fp) {
            fprintf(stderr, "[rank 0] Cannot open '%s'\n", labels_csv);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        char line[512];
        /* FIX E: check fgets when skipping header */
        if (!fgets(line, sizeof(line), fp)) {
            fprintf(stderr, "[rank 0] Labels file is empty\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        /* FIX G: wider buffer, snprintf */
        char known[64][512];
        int  n_known = 0;
        for (int i = 0; i < m->N; i++) {
            if (!fgets(line, sizeof(line), fp)) break;
            line[strcspn(line, "\r\n")] = '\0';
            char *s = line;
            /* strip quotes and spaces */
            while (*s == ' ' || *s == '\'' || *s == '"') s++;
            char *end = s + strlen(s) - 1;
            while (end > s && (*end == '\'' || *end == '"' || *end == ' '))
                *end-- = '\0';
            int id = -1;
            for (int k = 0; k < n_known; k++)
                if (strcmp(known[k], s) == 0) { id = k; break; }
            if (id == -1 && n_known < 64) {
                snprintf(known[n_known], 512, "%s", s);
                id = n_known++;
            } else if (id == -1) id = n_known - 1;
            labels[i] = id % m->C;
        }
        fclose(fp);
        printf("[rank 0] Loaded %d labels (%d unique -> %d clusters)\n",
               m->N, n_known, m->C);
    }
    return labels;
}

/* ════════════════════════════════════════════════════════════
 * INITIALISATION STRATEGIES
 * ════════════════════════════════════════════════════════════ */


/* ── Random ───────────────────────────────────────────────── */
void fcm_mpi_init_random(FCMMpiModel *m) {
    int N = m->N, F = m->F, C = m->C, P = m->n_procs;
    int *rc = (int*)malloc(P*sizeof(int)), *di = (int*)malloc(P*sizeof(int));
    for (int r=0;r<P;r++){int ls,ln;block_decompose(N,P,r,&ls,&ln);rc[r]=ln*F;di[r]=ls*F;}
    double *gd = NULL;
    if (m->rank==0) gd = alloc_flat(N, F);
    MPI_Gatherv(m->local_data, m->local_n*F, MPI_DOUBLE,
                gd, rc, di, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (m->rank==0) {
        srand((unsigned int)time(NULL));
        int *picked = (int*)calloc(N, sizeof(int));
        for (int k=0;k<C;k++) {
            int idx; do{idx=rand()%N;}while(picked[idx]);
            picked[idx]=1;
            memcpy(m->centroids+k*F, gd+idx*F, F*sizeof(double));
        }
        free(picked); free(gd);
        printf("[init] Random init: %d centroids picked.\n", C);
    }
    free(rc); free(di);
    MPI_Bcast(m->centroids, C*F, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

/* ── K-Means++ (uses ALL N points, not a subsample) ─────── */
void fcm_mpi_init_kmeanspp(FCMMpiModel *m) {
    int N = m->N, F = m->F, C = m->C, P = m->n_procs;
    int *rc = (int*)malloc(P*sizeof(int)), *di = (int*)malloc(P*sizeof(int));
    for (int r=0;r<P;r++){int ls,ln;block_decompose(N,P,r,&ls,&ln);rc[r]=ln*F;di[r]=ls*F;}
    double *gd = NULL;
    if (m->rank==0) gd = alloc_flat(N, F);
    MPI_Gatherv(m->local_data, m->local_n*F, MPI_DOUBLE,
                gd, rc, di, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (m->rank==0) {
        srand((unsigned int)time(NULL));
        int    *seed = (int*)malloc(C*sizeof(int));
        double *md2  = (double*)malloc(N*sizeof(double));
        seed[0] = rand() % N;
        for (int k=1;k<C;k++) {
            double total = 0.0;
            for (int i=0;i<N;i++) {
                double best = DBL_MAX;
                for (int s=0;s<k;s++) {
                    double d = l2_distance(gd+i*F, gd+seed[s]*F, F);
                    if (d*d < best) best = d*d;
                }
                md2[i] = best; total += best;
            }
            double r2 = ((double)rand()/RAND_MAX)*total, cum = 0.0;
            seed[k] = 0;
            for (int i=0;i<N;i++) { cum+=md2[i]; if(cum>=r2){seed[k]=i;break;} }
        }
        for (int k=0;k<C;k++)
            memcpy(m->centroids+k*F, gd+seed[k]*F, F*sizeof(double));
        free(seed); free(md2); free(gd);
        printf("[init] K-Means++ init complete.\n");
    }
    free(rc); free(di);
    MPI_Bcast(m->centroids, C*F, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

/* ── Domain-guided ───────────────────────────────────────── */
/*
 * FIX DOMAIN-INIT: Averaged class-mean centroids collapse to the global
 * mean after L2 normalisation because all medical TF-IDF vectors share
 * dominant terms ("patient", "history", …) and live in a narrow cone on
 * the unit sphere.  Normalising their arithmetic mean just re-points it
 * at the same global-mean direction for every cluster → identical centroids
 * → uniform E-step → delta stays near zero but membership is meaningless.
 *
 * Solution: medoid selection — pick the document inside each domain group
 * whose feature vector is closest (in L2) to that group's mean.  Medoids
 * are real data points, so they are already on the unit sphere and are
 * genuinely spread out across the feature space.  This gives the E-step
 * real angular separation to work with.
 *
 * Additionally, the MPI collective structure is corrected so all ranks
 * always call MPI_Gatherv (it is a collective; skipping it on any rank
 * causes a hang or silent corruption).
 */
void fcm_mpi_init_domain(FCMMpiModel *m, int *domain_labels) {
    int N = m->N, F = m->F, C = m->C, P = m->n_procs;

    /* ── Step 1: all ranks gather full data to rank 0 (collective) ── */
    int *rc=(int*)malloc(P*sizeof(int)), *di=(int*)malloc(P*sizeof(int));
    for(int r=0;r<P;r++){int ls,ln;block_decompose(N,P,r,&ls,&ln);rc[r]=ln*F;di[r]=ls*F;}
    double *gd = NULL;
    if(m->rank==0) gd = alloc_flat(N,F);
    MPI_Gatherv(m->local_data, m->local_n*F, MPI_DOUBLE,
                gd, rc, di, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(rc); free(di);

    /* ── Step 2: rank 0 selects medoids and broadcasts ── */
    if(m->rank==0) {
        /* Compute per-cluster mean (unnormalised, for medoid search only) */
        double *sums = alloc_flat(C,F);
        int    *cnts = (int*)calloc(C,sizeof(int));
        for(int i=0;i<N;i++){
            int c=domain_labels[i]; cnts[c]++;
            for(int f=0;f<F;f++) sums[c*F+f]+=gd[i*F+f];
        }
        for(int j=0;j<C;j++)
            if(cnts[j]>0)
                for(int f=0;f<F;f++) sums[j*F+f]/=cnts[j];

        /* Select medoid: doc in cluster j closest to that cluster's mean */
        for(int j=0;j<C;j++){
            int best_idx=-1; double best_d=DBL_MAX;
            if(cnts[j]==0){
                /* empty cluster: fall back to a random k-means++-style pick */
                best_idx=rand()%N;
            } else {
                for(int i=0;i<N;i++){
                    if(domain_labels[i]!=j) continue;
                    double d=l2_distance(gd+i*F, sums+j*F, F);
                    if(d<best_d){best_d=d;best_idx=i;}
                }
            }
            /* Centroid = medoid document (already unit-length after load) */
            memcpy(m->centroids+j*F, gd+best_idx*F, F*sizeof(double));
        }
        free(sums); free(cnts); free(gd);
        printf("[init] Domain-guided init complete (medoid selection).\n");
    }
    MPI_Bcast(m->centroids, C*F, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ── Step 3: scatter labels and seed local_U ── */
    int *lsc=(int*)malloc(P*sizeof(int)), *ldi=(int*)malloc(P*sizeof(int));
    for(int r=0;r<P;r++){int ls,ln;block_decompose(N,P,r,&ls,&ln);lsc[r]=ln;ldi[r]=ls;}
    int *ll=(int*)malloc(m->local_n*sizeof(int));
    MPI_Scatterv(domain_labels, lsc, ldi, MPI_INT,
                 ll, m->local_n, MPI_INT, 0, MPI_COMM_WORLD);
    free(lsc); free(ldi);

    /* Seed local_U: high confidence on assigned cluster, uniform elsewhere */
    for(int i=0;i<m->local_n;i++){
        int dom=ll[i]; double rs=0.0;
        for(int j=0;j<m->C;j++){
            m->local_U[i*m->C+j]=(j==dom)?0.7:0.3/(m->C-1);
            rs+=m->local_U[i*m->C+j];
        }
        for(int j=0;j<m->C;j++) m->local_U[i*m->C+j]/=rs;
    }
    free(ll);
}

/* ════════════════════════════════════════════════════════════
 * PARALLEL M-STEP
 * Non-blocking Iallreduce for numerator overlaps denominator reduce.
 * FIX B: centroids re-normalised after each update.
 * ════════════════════════════════════════════════════════════ */
void fcm_mpi_update_centroids(FCMMpiModel *m) {
    int ln=m->local_n, F=m->F, C=m->C;
    double *ln_num=alloc_flat(C,F), *ln_den=(double*)calloc(C,sizeof(double));
    double *gn_num=alloc_flat(C,F), *gn_den=(double*)calloc(C,sizeof(double));

    for(int i=0;i<ln;i++)
        for(int j=0;j<C;j++){
            double u_m=pow(m->local_U[i*C+j], FCM_M);
            ln_den[j]+=u_m;
            for(int f=0;f<F;f++) ln_num[j*F+f]+=u_m*m->local_data[i*F+f];
        }

    /* Non-blocking numerator + blocking denominator (overlap comm) */
    MPI_Request req;
    MPI_Iallreduce(ln_num, gn_num, C*F, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &req);
    MPI_Allreduce (ln_den, gn_den, C,   MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Wait(&req, MPI_STATUS_IGNORE);

    for(int j=0;j<C;j++){
        double denom=(gn_den[j]<FCM_MIN_DIST)?FCM_MIN_DIST:gn_den[j];
        for(int f=0;f<F;f++) m->centroids[j*F+f]=gn_num[j*F+f]/denom;
    }
    /* Keep centroids on unit sphere */
    normalise_rows(m->centroids, C, F);

    free(ln_num);free(ln_den);free(gn_num);free(gn_den);
}
/* ════════════════════════════════════════════════════════════
 * PARALLEL E-STEP  (embarrassingly parallel, no communication)
 * ════════════════════════════════════════════════════════════ */
void fcm_mpi_update_membership(FCMMpiModel *m) {
    int ln=m->local_n, F=m->F, C=m->C;
    double expn=2.0/(FCM_M-1.0);
    double *dist=(double*)malloc(C*sizeof(double));
    for(int i=0;i<ln;i++){
        int zc=-1;
        for(int j=0;j<C;j++){
            dist[j]=l2_distance(m->local_data+i*F, m->centroids+j*F, F);
            if(dist[j]<FCM_MIN_DIST) zc=j;
        }
        if(zc>=0){for(int j=0;j<C;j++)m->local_U[i*C+j]=(j==zc)?1.0:0.0;continue;}
        for(int j=0;j<C;j++){
            double sum=0.0;
            for(int k=0;k<C;k++) sum+=pow(dist[j]/dist[k], expn);
            m->local_U[i*C+j]=1.0/sum;
        }
    }
    free(dist);
}

/* ════════════════════════════════════════════════════════════
 * PARALLEL CONVERGENCE  (distributed Frobenius norm)
 * ════════════════════════════════════════════════════════════ */
double fcm_mpi_convergence(FCMMpiModel *m) {
    int ln=m->local_n, C=m->C;
    double loc=0.0;
    for(int i=0;i<ln;i++)
        for(int j=0;j<C;j++){
            double d=m->local_U[i*C+j]-m->local_U_old[i*C+j];
            loc+=d*d;
        }
    double glob=0.0;
    MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(glob);
}

/* ════════════════════════════════════════════════════════════
 * MAIN PARALLEL FCM LOOP
 * FIX A: no pre-loop M-step. E-step runs first every iteration.
 * ════════════════════════════════════════════════════════════ */
void fcm_mpi_run(FCMMpiModel *m, InitStrategy strategy,
                 int *domain_labels) {
    if(m->rank==0)
        printf("\n[fcm_mpi] N=%d  F=%d  C=%d  P=%d  m=%.1f  eps=%.0e\n",
               m->N, m->F, m->C, m->n_procs, FCM_M, FCM_EPSILON);

    switch(strategy){
        case INIT_KMEANSPP: fcm_mpi_init_kmeanspp(m);              break;
        case INIT_DOMAIN:   fcm_mpi_init_domain(m, domain_labels); break;
        default:            fcm_mpi_init_random(m);                 break;
    }
    for(int iter=1; iter<=FCM_MAX_ITER; iter++){
        double t0=MPI_Wtime();
        copy_flat(m->local_U_old, m->local_U, m->local_n*m->C);
        fcm_mpi_update_membership(m);   /* E-step: no comm */
        fcm_mpi_update_centroids(m);    /* ← UNCOMMENT THIS LINE */
        double delta=fcm_mpi_convergence(m);
        m->iter_times[iter-1]=MPI_Wtime()-t0;
        if(m->rank==0 && (iter<=5||iter%10==0))
            printf("[fcm_mpi] iter %4d  delta=%.8f  time=%.4fs\n",
                   iter, delta, m->iter_times[iter-1]);
        if(delta<FCM_EPSILON){
            if(m->rank==0)
                printf("[fcm_mpi] Converged at iter %d (delta=%.2e)\n",iter,delta);
            m->iterations=iter; m->final_delta=delta;
            fcm_mpi_update_membership(m);
            return;
        }
    }
    if(m->rank==0) printf("[fcm_mpi] Reached MAX_ITER=%d\n",FCM_MAX_ITER);
    m->iterations=FCM_MAX_ITER;
    m->final_delta=fcm_mpi_convergence(m);
}

/* ════════════════════════════════════════════════════════════
 * OUTPUT
 * ════════════════════════════════════════════════════════════ */
void fcm_mpi_gather_and_save(FCMMpiModel *m,
                              const char *membership_path,
                              const char *centroids_path) {
    int N=m->N, C=m->C, F=m->F, P=m->n_procs;
    int *rc=(int*)malloc(P*sizeof(int)), *di=(int*)malloc(P*sizeof(int));
    for(int r=0;r<P;r++){int ls,ln;block_decompose(N,P,r,&ls,&ln);rc[r]=ln*C;di[r]=ls*C;}
    MPI_Gatherv(m->local_U,m->local_n*C,MPI_DOUBLE,
                m->all_U,rc,di,MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(m->rank==0){
        FILE *fp=fopen(membership_path,"w");
        if(fp){
            for(int i=0;i<N;i++){
                for(int j=0;j<C;j++){fprintf(fp,"%.6f",m->all_U[i*C+j]);if(j<C-1)fputc(',',fp);}
                fputc('\n',fp);
            }
            fclose(fp);
            printf("[fcm_mpi] Membership saved -> %s\n",membership_path);
        }
        FILE *fc=fopen(centroids_path,"w");
        if(fc){
            for(int j=0;j<C;j++){
                for(int f=0;f<F;f++){fprintf(fc,"%.6f",m->centroids[j*F+f]);if(f<F-1)fputc(',',fc);}
                fputc('\n',fc);
            }
            fclose(fc);
            printf("[fcm_mpi] Centroids saved -> %s\n",centroids_path);
        }
    }
    free(rc);free(di);
}

void fcm_mpi_print_summary(FCMMpiModel *m){
    if(m->rank!=0) return;
    int N=m->N, C=m->C;
    int *cnt=(int*)calloc(C,sizeof(int));
    for(int i=0;i<N;i++){
        int best=0;
        for(int j=1;j<C;j++) if(m->all_U[i*C+j]>m->all_U[i*C+best]) best=j;
        cnt[best]++;
    }
    printf("\n[fcm_mpi] -- Summary --\n");
    printf("[fcm_mpi] Iterations : %d\n",m->iterations);
    printf("[fcm_mpi] Final delta: %.2e\n",m->final_delta);
    printf("[fcm_mpi] Cluster distribution:\n");
    for(int j=0;j<C;j++)
        printf("  Cluster %2d : %4d documents (%.1f%%)\n",
               j,cnt[j],100.0*cnt[j]/N);
    free(cnt);
}

void fcm_mpi_print_timing(FCMMpiModel *m){
    if(m->rank!=0) return;
    double total=0.0;
    for(int i=0;i<m->iterations;i++) total+=m->iter_times[i];
    printf("[fcm_mpi] Total wall time : %.4f s\n",total);
    printf("[fcm_mpi] Avg time/iter   : %.4f s\n",total/m->iterations);
}