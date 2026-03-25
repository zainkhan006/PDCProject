/* ============================================================
 * data.c  —  Dummy clinical data generator & CSV loader
 *
 * The dummy generator creates TF-IDF-like feature vectors that
 * simulate real clinical notes.  Four synthetic "condition groups"
 * are embedded so clusters are actually meaningful even without
 * real MIMIC-III data:
 *
 *   Group 0 — Cardiovascular  (hypertension, chest pain, ECG)
 *   Group 1 — Respiratory     (dyspnoea, cough, SpO2)
 *   Group 2 — Metabolic       (diabetes, glucose, HbA1c)
 *   Group 3 — Neurological    (headache, dizziness, neuro exam)
 *
 * Each group "activates" a distinct subset of features with higher
 * weights, mimicking how clinical concepts cluster in TF-IDF space.
 *
 * Member 1: Khansa Danish
 * ============================================================ */

#include "fcm.h"

/* ── Simple LCG random in [0,1] ─────────────────────────────
   We keep our own generator so results are reproducible without
   depending on any platform's rand() implementation.           */
static unsigned long _lcg_state = 12345UL;

static void lcg_seed(unsigned int s) { _lcg_state = (unsigned long)s; }

static double lcg_rand(void) {
    _lcg_state = (_lcg_state * 1664525UL + 1013904223UL) & 0xFFFFFFFFUL;
    return (double)_lcg_state / (double)0xFFFFFFFFUL;
}

/* ── Generate synthetic clinical TF-IDF vectors ─────────────
 *
 * Parameters
 *   model  — must already be allocated (fcm_create called)
 *   seed   — random seed for reproducibility
 *
 * Strategy
 *   1. Divide features into C equal "concept bands"
 *      (one band per clinical condition group).
 *   2. Assign each document a dominant group randomly.
 *   3. Within the dominant band, sample from a higher-mean
 *      distribution (signal); all other features get low noise.
 *   4. Values are clipped to [0,1] — valid TF-IDF range.
 * ──────────────────────────────────────────────────────────── */
void fcm_generate_clinical_dummy(FCMModel *model, unsigned int seed) {
    lcg_seed(seed);

    int N = model->n_points;
    int F = model->n_features;
    int C = model->n_clusters;

    /* Width of each condition's "concept band" in feature space */
    int band = F / C;

    for (int i = 0; i < N; i++) {
        /* Pick dominant clinical group for this document */
        int group = (int)(lcg_rand() * C);
        if (group >= C) group = C - 1;  /* boundary guard */

        for (int f = 0; f < F; f++) {
            double val;
            /* Is feature f inside the dominant band? */
            if (f >= group * band && f < (group + 1) * band) {
                /* Signal: high TF-IDF weight (mean ~0.6, noise ±0.3) */
                val = 0.45 + lcg_rand() * 0.45;
            } else {
                /* Background noise (mean ~0.05) */
                val = lcg_rand() * 0.12;
            }
            /* Clip to valid range */
            model->data[i][f] = (val > 1.0) ? 1.0 : (val < 0.0 ? 0.0 : val);
        }
    }

    printf("[data] Generated %d clinical dummy vectors (%d features, %d groups)\n",
           N, F, C);
    printf("[data] Condition bands: ~%d features per group\n", band);
}

/* ── Load feature matrix from a CSV file ────────────────────
 *
 * Expected format (no header row):
 *   f0,f1,f2,...,fF-1
 *   one row per document
 *
 * Returns 0 on success, -1 on error.
 * model->n_points and model->n_features must already be set.
 * ──────────────────────────────────────────────────────────── */
int fcm_load_csv(FCMModel *model, const char *filepath) {
    FILE *fp = fopen(filepath, "r");
    if (!fp) {
        fprintf(stderr, "[data] ERROR: cannot open '%s'\n", filepath);
        return -1;
    }

    int N = model->n_points;
    int F = model->n_features;
    char line[1 << 16];   /* 64 KB line buffer — handles high-dim vectors */

    for (int i = 0; i < N; i++) {
        if (!fgets(line, sizeof(line), fp)) {
            fprintf(stderr, "[data] ERROR: unexpected EOF at row %d\n", i);
            fclose(fp);
            return -1;
        }
        char *token = strtok(line, ",\n");
        for (int f = 0; f < F; f++) {
            if (!token) {
                fprintf(stderr, "[data] ERROR: too few columns at row %d\n", i);
                fclose(fp);
                return -1;
            }
            model->data[i][f] = atof(token);
            token = strtok(NULL, ",\n");
        }
    }

    fclose(fp);
    printf("[data] Loaded %d x %d feature matrix from '%s'\n", N, F, filepath);
    return 0;
}
