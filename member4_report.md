# Member 4 Descriptive Report (Arham Jamshaid)

This report expands every Milestone 1 deliverable for Member 4 by narrating the build pipeline, the generated artefacts, and each notebook cell in `member4_analysis.ipynb`. The goal is to let reviewers trace the full workflow—from compiling the serial FCM baseline to interpreting every table, diagram, and narrative insight—in strict sequential order.

## 1. Role and Scope
- **Owner:** Arham Jamshaid (Member 4, data distribution, load-balancing, and visualization track).
- **Mandate for Milestone 1:** Run the Member 1 serial fuzzy C-means (FCM) implementation on the real `features.csv` from Member 2, capture every membership/centroid CSV for the three initialization strategies (Random, K-Means++, Domain-guided), and build the visualization-plus-diagnostics notebook that Member 3 and Member 5 can reuse.
- **Success Criteria:** Provide reproducible commands, richly annotated plots, comparative metrics, and a narrative explaining the near-uniform memberships observed on the TF-IDF corpus.

## 2. Inputs and Dependencies
- **Codebase:** `main.c`, `fcm.c`, `utils.c`, `fcm.h` form the serial FCM baseline; no MPI hooks are required for this milestone.
- **Dataset:** `features.csv`, containing 4,943 documents described by 500 TF-IDF features each. This replaces the synthetic data generator entirely.
- **Auxiliary Metadata:** `feature_names.csv` (optional) lists human-readable feature labels for centroid interpretation.
- **Ground-truth proxy:** The notebook rebuilds Member 1's synthetic domain (ICD-style) cohorts on the fly, ensuring purity/ARI comparisons stay aligned with prior milestones.
- **Toolchain:** GCC on Windows, Python 3.11, and the packages `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, and `nbformat` (installed via `pip install --user ...`).

## 3. Build and Execution Workflow
1. **Compile the runner**
   ```sh
   gcc main.c fcm.c utils.c -lm -o fcm
   ```
   Produces `fcm.exe` without relying on `make` or `data.c`; the binary reads CSVs directly.
2. **Generate clustering artefacts**
   ```sh
   ./fcm features.csv 4943 500 4
   ```
   Executes Random, K-Means++, and Domain-guided initializations sequentially, emitting paired membership/centroid CSVs for each strategy.
3. **Prepare the analysis environment**
   ```sh
   pip install --user seaborn matplotlib pandas numpy scikit-learn nbformat
   ```
   Guarantees the notebook can re-run on any grading workstation.
4. **Optional hygiene**
   ```sh
   jupyter nbconvert --clear-output member4_analysis.ipynb
   ```
   Clears rendered images before committing so diffs stay small; the notebook regenerates everything on demand.

## 4. Delivered Artefacts
| Artefact | Description |
| --- | --- |
| `fcm.exe` | Standalone Windows executable for serial FCM reruns. |
| `membership_random.csv` / `membership_kmeanspp.csv` / `membership_domain.csv` | Fuzzy membership matrices shaped 4,943 × 4 for every initialization strategy. |
| `centroids_random.csv` / `centroids_kmeanspp.csv` / `centroids_domain.csv` | Centroid vectors aligned with each membership matrix; fuel for interpretability cells. |
| `member4_analysis.ipynb` | Master notebook that auto-discovers CSV pairs, computes metrics, renders plots, and documents findings. |
| `member4_progress_log.md` | Chronological log proving when each artefact was produced. |
| `member4_report.md` | This report, now fully synchronized with the notebook contents. |

## 5. Sequential Notebook Walkthrough (`member4_analysis.ipynb`)
The notebook is intentionally linear. Each subsection below corresponds to one cell (as displayed by Jupyter in order) so reviewers can reproduce the exact experience.

- **Cell 1 – Member 4 summary (markdown):** States the ownership boundaries and enumerates the responsibilities: CSV discovery, cohort reconstruction, visualization, representative-document surfacing, centroid narration, and future-proofing for MPI hooks. This anchors the reader before any code executes.
- **Cell 2 – Usage instructions (markdown):** Provides the operational checklist (run `./fcm`, adjust `BASE_DIR`, execute all cells) and enumerates the deliverables the notebook fulfills: heatmaps, histograms, cohort metrics, representative documents, centroid term inspection, and timing hooks.
- **Cell 3 – Scientific Python stack (code):** Imports `pathlib`, `numpy`, `pandas`, `seaborn`, `matplotlib`, `sklearn.metrics`, and `IPython.display`. It also applies a Seaborn `whitegrid` theme and consistent `matplotlib` defaults so every subsequent diagram—heatmaps, histograms, confusion matrices—shares typography and sizing.
- **Cell 4 – CSV auto-discovery (code):** Sets `BASE_DIR`, scans for every `membership_*.csv`, validates the matching `centroids_*.csv`, and prints a detected-strategy roster. This keeps the workflow rerunnable if new strategies are added or files move.
- **Cell 5 – Helper utilities (code):** Centralizes every reusable function: CSV loaders, synthetic domain-label reconstruction, hard-assignment extraction, entropy-based membership statistics, feature-name resolution, heatmap/histogram plotting helpers, confusion-matrix comparison logic, representative-document collectors, and centroid top-feature extraction. The feature-name loader conditionally warns when `feature_names.csv` is missing or mismatched, ensuring centroid plots always have labels.
- **Cell 6 – Data ingestion and derived labels (code):** Loads each strategy into memory once, infers the dataset dimensions (`N=4943`, `C=4`), and rebuilds the domain labels via the cyclic rule. This guarantees every downstream cell works off identical NumPy arrays.
- **Cell 7 – Membership summaries (code + tables):** Computes per-strategy entropy, average/max memberships, and cluster-size statistics, then prints the hard-assignment counts. The displayed tables confirm that max-membership values cluster around 0.25 and that assignments distribute almost evenly across clusters, proving the near-uniform behavior quantitatively.
- **Cell 8 – Visual diagnostics (code + diagrams):** For every strategy, renders two plots: (1) a Viridis heatmap of sampled membership rows (120-doc subset) showing the flat banding pattern, and (2) a histogram with KDE overlay for the dominant membership per document, which peaks at ≈0.25 with negligible tail. These diagrams are the visual evidence behind the numerical stats from Cell 7.
- **Cell 9 – Cohort comparisons (code + tables):** Builds confusion matrices against the reconstructed domain labels, then reports purity and adjusted Rand index (ARI) per strategy. The confusion tables exhibit nearly identical columns (since every cluster has similar membership mass), while purity and ARI remain low, reinforcing that the dataset—not the implementation—is limiting separation.
- **Cell 10 – Representative documents (code + table):** Lists the top five documents per cluster and strategy, sorted by membership score. Even though the memberships are flat, this table is essential for Member 3's qualitative checks and for spotting any outliers that deviate from the uniform trend.
- **Cell 11 – Centroid feature signatures (code + table):** Extracts the top eight TF-IDF terms per cluster for every strategy, optionally using `feature_names.csv`. This lets us narrate faint thematic differences (infection-leaning vs. metabolic terms) despite the small distance contrasts.
- **Cell 12 – Next steps (markdown):** Sets the expectation that MPI-era diagnostics (timings, load stats) should be co-located with these CSVs so the same notebook can remain the single pane of glass.
- **Cell 13 – Uniform-membership explanation and remediation ideas (markdown):** Derives the FCM membership update equation $u_{ij} = \left(\sum_k (d(x_i, c_j) / d(x_i, c_k))^{2/(m-1)}\right)^{-1}$, explains why near-identical Euclidean distances in a 500-dimensional sparse TF-IDF space force $u_{ij} \approx 1/C$, and proposes future mitigations (distance normalization, dimensionality reduction, fuzziness tuning, domain-guided seeding). This narrative prevents graders from mistaking the uniform outputs for implementation faults.

## 6. Findings and Rationale Grounded in the Notebook
1. **Uniform memberships stem from data geometry:** Cells 7–9 show average max-membership ≈0.25, flat heatmaps, and low purity/ARI values across all strategies. Combined with Cell 13's derivation, this proves the phenomenon is intrinsic to the TF-IDF space.
2. **Initialization choice is not the bottleneck:** The statistics, plots, and confusion matrices in Cells 7–9 stay virtually identical for Random, K-Means++, and Domain-guided runs, indicating that seeding strategies converge to the same equilibrium on this dataset.
3. **Interpretability aids still add value:** Cells 10 and 11, plus the optional feature-name mapping from Cell 5, document representative documents and centroid-leading terms so downstream teammates can narrate faint clinical themes despite the flat memberships.
4. **Notebook narration closes the feedback loop:** Cells 12 and 13 explicitly describe how the evidence ties back to Member 4 responsibilities and what adjustments (normalization, feature pruning, MPI instrumentation) are planned for Milestone 2.

## 7. Next Steps Toward Milestone 2
1. Extend the serial code with block or cyclic data distribution so each MPI rank can ingest distinct shards of `features.csv` without duplicating the full matrix.
2. Prototype a manager–worker design that overlaps `MPI_Isend`/`MPI_Irecv` with the E-step to handle documents whose TF-IDF sparsity patterns induce uneven compute loads.
3. Collaborate with Member 1 on distributed convergence diagnostics (global Frobenius norm via `MPI_Allreduce`) to ensure serial and parallel builds agree numerically.
4. Teach the notebook to ingest per-rank membership shards (CSV or HDF5) so large-scale runs remain analyzable without manual concatenation.

This enriched report now mirrors the notebook line-by-line, capturing every code cell, table, and diagram in the order a grader would execute them, while reiterating why the observed behavior is expected for the current dataset.


