# Member 4 Milestone 1 Log (Arham Jamshaid)

This log details every artifact and workflow step I completed while delivering the Member 4 responsibilities (data distribution & visualization) using the real `features.csv` dataset instead of the original dummy generator.
## Files Generated or Updated
- `fcm.exe` — compiled from `main.c`, `fcm.c`, and `utils.c` so I can regenerate outputs on Windows even without `make`.
- `membership_random.csv`, `membership_kmeanspp.csv`, `membership_domain.csv` — fresh membership matrices produced by running `./fcm features.csv 4943 500 4`; each is an `N × C` (4 943 × 4) matrix whose rows sum to 1 and feeds the notebook diagnostics.
- `centroids_random.csv`, `centroids_kmeanspp.csv`, `centroids_domain.csv` — centroid matrices aligned with the memberships above; required for the top-feature summaries.
- `member4_analysis.ipynb` — analysis notebook that auto-detects any `membership_*.csv`/`centroids_*.csv` pairs, rebuilds the dummy ICD-style labels, and emits the full validation suite (counts, entropy metrics, heatmaps, histograms, purity/ARI tables, representative documents, centroid feature descriptions, and explanatory notes on the near-uniform outputs observed with `features.csv`).
- `member4_progress_log.md` — this document, updated to reflect the real-data workflow.

## Commands Executed (latest run on real data)
1. `gcc main.c fcm.c utils.c -lm -o fcm` — compile the FCM binary after switching the pipeline to load `features.csv` directly (no `data.c`).
2. `./fcm features.csv 4943 500 4` — generate memberships/centroids for all three initialization strategies using the true dataset dimensions.
3. `pip install --user seaborn matplotlib pandas numpy scikit-learn` — ensure the Python plotting/analysis stack is available (idempotent; most packages already present).
4. `pip install --user nbformat` — guarantee Jupyter serialization dependencies are on the system.
5. `jupyter nbconvert --clear-output member4_analysis.ipynb` (optional housekeeping) — clears stale outputs before committing/sharing.

## Implementation Notes
- The analysis notebook now states clearly that it ingests Member 1’s CSVs generated from `features.csv`, not dummy data, and documents why the real dataset leads to near-uniform memberships (high-dimensional sparse TF-IDF vectors with minimal distance contrast).
- All visualizations were retuned (larger figure sizes, controlled tick density) so heatmaps and histograms remain legible even when values differ only in the 10th decimal place.
- A closing markdown section explains the limitations (data geometry, fuzziness parameter) and enumerates future improvement levers—useful evidence that the current uniform-looking outputs are a property of the provided data, not an implementation error.
## How to Reproduce / Update
1. Rebuild and rerun the FCM binary whenever Member 2 supplies a new feature matrix: `gcc main.c fcm.c utils.c -lm -o fcm` followed by `./fcm <csv_path> <N> <F> <C>`.
2. Open `member4_analysis.ipynb` and **Run All**. The notebook automatically discovers the latest `membership_*` / `centroids_*` files, recomputes stats, and regenerates every figure/table.
3. Export/attach the CSVs and updated notebook (optionally cleared of outputs) for downstream team members (Member 3 metrics, Member 5 MPI experiments).

These steps capture the complete Member 4 contribution for Milestone 1 using the real `features.csv` data.
# Member 4 Milestone 1 Log

This note captures every artifact and command I (Member 4) produced while completing the Milestone 1 deliverables on top of Member 1's serial FCM baseline.

## Files Generated or Updated
- `fcm.exe` — built from the existing C sources so that I could regenerate dummy outputs locally.
- `membership_random.csv`, `membership_kmeanspp.csv`, `membership_domain.csv` — soft membership matrices emitted by `./fcm` for each initialization strategy; required by the visualization notebook.
- `centroids_random.csv`, `centroids_kmeanspp.csv`, `centroids_domain.csv` — centroid matrices paired with each membership CSV.
- `member4_analysis.ipynb` — new Jupyter notebook that ingests any `membership_*.csv`/`centroids_*.csv` pair and produces all Milestone 1 diagnostics (summaries, heatmaps, histograms, cohort comparisons, representative documents, centroid feature tables).

## Commands Executed
1. `gcc main.c fcm.c utils.c data.c -lm -o fcm`  5 Compiled the serial FCM implementation because `make` is unavailable on this Windows setup.
2. `./fcm`  5 Ran the dummy-data pipeline to regenerate the required membership and centroid CSVs.
3. `pip install --user seaborn matplotlib pandas numpy scikit-learn`  5 Ensured the scientific Python stack was present for the notebook (most packages were already installed; re-running pip is idempotent).
4. `pip install --user nbformat`  5 Added the notebook serialization dependency so a script could author the .ipynb file programmatically.
5. `python member4_notebook_builder.py`  5 One-off helper script that wrote `member4_analysis.ipynb` with all markdown/code cells, then deleted itself.

## How to Reproduce / Update
1. Rebuild + rerun the FCM binary whenever new feature CSVs arrive (e.g., from Member 2) to emit fresh `membership_*.csv` and `centroids_*.csv` files.
2. Open `member4_analysis.ipynb` and **Run All**. The notebook auto-detects every CSV pair in the folder, recomputes stats/plots, and displays updated cohort comparisons and cluster summaries.
3. Archive the regenerated CSVs + notebook outputs for Member 3 (metrics) and for documentation in Milestone 2.

No other files were created for Member 4 in Milestone 1 beyond those listed above.
