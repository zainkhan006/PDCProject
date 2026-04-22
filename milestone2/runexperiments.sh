#!/bin/bash
# ============================================================
# run_experiments.sh — Member 3: Systematic MPI Experiments
# Runs FCM with different configurations and collects timing data
#
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh
#
# Requires: mpirun, ./fcm_mpi, features.csv, specialty_labels.csv
# ============================================================

FEATURES="features.csv"
LABELS="specialty_labels.csv"
BINARY="./fcm_mpi"
RESULTS_DIR="experiment_results"
LOGFILE="$RESULTS_DIR/all_experiments.log"

# Docker runs as root — need this flag
MPI_FLAGS="--allow-run-as-root --oversubscribe"

mkdir -p "$RESULTS_DIR"
echo "=============================================" | tee "$LOGFILE"
echo "Member 3 — Milestone 2 Experiment Suite"       | tee -a "$LOGFILE"
echo "Date: $(date)"                                  | tee -a "$LOGFILE"
echo "=============================================" | tee -a "$LOGFILE"

# ── Experiment 1: Communication Mode Comparison ──────────────
# Same config (NP=4, K-Means++, Block), vary comm mode
echo "" | tee -a "$LOGFILE"
echo "=== EXPERIMENT 1: Communication Mode Comparison ===" | tee -a "$LOGFILE"
echo "Config: NP=4, K-Means++, Block distribution" | tee -a "$LOGFILE"

echo "--- Baseline (blocking) ---" | tee -a "$LOGFILE"
mpirun $MPI_FLAGS -np 4 $BINARY $FEATURES $LABELS 1 0 0 2>&1 | tee -a "$LOGFILE"
# Save output files
cp membership_mpi_kmeanspp.csv "$RESULTS_DIR/mem_baseline_np4.csv" 2>/dev/null
cp centroids_mpi_kmeanspp.csv "$RESULTS_DIR/cen_baseline_np4.csv" 2>/dev/null

echo "" | tee -a "$LOGFILE"
echo "--- Non-blocking optimized ---" | tee -a "$LOGFILE"
mpirun $MPI_FLAGS -np 4 $BINARY $FEATURES $LABELS 1 0 1 2>&1 | tee -a "$LOGFILE"
# Save output files
cp membership_mpi_kmeanspp.csv "$RESULTS_DIR/mem_nonblock_np4.csv" 2>/dev/null
cp centroids_mpi_kmeanspp.csv "$RESULTS_DIR/cen_nonblock_np4.csv" 2>/dev/null

# ── Experiment 2: Distribution Strategy Comparison ───────────
# Same config (NP=4, K-Means++, Baseline comm), vary distribution
echo "" | tee -a "$LOGFILE"
echo "=== EXPERIMENT 2: Distribution Strategy Comparison ===" | tee -a "$LOGFILE"
echo "Config: NP=4, K-Means++, Baseline comm" | tee -a "$LOGFILE"

for dist in 0 1 2; do
    dist_name="block"
    [ "$dist" -eq 1 ] && dist_name="cyclic"
    [ "$dist" -eq 2 ] && dist_name="dynamic"
    
    echo "" | tee -a "$LOGFILE"
    echo "--- Distribution: $dist_name ---" | tee -a "$LOGFILE"
    mpirun $MPI_FLAGS -np 4 $BINARY $FEATURES $LABELS 1 $dist 0 2>&1 | tee -a "$LOGFILE"
    cp membership_mpi_kmeanspp.csv "$RESULTS_DIR/mem_${dist_name}_np4.csv" 2>/dev/null
    cp centroids_mpi_kmeanspp.csv "$RESULTS_DIR/cen_${dist_name}_np4.csv" 2>/dev/null
done

# ── Experiment 3: Strong Scaling (fixed data, vary NP) ───────
echo "" | tee -a "$LOGFILE"
echo "=== EXPERIMENT 3: Strong Scaling ===" | tee -a "$LOGFILE"
echo "Config: K-Means++, Block, Baseline comm" | tee -a "$LOGFILE"

for np in 1 2 4 8; do
    echo "" | tee -a "$LOGFILE"
    echo "--- NP=$np ---" | tee -a "$LOGFILE"
    mpirun $MPI_FLAGS -np $np $BINARY $FEATURES $LABELS 1 0 0 2>&1 | tee -a "$LOGFILE"
    cp membership_mpi_kmeanspp.csv "$RESULTS_DIR/mem_block_np${np}.csv" 2>/dev/null
    cp centroids_mpi_kmeanspp.csv "$RESULTS_DIR/cen_block_np${np}.csv" 2>/dev/null
done

# ── Experiment 4: Consistency Check ──────────────────────────
# Run same config 3 times and verify results match
echo "" | tee -a "$LOGFILE"
echo "=== EXPERIMENT 4: Consistency/Reproducibility Check ===" | tee -a "$LOGFILE"
echo "Config: NP=4, K-Means++, Block, Baseline" | tee -a "$LOGFILE"

for run in 1 2 3; do
    echo "--- Run $run ---" | tee -a "$LOGFILE"
    mpirun $MPI_FLAGS -np 4 $BINARY $FEATURES $LABELS 1 0 0 2>&1 | tee -a "$LOGFILE"
    cp membership_mpi_kmeanspp.csv "$RESULTS_DIR/mem_consistency_run${run}.csv" 2>/dev/null
done

echo "" | tee -a "$LOGFILE"
echo "=============================================" | tee -a "$LOGFILE"
echo "All experiments complete. Results in $RESULTS_DIR/" | tee -a "$LOGFILE"
echo "Run: python3 evaluate_metrics.py <mem.csv> <cen.csv> features.csv" | tee -a "$LOGFILE"
echo "  on each output to compute quality metrics." | tee -a "$LOGFILE"
echo "=============================================" | tee -a "$LOGFILE"