# ─────────────────────────────────────────────────────────────────────────────
# Makefile  —  Parallel FCM for Clinical Notes
# PDC Project 21 | IBA Karachi Spring 2026
#
# Usage:
#   make                  # build
#   make run              # K-Means++, 4 processes, block, baseline comm
#   make run_random       # Random init, 4 processes
#   make run_domain       # Domain-guided, 4 processes
#   make run_cyclic       # K-Means++, 4 processes, cyclic distribution
#   make run_dynamic      # K-Means++, 4 processes, dynamic distribution
#   make run_nonblock     # K-Means++, 4 processes, non-blocking comm
#   make run_np NP=8      # K-Means++, 8 processes
#   make scaling          # full strong scaling sweep P=1,2,4,8
#   make weak_scaling     # weak scaling sweep (requires synth data scripts)
#   make vary_k           # vary cluster count k=5,10,20,50
#   make clean
# ─────────────────────────────────────────────────────────────────────────────

CC       = mpicc
CFLAGS   = -O2 -Wall -Wextra -std=c99
LDFLAGS  = -lm
TARGET   = fcm_mpi
SRCS     = fcm_mpi.c main_mpi.c
HDRS     = fcm_mpi.h

# Default data files (place in same directory)
FEAT     = features.csv
LABELS   = specialty_labels.csv
NP      ?= 4

# Init:  0=random  1=kmeanspp  2=domain
# Dist:  0=block   1=cyclic    2=dynamic
# Comm:  0=baseline 1=nonblock

.PHONY: all clean run run_random run_domain run_cyclic run_dynamic \
        run_nonblock scaling weak_scaling vary_k check_data

all: $(TARGET)
	@echo "Build successful --> ./$(TARGET)"

$(TARGET): $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

# ── Standard runs ────────────────────────────────────────────────────────────
run: $(TARGET) check_data
	mpirun --oversubscribe -np $(NP) ./$(TARGET) $(FEAT) $(LABELS) 1 0 0

run_random: $(TARGET) check_data
	mpirun --oversubscribe -np $(NP) ./$(TARGET) $(FEAT) $(LABELS) 0 0 0

run_domain: $(TARGET) check_data
	mpirun --oversubscribe -np $(NP) ./$(TARGET) $(FEAT) $(LABELS) 2 0 0

run_cyclic: $(TARGET) check_data
	mpirun --oversubscribe -np $(NP) ./$(TARGET) $(FEAT) $(LABELS) 1 1 0

run_dynamic: $(TARGET) check_data
	mpirun --oversubscribe -np $(NP) ./$(TARGET) $(FEAT) $(LABELS) 1 2 0

run_nonblock: $(TARGET) check_data
	mpirun --oversubscribe -np $(NP) ./$(TARGET) $(FEAT) $(LABELS) 1 0 1

run_np: $(TARGET) check_data
	mpirun --oversubscribe -np $(NP) ./$(TARGET) $(FEAT) $(LABELS) 1 0 0

# ── Strong scaling sweep ──────────────────────────────────────────────────────
scaling: $(TARGET) check_data
	@echo "====== STRONG SCALING SWEEP (K-Means++, block, baseline) ======"
	@for P in 1 2 4 8; do \
		echo "--- P=$$P ---"; \
		mpirun --oversubscribe -np $$P ./$(TARGET) $(FEAT) $(LABELS) 1 0 0; \
	done

# ── Distribution strategy comparison ─────────────────────────────────────────
dist_compare: $(TARGET) check_data
	@echo "====== DISTRIBUTION STRATEGY COMPARISON (P=$(NP)) ======"
	@echo "--- Block ---"
	mpirun --oversubscribe -np $(NP) ./$(TARGET) $(FEAT) $(LABELS) 1 0 0
	@echo "--- Cyclic ---"
	mpirun --oversubscribe -np $(NP) ./$(TARGET) $(FEAT) $(LABELS) 1 1 0
	@echo "--- Dynamic ---"
	mpirun --oversubscribe -np $(NP) ./$(TARGET) $(FEAT) $(LABELS) 1 2 0

# ── Communication mode comparison ────────────────────────────────────────────
comm_compare: $(TARGET) check_data
	@echo "====== COMMUNICATION MODE COMPARISON (P=$(NP)) ======"
	@echo "--- Baseline ---"
	mpirun --oversubscribe -np $(NP) ./$(TARGET) $(FEAT) $(LABELS) 1 0 0
	@echo "--- Non-blocking ---"
	mpirun --oversubscribe -np $(NP) ./$(TARGET) $(FEAT) $(LABELS) 1 0 1

# ── Vary cluster count (requires editing N_CLUSTERS in header) ───────────────
# Automated via temporary copies
vary_k: $(SRCS) $(HDRS) check_data
	@echo "====== VARY CLUSTER COUNT k=5,10,20,50 ======"
	@for K in 5 10 20 50; do \
		echo "--- k=$$K ---"; \
		sed "s/#define N_CLUSTERS.*/#define N_CLUSTERS   $$K/" fcm_mpi.h > fcm_mpi_tmp.h; \
		$(CC) $(CFLAGS) -o fcm_mpi_k$$K \
			fcm_mpi.c main_mpi.c -include fcm_mpi_tmp.h $(LDFLAGS) 2>/dev/null || \
		$(CC) $(CFLAGS) -DNCLUST=$$K -o fcm_mpi_k$$K $(SRCS) $(LDFLAGS); \
		mpirun --oversubscribe -np 4 ./fcm_mpi_k$$K $(FEAT) $(LABELS) 1 0 0; \
		rm -f fcm_mpi_k$$K fcm_mpi_tmp.h; \
	done

# ── Weak scaling (uses generate_weak_data.py to create subsampled datasets) ──
weak_scaling: $(TARGET) check_data
	@echo "====== WEAK SCALING SWEEP ======"
	@echo "--- P=1 N=1236 ---"
	python3 generate_weak_data.py $(FEAT) 1236 features_weak_1.csv 2>/dev/null || \
		head -1236 $(FEAT) > features_weak_1.csv
	mpirun --oversubscribe -np 1 ./$(TARGET) features_weak_1.csv $(LABELS) 1 0 0
	@echo "--- P=2 N=2472 ---"
	python3 generate_weak_data.py $(FEAT) 2472 features_weak_2.csv 2>/dev/null || \
		head -2472 $(FEAT) > features_weak_2.csv
	mpirun --oversubscribe -np 2 ./$(TARGET) features_weak_2.csv $(LABELS) 1 0 0
	@echo "--- P=4 N=4943 ---"
	mpirun --oversubscribe -np 4 ./$(TARGET) $(FEAT) $(LABELS) 1 0 0

# ── Check data files exist ────────────────────────────────────────────────────
check_data:
	@test -f $(FEAT)   || (echo "ERROR: $(FEAT) not found. Copy Member 2 data here." && false)
	@test -f $(LABELS) || (echo "ERROR: $(LABELS) not found. Copy Member 2 data here." && false)

clean:
	rm -f $(TARGET) fcm_mpi_k* fcm_mpi_tmp.h \
	      membership_mpi_*.csv centroids_mpi_*.csv \
	      viz_top_terms.csv viz_membership_sample.csv \
	      features_weak_*.csv
	@echo "Clean done."
