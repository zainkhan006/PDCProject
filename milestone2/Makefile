# ============================================================
# Makefile — Parallel FCM with OpenMPI
# Member 1: Khansa Danish | IBA Karachi, Spring 2026
#
# Usage:
#   make              — build ./fcm_mpi
#   make run          — run with 4 processes (K-Means++)
#   make run_random   — run with 4 processes (Random)
#   make run_domain   — run with 4 processes (Domain-guided)
#   make run_np NP=8  — run with 8 processes
#   make clean        — remove compiled files
# ============================================================

CC      = mpicc
CFLAGS  = -Wall -Wextra -O2 -std=c99
LIBS    = -lm
TARGET  = fcm_mpi
SRCS    = main_mpi.c fcm_mpi.c
OBJS    = $(SRCS:.c=.o)

# Default number of processes
NP      = 4

# Member 2's data files (must be in same directory)
FEATURES = features.csv
LABELS   = specialty_labels.csv

# ── Build ───────────────────────────────────────────────────
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)
	@echo "Build successful → ./$(TARGET)"

%.o: %.c fcm_mpi.h
	$(CC) $(CFLAGS) -c $< -o $@

# ── Run targets ─────────────────────────────────────────────
run: all
	mpirun -np $(NP) ./$(TARGET) $(FEATURES) $(LABELS) 1

run_random: all
	mpirun -np $(NP) ./$(TARGET) $(FEATURES) $(LABELS) 0

run_domain: all
	mpirun -np $(NP) ./$(TARGET) $(FEATURES) $(LABELS) 2

run_np: all
	mpirun -np $(NP) ./$(TARGET) $(FEATURES) $(LABELS) 1

# ── Clean ───────────────────────────────────────────────────
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all run run_random run_domain run_np clean
