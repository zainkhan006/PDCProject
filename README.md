# Parallel Soft Clustering for Clinical Notes with OpenMPI
## IBA Karachi — Parallel and Distributed Computing, Spring 2026
### Project 21
Member || Role
Khansa Danish || Core Algorithm & Parallelisation
Ali Hamza || Clinical Text Processing & Feature Engineering
Zain Khan || Metrics, Testing & Validation
Arham Jumshaid || Data Distribution, Load Balancing & Visualisation

# What this project does
Electronic health records contain thousands of unstructured clinical notes. This project clusters them by medical similarity using Fuzzy C-Means (FCM) — a soft clustering algorithm where each note can partially belong to multiple clusters. Because FCM is computationally expensive at scale, the algorithm is parallelised across multiple processors using OpenMPI.
The full pipeline:

Raw clinical notes → numerical feature vectors (Python, NLP)
Parallel FCM clustering (C + OpenMPI)
Cluster validation, scaling analysis, and clinical interpretation (Python)

**To run the project:**


**1. Setup build**
```bash

make clean && make
make run
```

**2. Run all three init strategies (Milestone 1 validation)**
```bash
mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 0 0 0
mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 1 0 0
mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 2 0 0
```

**3. SNOMED validation**
```bash
python3 snomed_validate.py \
    --top_terms  viz_top_terms.csv \
    --feat_names feature_names.csv \
    --output     cluster_snomed_map.csv
```

**4. Clustering quality metrics **
```bash
python3 evaluate_metrics.py \
    --membership membership_mpi_kmeanspp.csv \
    --centroids  centroids_mpi_kmeanspp.csv  \
    --features   features.csv               \
    --labels     specialty_labels.csv        \
    --sample     2000
```

**5. Strong scaling — P=1,2,4,8 fixed dataset**
```bash
for P in 1 2 4 8; do
    echo "========== P=$P =========="
    mpirun --oversubscribe -np $P ./fcm_mpi features.csv specialty_labels.csv 1 0 0
done
```

**6. Baseline vs non-blocking communication comparison **
```bash
echo "=== BASELINE P=4 ===" && mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 1 0 0
echo "=== NONBLOCK P=4 ===" && mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 1 0 1
```

**7. Block vs cyclic vs dynamic distribution comparison **
```bash
echo "=== BLOCK ===" && mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 1 0 0
echo "=== CYCLIC ===" && mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 1 1 0
echo "=== DYNAMIC ===" && mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 1 2 0
```

**8. Weak scaling **
```bash
python3 generate_weak_data.py --suite features.csv

mpirun --oversubscribe -np 1 ./fcm_mpi features_weak_P1.csv specialty_labels.csv 1 0 0
mpirun --oversubscribe -np 2 ./fcm_mpi features_weak_P2.csv specialty_labels.csv 1 0 0
mpirun --oversubscribe -np 4 ./fcm_mpi features.csv           specialty_labels.csv 1 0 0
mpirun --oversubscribe -np 8 ./fcm_mpi features_weak_P8.csv specialty_labels.csv 1 0 0
```

**9. Varying cluster counts k=5,10,20,50 **
```bash
for K in 5 10 20 50; do
    sed -i "s/#define N_CLUSTERS.*/#define N_CLUSTERS   $K/" fcm_mpi.h
    make clean && make -s
    echo "=== k=$K ===" && mpirun --oversubscribe -np 4 ./fcm_mpi features.csv specialty_labels.csv 1 0 0
done
sed -i 's/#define N_CLUSTERS.*/#define N_CLUSTERS   10/' fcm_mpi.h
make clean && make
```

**10. Domain expert evaluation **
```bash
make run   # ensures membership_mpi_kmeanspp.csv exists

python3 domain_expert_evaluation.py \
    --auto \
    --top_terms  viz_top_terms.csv \
    --membership membership_mpi_kmeanspp.csv \
    --labels     specialty_labels.csv
```

