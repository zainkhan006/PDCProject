#!/bin/bash
# Run Member 4 FCM for all np x distribution combinations
cd /mnt/c/Users/29235/IBA/PDC/fcm_milestone2/fcm_milestone2
LOG=./run_logs_member4_latest
mkdir -p "$LOG"

for dist in 0 1 2; do
    if [ $dist -eq 0 ]; then
        dname="block"
    elif [ $dist -eq 1 ]; then
        dname="cyclic"
    else
        dname="dynamic"
    fi

    for np in 1 2 4 8; do
        echo "=== np=$np dist=$dname ==="
        mpirun --oversubscribe -np $np ./fcm_mpi features.csv specialty_labels.csv 1 $dist 0 \
            2>&1 | tee "$LOG/run_np${np}_${dname}.txt" \
            | grep -E "\[fcm_mpi\] (N=|Converged|Reached|Total|Avg|Final)|\[main\] FCM"
        echo "---"
    done
done

echo "All runs complete."
