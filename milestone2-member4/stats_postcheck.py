import csv, math, itertools
from pathlib import Path

cent_path = Path("centroids_mpi_kmeanspp.csv")
mem_path = Path("membership_mpi_kmeanspp.csv")

cent_rows = []
with cent_path.open(newline="") as f:
    for row in csv.reader(f):
        if row:
            cent_rows.append(tuple(float(x) for x in row))

cent_unique = len(set(cent_rows))
if len(cent_rows) < 2:
    cent_min_l2 = 0.0
else:
    cent_min_l2 = min(
        math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
        for a,b in itertools.combinations(cent_rows,2)
    )

print(f"CENTROID_ROWS={len(cent_rows)}")
print(f"CENTROID_UNIQUE_ROWS={cent_unique}")
print(f"CENTROID_MIN_PAIRWISE_L2={cent_min_l2:.12g}")

mem_rows = []
with mem_path.open(newline="") as f:
    for row in csv.reader(f):
        if row:
            mem_rows.append(tuple(float(x) for x in row))

mem_unique = len(set(mem_rows))
max_abs_dev_01 = 0.0
for row in mem_rows:
    for v in row:
        d = abs(v - 0.1)
        if d > max_abs_dev_01:
            max_abs_dev_01 = d

print(f"MEMBERSHIP_ROWS={len(mem_rows)}")
print(f"MEMBERSHIP_UNIQUE_ROWS={mem_unique}")
print(f"MEMBERSHIP_MAX_ABS_DEV_FROM_0.1={max_abs_dev_01:.12g}")
print("MEMBERSHIP_FIRST_3_ROWS_START")
for row in mem_rows[:3]:
    print(",".join(f"{v:.12g}" for v in row))
print("MEMBERSHIP_FIRST_3_ROWS_END")