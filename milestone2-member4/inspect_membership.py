import csv
from pathlib import Path
p = Path("membership_mpi_kmeanspp.csv")
rows = []
with p.open(newline="") as f:
    r = csv.reader(f)
    for row in r:
        if row:
            rows.append(row)
print("FIRST_3_ROWS_START")
for row in rows[:3]:
    print(",".join(row))
print("FIRST_3_ROWS_END")
first200 = rows[:200]
print(f"FIRST200_TOTAL={len(first200)}")
print(f"FIRST200_UNIQUE_EXACT={len(set(','.join(r) for r in first200))}")
if first200:
    cols = list(zip(*[[float(x) for x in r] for r in first200]))
    any_spread = False
    for i, c in enumerate(cols):
        cmin = min(c)
        cmax = max(c)
        spread = cmax - cmin
        if spread > 0:
            any_spread = True
        print(f"COL{i}_MIN={cmin:.17g} MAX={cmax:.17g} SPREAD={spread:.17g}")
    print(f"ANY_COLUMN_SPREAD_GT_0={any_spread}")
else:
    print("ANY_COLUMN_SPREAD_GT_0=False")
