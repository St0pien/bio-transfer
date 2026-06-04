# scripts/generate_all_downstream_splits.py

import subprocess
from itertools import product

SUBSETS = [
    "0.089799", # 500
    "0.017960", # 100
    "0.008980", # 50
    "0.003592", # 20
    "0.001796", # 10
]
SEEDS = [42, 123, 2137]

BASE_CMD = [
    "python3",
    "/net/tscratch/people/plgvltkv/bio-transfer/scripts/generate_downstream_dataset.py",
]

for subset, seed in product(SUBSETS, SEEDS):
    cmd = BASE_CMD.copy()

    cmd.extend(["--seed", str(seed)])

    if subset is not None:
        cmd.extend(["--subset", str(subset)])

    print("=" * 80)
    print("RUNNING:", " ".join(cmd))
    print("=" * 80)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[FAILED] subset={subset}, seed={seed}")
    else:
        print(f"[DONE] subset={subset}, seed={seed}")