import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from data.config import DOWNSTREAM_TARGETS
from data.processing import (
    filter_upstream_by_similarity_to_downstream,
    generate_dataset,
    scaffold_split,
)


def main(
    subset: Optional[float] = None,
    seed: Optional[int] = 42,
):
    sanitized_dir = Path("data/sanitized")
    downstream_csv = sanitized_dir / "downstream_full_raw.csv"

    if not downstream_csv.exists():
        generate_dataset(
            DOWNSTREAM_TARGETS, downstream_csv.stem, data_dir=sanitized_dir
        )

    downstream_df_full = pd.read_csv(downstream_csv)

    downstream_dfs = [
        group.copy() for _, group in downstream_df_full.groupby("target_chembl_id")
    ]

    for df in downstream_dfs:
        train_df, val_df, test_df = scaffold_split(df, 0.6, 0.2)

        if subset is not None:
            train_df = train_df.sample(frac=subset, random_state=seed).reset_index(drop=True)

        save_dir = Path("data", "splits", "downstream", str(seed))
        save_dir.mkdir(parents=True, exist_ok=True)

        val_df.to_csv(save_dir / f"{train_df['target_name'][0]}_val.csv")
        test_df.to_csv(save_dir / f"{train_df['target_name'][0]}_test.csv")

        train_dir = save_dir / (str(subset) if subset is not None else "full")
        train_dir.mkdir(exist_ok=True)
        train_df.to_csv(train_dir / f"{train_df['target_name'][0]}_train.csv")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and split upstream dataset.")

    # Float argument for threshold
    parser.add_argument(
        "--subset",
        type=float,
        default=None,
        help="Fraction of a full train set to take",
    )

    # Integer argument for seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for scaffold splitting (default: 42).",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(subset=args.subset, seed=args.seed)
