import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from data.config import SIMILARS, UPSTREAM_TARGETS
from data.processing import (
    filter_upstream_by_similarity_to_downstream,
    generate_dataset,
    scaffold_split,
)


def main(
    filter_similar_targets: bool = False,
    similarity_thershold: Optional[float] = None,
    seed: Optional[int] = 42,
):
    sanitized_dir = Path("data/sanitized")
    upstream_csv = sanitized_dir / "upstream_full_raw.csv"
    upstream_targets = UPSTREAM_TARGETS

    if filter_similar_targets:
        upstream_csv = sanitized_dir / "upstream_filtered_raw.csv"

    if not upstream_csv.exists():
        if filter_similar_targets:
            for target in SIMILARS:
                del upstream_targets[target]

        generate_dataset(upstream_targets, upstream_csv.stem, data_dir=sanitized_dir)

    if similarity_thershold is not None:
        downstream_csv = sanitized_dir / f"downstream_raw.csv"
        if not downstream_csv.exists():
            raise Exception(
                "Please if you want to use similarity filtering, generate downstream data first"
            )
        downstream_df = pd.read_csv(downstream_csv)
        upstream_df = pd.read_csv(upstream_csv)
        upstream_df = filter_upstream_by_similarity_to_downstream(
            upstream_df, downstream_df, similarity_thershold
        )
    else:
        upstream_df = pd.read_csv(upstream_csv)

    splits_path = Path(
        "data",
        "splits",
        "upstream",
        "filtered" if filter_similar_targets else "full",
        str(similarity_thershold),
    )
    splits_path.mkdir(exist_ok=True, parents=True)

    train_df, val_df, test_df = scaffold_split(upstream_df, seed=seed)
    train_df.to_csv(splits_path / f"{upstream_csv.stem}_train.csv")
    val_df.to_csv(splits_path / f"{upstream_csv.stem}_val.csv")
    test_df.to_csv(splits_path / f"{upstream_csv.stem}_test.csv")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and split upstream dataset.")

    # Boolean flag: including --filter_similar_targets sets it to True
    parser.add_argument(
        "--filter_similar_targets",
        action="store_true",
        help="Filter out similar targets from the upstream dataset.",
    )

    # Float argument for threshold
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=None,
        help="Similarity threshold for filtering upstream by downstream (float).",
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
    main(
        filter_similar_targets=args.filter_similar_targets,
        similarity_thershold=args.similarity_threshold,
        seed=args.seed,
    )
