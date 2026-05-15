import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data.downloading import fetch_targets_sequences


def main():
    parser = argparse.ArgumentParser(
        description="Fetch protein sequences for unique target IDs from a CSV file."
    )

    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to input CSV file",
    )

    parser.add_argument(
        "--target-column",
        type=str,
        default="target_chembl_id",
        help="Column name containing target IDs",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to output JSON file",
    )

    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.csv_path)

    # Validate column
    if args.target_column not in df.columns:
        raise ValueError(
            f"Column '{args.target_column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    # Extract unique target IDs
    target_ids = df[args.target_column].unique().tolist()

    print(f"Found {len(target_ids)} unique target IDs")

    # Fetch sequences
    target_sequences = fetch_targets_sequences(target_ids)

    # Ensure output directory exists
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with open(output_path, "w") as f:
        json.dump(target_sequences, f, indent=2)

    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
