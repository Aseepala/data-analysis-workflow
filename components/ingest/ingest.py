# --- Imports ---
import argparse
import os
import logging
import pandas as pd  # tabular data loading and manipulation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    # --raw_data:       path to the input CSV file (e.g. issues.csv from the datastore)
    # --text_column:    name of the column containing the raw issue text to analyse
    # --validated_data: output folder path where validated_data.csv will be written
    parser = argparse.ArgumentParser(description="Data Ingestion & Validation")
    parser.add_argument("--raw_data", type=str, required=True)
    parser.add_argument("--text_column", type=str, required=True)
    parser.add_argument("--validated_data", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load raw data ---
    # The raw_data path is provided by the Azure ML pipeline as a mounted URI
    logger.info(f"Loading raw data from: {args.raw_data}")
    df = pd.read_csv(args.raw_data)
    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    # --- Validate the text column exists ---
    # Fail early with a clear message if the expected column isn't in the data
    if args.text_column not in df.columns:
        raise ValueError(
            f"Text column '{args.text_column}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )

    # --- Drop rows with empty text ---
    # Empty rows would produce meaningless summaries in the extract step
    before = len(df)
    df = df.dropna(subset=[args.text_column])
    df = df[df[args.text_column].astype(str).str.strip() != ""]
    after = len(df)
    logger.info(f"Dropped {before - after} rows with empty text. Remaining: {after}")

    # --- Remove exact duplicate rows ---
    # Duplicate issues would inflate cluster counts and skew the ranking
    df = df.drop_duplicates()
    logger.info(f"After dedup: {len(df)} rows")

    # --- Save validated output ---
    # downstream steps (extract, cluster, rank) all read from this folder
    os.makedirs(args.validated_data, exist_ok=True)
    out_path = os.path.join(args.validated_data, "validated_data.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Validated data saved to: {out_path}")


if __name__ == "__main__":
    main()
