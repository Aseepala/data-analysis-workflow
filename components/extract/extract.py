import argparse
import os
import logging
import pandas as pd
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# SLM model - using a small summarization model suitable for CPU compute
SLM_MODEL = "facebook/bart-large-cnn"  # swap for "microsoft/phi-2" on GPU compute

PROMPT_TEMPLATE = (
    "Summarize the following issue in one short sentence, focusing on the core problem:\n\n{text}"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Issue Extraction via SLM")
    parser.add_argument("--validated_data", type=str, required=True)
    parser.add_argument("--text_column", type=str, required=True)
    parser.add_argument("--extracted_issues", type=str, required=True)
    return parser.parse_args()


def extract_issue_summary(text: str, summarizer) -> str:
    """Use the SLM to extract a concise issue summary from raw text."""
    try:
        prompt = PROMPT_TEMPLATE.format(text=text[:1024])  # cap input length
        result = summarizer(prompt, max_length=60, min_length=10, do_sample=False)
        return result[0]["summary_text"].strip()
    except Exception as e:
        logger.warning(f"SLM extraction failed for row: {e}")
        return str(text)[:200]


def main():
    args = parse_args()

    input_file = os.path.join(args.validated_data, "validated_data.csv")
    logger.info(f"Loading validated data from: {input_file}")
    df = pd.read_csv(input_file)

    logger.info(f"Loading SLM model: {SLM_MODEL}")
    summarizer = pipeline("summarization", model=SLM_MODEL)

    logger.info(f"Extracting issue summaries for {len(df)} rows...")
    df["issue_summary"] = df[args.text_column].apply(
        lambda text: extract_issue_summary(str(text), summarizer)
    )

    logger.info("Sample extractions:")
    for _, row in df.head(3).iterrows():
        logger.info(f"  Original : {str(row[args.text_column])[:100]}")
        logger.info(f"  Summary  : {row['issue_summary']}")

    os.makedirs(args.extracted_issues, exist_ok=True)
    out_path = os.path.join(args.extracted_issues, "extracted_issues.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Extracted issues saved to: {out_path}")


if __name__ == "__main__":
    main()
