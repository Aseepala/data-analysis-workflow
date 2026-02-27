# --- Imports ---
import argparse
import os
import logging
import pandas as pd
from transformers import pipeline  # HuggingFace pipeline wrapper for summarization models

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Model config ---
# BART is a sequence-to-sequence model well-suited for summarization on CPU.
# It acts as a lightweight stand-in for a true SLM like Phi-3-mini.
# Swap to "microsoft/phi-2" or similar when GPU compute is available.
SLM_MODEL = "facebook/bart-large-cnn"  # swap for "microsoft/phi-2" on GPU compute

# Prompt template sent to the model for each issue.
# The {text} placeholder is filled with the raw issue text at runtime.
PROMPT_TEMPLATE = (
    "Summarize the following issue in one short sentence, focusing on the core problem:\n\n{text}"
)


def parse_args():
    # --validated_data:    folder path containing validated_data.csv from the ingest step
    # --text_column:       name of the column containing the raw issue text
    # --extracted_issues:  output folder path where extracted_issues.csv will be written
    parser = argparse.ArgumentParser(description="Issue Extraction via SLM")
    parser.add_argument("--validated_data", type=str, required=True)
    parser.add_argument("--text_column", type=str, required=True)
    parser.add_argument("--extracted_issues", type=str, required=True)
    return parser.parse_args()


def extract_issue_summary(text: str, summarizer) -> str:
    """
    Pass raw issue text through the summarization model to produce a
    concise one-sentence summary. Input is capped at 1024 characters to
    stay within the model's token limit. Falls back to the first 200
    characters of the original text if the model call fails.
    """
    try:
        prompt = PROMPT_TEMPLATE.format(text=text[:1024])  # cap input length
        result = summarizer(prompt, max_length=60, min_length=10, do_sample=False)
        return result[0]["summary_text"].strip()
    except Exception as e:
        logger.warning(f"SLM extraction failed for row: {e}")
        return str(text)[:200]  # graceful fallback: use truncated raw text


def main():
    args = parse_args()

    # --- Load validated data from the ingest step ---
    input_file = os.path.join(args.validated_data, "validated_data.csv")
    logger.info(f"Loading validated data from: {input_file}")
    df = pd.read_csv(input_file)

    # --- Load the summarization model ---
    # The model is downloaded from HuggingFace Hub on first run,
    # then cached on the compute node for subsequent steps.
    logger.info(f"Loading SLM model: {SLM_MODEL}")
    summarizer = pipeline("summarization", model=SLM_MODEL)

    # --- Run summarization on every row ---
    # Each issue's raw text is passed through the model to produce a short
    # summary, stored in a new 'issue_summary' column for the cluster step.
    logger.info(f"Extracting issue summaries for {len(df)} rows...")
    df["issue_summary"] = df[args.text_column].apply(
        lambda text: extract_issue_summary(str(text), summarizer)
    )

    # --- Log a few samples for quick sanity-check in AML Studio ---
    logger.info("Sample extractions:")
    for _, row in df.head(3).iterrows():
        logger.info(f"  Original : {str(row[args.text_column])[:100]}")
        logger.info(f"  Summary  : {row['issue_summary']}")

    # --- Save extracted issues ---
    # The output CSV carries all original columns plus the new 'issue_summary' column
    os.makedirs(args.extracted_issues, exist_ok=True)
    out_path = os.path.join(args.extracted_issues, "extracted_issues.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Extracted issues saved to: {out_path}")


if __name__ == "__main__":
    main()
