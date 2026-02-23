import argparse
import os
import logging
from typing import List, Dict
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Issue Ranking & Report Generation")
    parser.add_argument("--clustered_issues", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--final_report", type=str, required=True)
    return parser.parse_args()


def build_top_issues(df: pd.DataFrame, top_n: int) -> List[Dict]:
    """
    Rank clusters by issue count (frequency) and build a structured top-issues list.
    Each entry includes: rank, cluster label, count, % of total, and example issues.
    """
    total = len(df)
    cluster_stats = (
        df.groupby("cluster_id")
        .agg(
            count=("cluster_id", "size"),
            cluster_label=("cluster_label", "first"),
        )
        .sort_values("count", ascending=False)
        .reset_index()
    )

    top_issues = []
    for rank, row in enumerate(cluster_stats.head(top_n).itertuples(), start=1):
        examples = (
            df[df["cluster_id"] == row.cluster_id]["issue_summary"]
            .dropna()
            .head(3)
            .tolist()
        )
        top_issues.append(
            {
                "rank": rank,
                "cluster_id": int(row.cluster_id),
                "issue_theme": row.cluster_label,
                "count": int(row.count),
                "percentage": round(row.count / total * 100, 1),
                "example_issues": examples,
            }
        )

    return top_issues


def main():
    args = parse_args()

    input_csv = os.path.join(args.clustered_issues, "clustered_issues.csv")
    logger.info(f"Loading clustered issues from: {input_csv}")
    df = pd.read_csv(input_csv)

    logger.info(f"Building top {args.top_n} issues...")
    top_issues = build_top_issues(df, args.top_n)

    for issue in top_issues:
        logger.info(
            f"  #{issue['rank']} ({issue['count']} issues, {issue['percentage']}%): "
            f"{issue['issue_theme'][:80]}"
        )

    # Log metrics to AML run via print (picked up by AML logging agent)
    logger.info(f"total_issues={int(df.shape[0])}")
    logger.info(f"total_clusters={int(df['cluster_id'].nunique())}")
    for issue in top_issues:
        logger.info(f"rank_{issue['rank']}_count={issue['count']}")

    # Save outputs
    os.makedirs(args.final_report, exist_ok=True)

    # Build flat CSV - one row per ranked issue cluster
    report_rows = []
    for issue in top_issues:
        report_rows.append({
            "rank":           issue["rank"],
            "issue_theme":    issue["issue_theme"],
            "count":          issue["count"],
            "percentage":     issue["percentage"],
            "example_issues": "; ".join(issue["example_issues"]),
        })

    report_df = pd.DataFrame(report_rows)
    csv_path = os.path.join(args.final_report, "top_issues.csv")
    report_df.to_csv(csv_path, index=False)
    logger.info(f"Top issues dataset saved to: {csv_path}")


if __name__ == "__main__":
    main()
