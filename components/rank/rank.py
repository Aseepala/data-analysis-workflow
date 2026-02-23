import argparse
import os
import json
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


def generate_markdown(top_issues: List[Dict], top_n: int) -> str:
    lines = [
        f"# Top {top_n} Issues Report\n",
        f"**Total issues analysed:** {sum(i['count'] for i in top_issues)}\n",
        "---\n",
    ]
    for issue in top_issues:
        lines.append(f"## #{issue['rank']} — {issue['issue_theme']}")
        lines.append(f"- **Count:** {issue['count']} issues ({issue['percentage']}% of total)")
        lines.append("- **Examples:**")
        for ex in issue["example_issues"]:
            lines.append(f"  - _{ex}_")
        lines.append("")
    return "\n".join(lines)


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

    json_path = os.path.join(args.final_report, "top_issues.json")
    with open(json_path, "w") as f:
        json.dump({"top_issues": top_issues}, f, indent=2)
    logger.info(f"JSON report saved to: {json_path}")

    md_path = os.path.join(args.final_report, "top_issues_report.md")
    with open(md_path, "w") as f:
        f.write(generate_markdown(top_issues, args.top_n))
    logger.info(f"Markdown report saved to: {md_path}")


if __name__ == "__main__":
    main()
