import argparse
import os
import logging
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EMBED_MODEL = "all-MiniLM-L6-v2"   # fast, lightweight sentence embedding model
MIN_CLUSTERS = 3
MAX_CLUSTERS = 15


def parse_args():
    parser = argparse.ArgumentParser(description="Issue Clustering")
    parser.add_argument("--extracted_issues", type=str, required=True)
    parser.add_argument("--clustered_issues", type=str, required=True)
    return parser.parse_args()


def find_optimal_clusters(embeddings: np.ndarray, min_k: int, max_k: int) -> int:
    """Use silhouette score to find the best number of clusters."""
    max_k = min(max_k, len(embeddings) - 1)
    best_k, best_score = min_k, -1
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        logger.info(f"  k={k}, silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k = score, k
    logger.info(f"Optimal clusters: {best_k} (silhouette={best_score:.4f})")
    return best_k


def get_cluster_label(cluster_df: pd.DataFrame, text_col: str = "issue_summary") -> str:
    """Pick the most representative (most common keywords) summary as the cluster label."""
    summaries = cluster_df[text_col].tolist()
    # Use the longest summary as cluster representative label
    return max(summaries, key=len)


def main():
    args = parse_args()

    input_file = os.path.join(args.extracted_issues, "extracted_issues.csv")
    logger.info(f"Loading extracted issues from: {input_file}")
    df = pd.read_csv(input_file)

    summaries = df["issue_summary"].fillna("").tolist()
    logger.info(f"Embedding {len(summaries)} issue summaries with '{EMBED_MODEL}'...")

    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(summaries, show_progress_bar=True, batch_size=32)

    logger.info("Finding optimal number of clusters...")
    n_clusters = find_optimal_clusters(embeddings, MIN_CLUSTERS, MAX_CLUSTERS)

    logger.info(f"Clustering into {n_clusters} clusters...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster_id"] = km.fit_predict(embeddings)

    # Assign a human-readable label per cluster
    cluster_labels = {}
    for cid in range(n_clusters):
        cluster_df = df[df["cluster_id"] == cid]
        cluster_labels[cid] = get_cluster_label(cluster_df)

    df["cluster_label"] = df["cluster_id"].map(cluster_labels)

    logger.info("Cluster sizes:")
    for cid, label in cluster_labels.items():
        count = (df["cluster_id"] == cid).sum()
        logger.info(f"  Cluster {cid} ({count} issues): {label[:80]}")

    os.makedirs(args.clustered_issues, exist_ok=True)
    out_csv = os.path.join(args.clustered_issues, "clustered_issues.csv")
    df.to_csv(out_csv, index=False)

    with open(os.path.join(args.clustered_issues, "cluster_labels.json"), "w") as f:
        json.dump(cluster_labels, f, indent=2)

    logger.info(f"Clustered issues saved to: {out_csv}")


if __name__ == "__main__":
    main()
