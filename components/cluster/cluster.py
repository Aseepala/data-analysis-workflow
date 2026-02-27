# --- Imports ---
import argparse
import os
import logging
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer  # converts text into numeric vectors
from sklearn.cluster import KMeans                     # groups similar vectors together
from sklearn.metrics import silhouette_score           # measures how well-separated clusters are

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Model & clustering config ---
EMBED_MODEL = "all-MiniLM-L6-v2"   # fast, lightweight sentence embedding model (~80MB)
MIN_CLUSTERS = 3                    # minimum number of clusters to try
MAX_CLUSTERS = 15                   # maximum number of clusters to try


def parse_args():
    # --extracted_issues: folder path containing extracted_issues.csv from the extract step
    # --clustered_issues: folder path where clustered output will be written
    parser = argparse.ArgumentParser(description="Issue Clustering")
    parser.add_argument("--extracted_issues", type=str, required=True)
    parser.add_argument("--clustered_issues", type=str, required=True)
    return parser.parse_args()


def find_optimal_clusters(embeddings: np.ndarray, min_k: int, max_k: int) -> int:
    """
    Try every value of k between min_k and max_k and pick the one with the
    highest silhouette score. The silhouette score measures how similar each
    point is to its own cluster vs other clusters — higher is better (range: -1 to 1).
    """
    # Cap max_k to avoid requesting more clusters than data points
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
    """
    Pick a human-readable label to represent this cluster.
    We use the longest summary in the cluster as a proxy for the most descriptive one.
    """
    summaries = cluster_df[text_col].tolist()
    return max(summaries, key=len)


def main():
    args = parse_args()

    # --- Load extracted issues from the previous pipeline step ---
    input_file = os.path.join(args.extracted_issues, "extracted_issues.csv")
    logger.info(f"Loading extracted issues from: {input_file}")
    df = pd.read_csv(input_file)

    # --- Embed issue summaries ---
    # Each summary is converted into a fixed-size numeric vector (embedding).
    # Summaries that are semantically similar will have vectors that are close
    # together in space — which is what K-Means will use to group them.
    summaries = df["issue_summary"].fillna("").tolist()
    logger.info(f"Embedding {len(summaries)} issue summaries with '{EMBED_MODEL}'...")
    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(summaries, show_progress_bar=True, batch_size=32)

    # --- Find the optimal number of clusters ---
    # Rather than hardcoding k, we try a range of values and pick the best one
    # using the silhouette score as our quality metric.
    logger.info("Finding optimal number of clusters...")
    n_clusters = find_optimal_clusters(embeddings, MIN_CLUSTERS, MAX_CLUSTERS)

    # --- Run K-Means with the optimal k ---
    # Each issue is assigned a cluster_id (0 to n_clusters-1)
    logger.info(f"Clustering into {n_clusters} clusters...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster_id"] = km.fit_predict(embeddings)

    # --- Assign a human-readable label to each cluster ---
    # We pick the most descriptive summary from each cluster as its label
    cluster_labels = {}
    for cid in range(n_clusters):
        cluster_df = df[df["cluster_id"] == cid]
        cluster_labels[cid] = get_cluster_label(cluster_df)

    # Map the label back onto every row so the rank step can use it
    df["cluster_label"] = df["cluster_id"].map(cluster_labels)

    # --- Log cluster sizes for visibility in AML Studio ---
    logger.info("Cluster sizes:")
    for cid, label in cluster_labels.items():
        count = (df["cluster_id"] == cid).sum()
        logger.info(f"  Cluster {cid} ({count} issues): {label[:80]}")

    # --- Save outputs ---
    # clustered_issues.csv: full dataset with cluster_id and cluster_label columns added
    # cluster_labels.json: mapping of cluster_id -> label (useful for debugging)
    os.makedirs(args.clustered_issues, exist_ok=True)
    out_csv = os.path.join(args.clustered_issues, "clustered_issues.csv")
    df.to_csv(out_csv, index=False)

    with open(os.path.join(args.clustered_issues, "cluster_labels.json"), "w") as f:
        json.dump(cluster_labels, f, indent=2)

    logger.info(f"Clustered issues saved to: {out_csv}")


if __name__ == "__main__":
    main()
