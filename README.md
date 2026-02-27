# M365 Copilot Top Issues Analysis Pipeline

This project runs an Azure ML pipeline to identify the top recurring issues from M365 Copilot user feedback. It uses a transformer-based summarization model (BART) to extract concise issue summaries, K-Means clustering to group similar issues, and frequency ranking to surface what matters most — replacing slow manual analysis with an automated, scalable pipeline.

> 📓 **Want to explore interactively?** Open [`notebook/Data-analysis.ipynb`](notebook/Data-analysis.ipynb) to run a lightweight data exploration of the dataset on AzureML Notebooks. It covers data loading, product area breakdowns, source distribution, trends over time, and a heatmap cross-analysis.

---

## Project Structure

```
data-analysis/
├── run_pipeline.py             # Main pipeline script — submits the AML job
├── config.py                   # Environment and pipeline configuration
├── test_data/
│   └── issues.csv              # Synthetic M365 Copilot issues dataset (40 rows)
├── notebook/
│   └── Data-analysis.ipynb     # Interactive data exploration notebook (local, no AML)
├── environments/
│   └── conda.yml               # Shared conda environment for all pipeline components
└── components/
    ├── ingest/
    │   ├── component.yml
    │   └── ingest.py           # Data ingestion & validation
    ├── extract/
    │   ├── component.yml
    │   └── extract.py          # Issue summarisation via BART (swap for Phi on GPU)
    ├── cluster/
    │   ├── component.yml
    │   └── cluster.py          # Issue clustering via sentence embeddings + K-Means
    └── rank/
        ├── component.yml
        └── rank.py             # Issue ranking & CSV report generation
```

---

## Prerequisites

- Python 3.9+
- Install required packages: `pip install azure-ai-ml azure-identity`
- Any Heron environment with an AML workspace provisioned

---

## Configuration

All environment and pipeline settings are in `config.py`. Update these values to match your Azure resources before running. All configuration values can be found on the Heron environment's AML card in AI Foundry.

```python
ENVIRONMENT = {
    "compute_name": "lin-cpu-00",       # AML compute cluster name
    "tenant_id": "...",                 # Torus tenant ID
    "azure_ml_account": "...",          # AML workspace name
    "azure_subscription_id": "...",     # Azure subscription ID
    "azure_resource_group": "...",      # Azure resource group
}

PIPELINE = {
    "experiment_name": "top-issues-analysis",
    "input_data": "test_data/issues.csv",  # local path or azureml:dataset@latest
    "text_column": "description",          # column containing issue text
    "top_n": 10,                           # number of top issues to surface
    "output_path": "top-issues-results/",  # path within heron_sandbox_storage
}
```

---

## Input Data

Local sample data is read from:

- `test_data/issues.csv` — 40 sample M365 Copilot issues across Word, Teams, Outlook, Excel, PowerPoint, and Business Chat

The CSV must contain at least a text column (default: `description`):

```csv
id,description,source,date_reported,product_area
1,"Copilot in Word keeps generating text in the wrong language",UserVoice,2025-11-01,Word
```

To use a registered AML dataset instead, update `config.py`:
```python
"input_data": "azureml:issues-dataset@latest"
```

---

## Running the Pipeline

```
python run_pipeline.py
```

A browser window will open prompting you to sign in with your MLPA account.

On success, the script prints the Azure ML Studio URL where you can monitor the job:

```
Pipeline submitted! Track at: https://ml.azure.com/runs/...
```

---

## Pipeline Steps

The pipeline runs 4 steps in sequence:

1. **Ingest** (`ingest.py`) — Loads the CSV, validates the text column exists, drops empty and duplicate rows, and outputs a clean dataset
2. **Extract** (`extract.py`) — Summarises each issue into one concise sentence using `facebook/bart-large-cnn` (a CPU-friendly summarization model; swap for `microsoft/phi-2` on GPU compute for a true SLM)
3. **Cluster** (`cluster.py`) — Embeds all summaries using `all-MiniLM-L6-v2`, auto-selects the optimal number of clusters via silhouette score, and groups similar issues using K-Means
4. **Rank** (`rank.py`) — Sorts clusters by frequency, picks the top N, and saves a structured CSV report

**How it flows:**

> Raw CSV → *(clean & validate)* → *(summarise)* → *(embed & cluster)* → *(rank by frequency)* → **Top issues CSV**

---

## Output

On completion, the pipeline writes the following file to `heron_sandbox_storage`:

### `top_issues.csv`
A ranked CSV report — one row per issue cluster:

```csv
rank,issue_theme,count,percentage,example_issues
1,"Copilot meeting summaries missing action items",8,20.0,"Meeting recap attributes quotes to wrong speakers; ..."
2,...
```

The output path is configured in `config.py` under `PIPELINE["output_path"]`.
