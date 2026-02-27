# --- Imports ---
# Standard library and Azure SDK imports for file handling, authentication,
# Azure ML pipeline orchestration, and project configuration.
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient, dsl, Input, Output, load_component
from azure.ai.ml.constants import AssetTypes
from config import ENVIRONMENT, PIPELINE

# --- Global component variables ---
# Populated in main() before the pipeline function is called.
ingest_data = None
extract_issues = None
cluster_issues = None
rank_issues = None


# --- Pipeline definition ---
# This function defines the pipeline DAG (directed acyclic graph).
# It does NOT execute anything — it just describes the steps and how
# data flows between them. AML runs them later in the cloud.
@dsl.pipeline(
    name="top_issues_analysis_pipeline",
    description="Analyses M365 Copilot issues using an SLM to extract, cluster and rank the top issues",
    default_datastore="heron_sandbox_storage"
)
def top_issues_pipeline(
    raw_data: Input,          # The input CSV file from the AML datastore
    text_column: str,         # Column name containing raw issue text (e.g. "description")
    top_n: int,               # How many top issues to include in the final report
    compute_name: str,        # AML compute cluster to run each step on (e.g. "lin-cpu-00")
):
    # --- Step 1: Ingest & validate raw data ---
    ingest_step = ingest_data(
        raw_data=raw_data,
        text_column=text_column,
    )
    ingest_step.compute = compute_name

    # --- Step 2: Extract issue summaries using SLM ---
    extract_step = extract_issues(
        validated_data=ingest_step.outputs.validated_data,
        text_column=text_column,
    )
    extract_step.compute = compute_name

    # --- Step 3: Cluster similar issues together ---
    cluster_step = cluster_issues(
        extracted_issues=extract_step.outputs.extracted_issues,
    )
    cluster_step.compute = compute_name

    # --- Step 4: Rank and generate top issues report ---
    rank_step = rank_issues(
        clustered_issues=cluster_step.outputs.clustered_issues,
        top_n=top_n,
    )
    rank_step.compute = compute_name

    return rank_step.outputs


def main():
    # Pull the global component variables so we can assign them here
    # and the pipeline function above can reference them when building the graph.
    global ingest_data, extract_issues, cluster_issues, rank_issues

    # --- Load configuration ---
    config = ENVIRONMENT
    pipeline_cfg = PIPELINE

    # --- Authenticate and create Azure ML client ---
    cred = InteractiveBrowserCredential(tenant_id=config["tenant_id"])
    ml_client = MLClient(
        cred,
        config["azure_subscription_id"],
        config["azure_resource_group"],
        config["azure_ml_account"]
    )

    # --- Load components from local YAML specs ---
    # Each YAML file describes one step: what script to run, what conda
    # environment to use, and what inputs/outputs it expects.
    ingest_data    = load_component(source="components/ingest/component.yml")
    extract_issues = load_component(source="components/extract/component.yml")
    cluster_issues = load_component(source="components/cluster/component.yml")
    rank_issues    = load_component(source="components/rank/component.yml")

    # --- Build the pipeline graph ---
    # Calls top_issues_pipeline() with the actual config values to construct
    # the pipeline object (the DAG of steps). Nothing runs yet.
    pipeline = top_issues_pipeline(
        raw_data=Input(type="uri_file", path=pipeline_cfg["input_data"]),
        text_column=pipeline_cfg["text_column"],
        top_n=pipeline_cfg["top_n"],
        compute_name=config["compute_name"],
    )

    # Route the final CSV output to heron_sandbox_storage
    pipeline.outputs.final_report = Output(
        type=AssetTypes.URI_FOLDER,
        path=f"azureml://datastores/heron_sandbox_storage/paths/{pipeline_cfg['output_path']}",
    )

    # --- Submit the pipeline to AML ---
    # This is the "go" button — sends the pipeline graph to Azure ML,
    # which spins up compute and runs all 4 steps in the cloud.
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline,
        experiment_name=pipeline_cfg["experiment_name"]
    )

    print("Pipeline submitted! Track at:", pipeline_job.studio_url)


if __name__ == "__main__":
    main()
