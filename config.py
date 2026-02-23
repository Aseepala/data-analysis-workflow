# Azure ML pipeline configuration for Top Issues Analysis

ENVIRONMENT = {
    "compute_name": "lin-cpu-00", #AML compute cluster name (all heron environments have a lin-cpu-00 cluster))
    "tenant_id": "cdc5aeea-15c5-4db6-b079-fcadd2505dc2", #This is the Torus Tenant (all Heron environments are Torus based)
    "azure_ml_account": "amlworkspacej5m54ypkuhrju", #AML workspace name
    "azure_openai_account": "cogSerAcc0c2eb660nam", #Azure OpenAI resource name
    "azure_subscription_id": "08047947-f71e-4462-a09d-266e3d34c431", #Azure subscription ID
    "azure_resource_group": "EYESON.HERON.PROD.ca212fed-605a-41a1-9c97-53f99a10b9e8", #Azure resource group
}

PIPELINE = {
    "pipeline_yml": "pipeline.yml",
    "experiment_name": "top-issues-analysis",
    "input_data": "test_data/issues.csv",  # local sample data (swap for azureml:issues-dataset@latest to use a registered dataset)
    "text_column": "description",
    "top_n": 10,
}
