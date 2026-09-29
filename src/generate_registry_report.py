import mlflow
from mlflow.tracking import MlflowClient


REGISTERED_MODEL_NAME = (
    "Online_Retail_Future_Spending_Model"
)


print("===================================")
print("MODEL REGISTRY TRACEABILITY REPORT")
print("===================================")


client = MlflowClient()


print("\nRegistered Model:")
print(REGISTERED_MODEL_NAME)


versions = client.search_model_versions(
    f"name='{REGISTERED_MODEL_NAME}'"
)


for version in sorted(
    versions,
    key=lambda x: int(x.version)
):

    run = client.get_run(
        version.run_id
    )

    params = run.data.params
    metrics = run.data.metrics
    tags = version.tags

    print("\n-----------------------------------")

    print(
        "Model Version:",
        version.version
    )

    print(
        "Run ID:",
        version.run_id
    )

    print(
        "Model:",
        params.get("model")
    )

    print(
        "Alpha:",
        params.get("alpha")
    )

    print(
        "Preprocessing Dependency:",
        params.get(
            "preprocessing_dependency"
        )
    )

    print(
        "Dataset:",
        params.get("dataset_file")
    )

    print(
        "Dataset Hash:",
        params.get("dataset_hash")
    )

    print(
        "Features:",
        params.get("features")
    )

    print(
        "Target:",
        params.get("target")
    )

    print(
        "MAE:",
        f"{metrics.get('MAE', 0):.4f}"
    )

    print(
        "RMSE:",
        f"{metrics.get('RMSE', 0):.4f}"
    )

    print(
        "R2:",
        f"{metrics.get('R2', 0):.4f}"
    )

    print(
        "Validation Status:",
        tags.get("validation_status", "N/A")
    )

    print(
        "Deployment Ready:",
        tags.get("deployment_ready", "N/A")
    )

    print(
        "Lifecycle Stage:",
        tags.get("lifecycle_stage", "N/A")
    )

    print(
        "Deployment Status:",
        tags.get("deployment_status", "N/A")
    )

    print(
        "Promotion Status:",
        tags.get("promotion_status", "N/A")
    )

    print(
        "Rollback Status:",
        tags.get("rollback_status", "N/A")
    )


print("\n===================================")
print("TRACEABILITY REPORT COMPLETED")
print("===================================")