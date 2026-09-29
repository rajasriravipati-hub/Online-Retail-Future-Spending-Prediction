import mlflow
from mlflow.tracking import MlflowClient


REGISTERED_MODEL_NAME = (
    "Online_Retail_Future_Spending_Model"
)


print("===================================")
print("MODEL VERSION COMPARISON")
print("===================================")


client = MlflowClient()


print("\nRegistered Model:")
print(REGISTERED_MODEL_NAME)


versions = client.search_model_versions(
    f"name='{REGISTERED_MODEL_NAME}'"
)


print("\nAvailable model versions:")

for version in versions:

    run_id = version.run_id

    run = client.get_run(run_id)

    metrics = run.data.metrics

    params = run.data.params

    print("\n-----------------------------------")

    print("Model Version:", version.version)

    print("Run ID:", run_id)

    print(
        "Alpha:",
        params.get("alpha")
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
        "Lifecycle Stage:",
        version.current_stage
    )


print("\n===================================")
print("MODEL COMPARISON COMPLETED")
print("===================================")