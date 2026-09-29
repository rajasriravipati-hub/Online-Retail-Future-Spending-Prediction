import mlflow
from mlflow.tracking import MlflowClient


REGISTERED_MODEL_NAME = (
    "Online_Retail_Future_Spending_Model"
)

MODEL_VERSION = 1

MAE_THRESHOLD = 1.0


print("===================================")
print("MODEL VALIDATION")
print("===================================")

client = MlflowClient()


versions = client.search_model_versions(
    f"name='{REGISTERED_MODEL_NAME}'"
)


selected_version = None


for version in versions:

    if int(version.version) == MODEL_VERSION:
        selected_version = version
        break


if selected_version is None:

    print("Model version not found.")

    raise SystemExit(1)


run = client.get_run(
    selected_version.run_id
)


metrics = run.data.metrics


mae = metrics.get("MAE")

rmse = metrics.get("RMSE")

r2 = metrics.get("R2")


print("\nModel Version:", MODEL_VERSION)

print("Run ID:", selected_version.run_id)

print(f"MAE  : {mae:.4f}")

print(f"RMSE : {rmse:.4f}")

print(f"R2   : {r2:.4f}")


print("\nValidation Rule:")

print(
    f"MAE must be less than {MAE_THRESHOLD}"
)


if mae < MAE_THRESHOLD:

    validation_status = "PASSED"

    deployment_ready = "YES"

else:

    validation_status = "FAILED"

    deployment_ready = "NO"


client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    str(MODEL_VERSION),
    "validation_status",
    validation_status
)


client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    str(MODEL_VERSION),
    "deployment_ready",
    deployment_ready
)


print("\nValidation Status:")
print(validation_status)


print("\nDeployment Ready:")
print(deployment_ready)


print("\n===================================")
print("MODEL VALIDATION COMPLETED")
print("===================================")