import mlflow
from mlflow.tracking import MlflowClient


REGISTERED_MODEL_NAME = (
    "Online_Retail_Future_Spending_Model"
)

MODEL_VERSION = 1


print("===================================")
print("MODEL PROMOTION")
print("===================================")


client = MlflowClient()


# Check validation status
version_details = client.get_model_version(
    REGISTERED_MODEL_NAME,
    str(MODEL_VERSION)
)


validation_status = version_details.tags.get(
    "validation_status"
)


deployment_ready = version_details.tags.get(
    "deployment_ready"
)


print("\nModel Version:", MODEL_VERSION)

print(
    "Validation Status:",
    validation_status
)

print(
    "Deployment Ready:",
    deployment_ready
)


if (
    validation_status != "PASSED"
    or deployment_ready != "YES"
):

    print("\nModel cannot be promoted.")

    raise SystemExit(1)


# Promote model to production
client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    str(MODEL_VERSION),
    "lifecycle_stage",
    "PRODUCTION"
)


client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    str(MODEL_VERSION),
    "deployment_status",
    "PRODUCTION"
)


client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    str(MODEL_VERSION),
    "promotion_status",
    "PROMOTED"
)


print("\nModel promoted successfully.")

print("\nModel Version:")
print(MODEL_VERSION)

print("\nLifecycle Stage:")
print("PRODUCTION")

print("\nDeployment Status:")
print("PRODUCTION")

print("\nPromotion Status:")
print("PROMOTED")


print("\n===================================")
print("MODEL PROMOTION COMPLETED")
print("===================================")