import mlflow
from mlflow.tracking import MlflowClient


REGISTERED_MODEL_NAME = (
    "Online_Retail_Future_Spending_Model"
)

ROLLBACK_VERSION = 1


print("===================================")
print("MODEL ROLLBACK")
print("===================================")


client = MlflowClient()


# Get the model version
version = client.get_model_version(
    REGISTERED_MODEL_NAME,
    str(ROLLBACK_VERSION)
)


print("\nRegistered Model:")
print(REGISTERED_MODEL_NAME)

print("\nRollback Version:")
print(ROLLBACK_VERSION)

print("\nModel Run ID:")
print(version.run_id)


# Restore the selected version as production
client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    str(ROLLBACK_VERSION),
    "lifecycle_stage",
    "PRODUCTION"
)


client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    str(ROLLBACK_VERSION),
    "deployment_status",
    "PRODUCTION"
)


client.set_model_version_tag(
    REGISTERED_MODEL_NAME,
    str(ROLLBACK_VERSION),
    "rollback_status",
    "RESTORED"
)


print("\nRollback completed successfully.")

print("\nActive Production Version:")
print(ROLLBACK_VERSION)

print("\nLifecycle Stage:")
print("PRODUCTION")

print("\nRollback Status:")
print("RESTORED")


print("\n===================================")
print("MODEL ROLLBACK COMPLETED")
print("===================================")