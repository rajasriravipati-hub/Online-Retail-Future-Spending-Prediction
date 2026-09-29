import hashlib
import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_FILE = "data/processed/lab5_processed_online_retail.csv"

EXPERIMENT_NAME = "Online_Retail_Model_Registry"

REGISTERED_MODEL_NAME = "Online_Retail_Future_Spending_Model"

MODEL_VERSION = "v2"

FEATURES = [
    "TotalSpent",
    "NumberOfOrders",
    "TotalQuantity",
    "AverageOrderValue",
    "UniqueProducts",
    "AverageUnitPrice"
]

TARGET = "FutureSpending"


def get_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            data = file.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


print("===================================")
print("LAB 6 MODEL REGISTRY - V2")
print("===================================")


# Check whether V2 is already registered
client = MlflowClient()

existing_versions = client.search_model_versions(
    f"name='{REGISTERED_MODEL_NAME}'"
)

for version in existing_versions:

    run = client.get_run(
        version.run_id
    )

    if (
        run.data.params.get("model_version")
        == MODEL_VERSION
    ):

        print("\nModel V2 is already registered.")

        print(
            "Existing MLflow Version:",
            version.version
        )

        print(
            "Run ID:",
            version.run_id
        )

        print("\nSkipping duplicate registration.")

        raise SystemExit(0)


print("\nLoading Lab 5 processed dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


dataset_hash = get_file_hash(INPUT_FILE)

print("\nDataset SHA256:")
print(dataset_hash)


X = df[FEATURES]

y = df[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


alpha = 10.0

model = Ridge(
    alpha=alpha
)

model.fit(
    X_train_scaled,
    y_train
)


y_pred = model.predict(
    X_test_scaled
)


mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n===================================")
print("MODEL EVALUATION")
print("===================================")

print(f"MAE  : {mae:.4f}")

print(f"RMSE : {rmse:.4f}")

print(f"R2   : {r2:.4f}")


mlflow.set_experiment(
    EXPERIMENT_NAME
)


with mlflow.start_run(
    run_name="Ridge_Model_V2"
) as run:

    mlflow.log_param(
        "model",
        "Ridge Regression"
    )

    mlflow.log_param(
        "model_version",
        MODEL_VERSION
    )

    mlflow.log_param(
        "alpha",
        alpha
    )

    mlflow.log_param(
        "scaler",
        "StandardScaler"
    )

    mlflow.log_param(
        "preprocessing_dependency",
        "Lab 5 preprocessing pipeline"
    )

    mlflow.log_param(
        "test_size",
        0.2
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_param(
        "dataset_file",
        INPUT_FILE
    )

    mlflow.log_param(
        "dataset_hash",
        dataset_hash
    )

    mlflow.log_param(
        "dataset_rows",
        len(df)
    )

    mlflow.log_param(
        "dataset_columns",
        len(df.columns)
    )

    mlflow.log_param(
        "features",
        ", ".join(FEATURES)
    )

    mlflow.log_param(
        "target",
        TARGET
    )

    mlflow.log_metric(
        "MAE",
        mae
    )

    mlflow.log_metric(
        "RMSE",
        rmse
    )

    mlflow.log_metric(
        "R2",
        r2
    )

    mlflow.set_tag(
        "lifecycle_stage",
        "STAGING"
    )

    mlflow.set_tag(
        "deployment_ready",
        "No"
    )

    mlflow.set_tag(
        "validation_status",
        "Pending"
    )

    mlflow.set_tag(
        "model_version",
        MODEL_VERSION
    )

    mlflow.sklearn.log_model(
        model,
        name="ridge_model",
        registered_model_name=REGISTERED_MODEL_NAME
    )

    run_id = run.info.run_id


print("\n===================================")
print("MODEL V2 REGISTERED")
print("===================================")

print("\nRegistered Model:")
print(REGISTERED_MODEL_NAME)

print("\nModel Version:")
print(MODEL_VERSION)

print("\nMLflow Run ID:")
print(run_id)

print("\nLifecycle Stage:")
print("STAGING")

print("\nValidation Status:")
print("Pending")

print("\nDeployment Ready:")
print("No")

print("\n===================================")
print("LAB 6 V2 COMPLETED")
print("===================================")