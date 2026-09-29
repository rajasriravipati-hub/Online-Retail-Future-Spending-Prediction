import os
import hashlib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_FILE = "data/processed/processed_online_retail.csv"
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)


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


print("Loading processed dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


dataset_hash = get_file_hash(INPUT_FILE)

print("Dataset SHA256:", dataset_hash)


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


mlflow.set_experiment(
    "Online_Retail_Future_Spending"
)


with mlflow.start_run(
    run_name="Ridge_Baseline"
):

    # Change this value for different experiments
    alpha = 10.0

    model = Ridge(alpha=alpha)

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


    # Parameters

    mlflow.log_param(
        "model",
        "Ridge Regression"
    )

    mlflow.log_param(
        "alpha",
        alpha
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
        "scaler",
        "StandardScaler"
    )

    mlflow.log_param(
        "feature_count",
        len(FEATURES)
    )

    mlflow.log_param(
        "preprocessing",
        "Customer-level aggregation + StandardScaler"
    )

    mlflow.log_param(
        "dataset_file",
        INPUT_FILE
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
        "dataset_sha256",
        dataset_hash
    )

    mlflow.log_param(
        "features",
        ", ".join(FEATURES)
    )

    mlflow.log_param(
        "target",
        TARGET
    )


    # Metrics

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


    # Save model information

    model_info_file = os.path.join(
        ARTIFACT_DIR,
        "ridge_mlflow_model_info.txt"
    )


    with open(
        model_info_file,
        "w"
    ) as file:

        file.write(
            "Online Retail Future Spending Model\n"
        )

        file.write(
            "Model: Ridge Regression\n"
        )

        file.write(
            f"Alpha: {alpha}\n"
        )

        file.write(
            f"Features: {FEATURES}\n"
        )

        file.write(
            f"Target: {TARGET}\n"
        )

        file.write(
            f"MAE: {mae}\n"
        )

        file.write(
            f"RMSE: {rmse}\n"
        )

        file.write(
            f"R2: {r2}\n"
        )

        file.write(
            f"Dataset SHA256: {dataset_hash}\n"
        )


    mlflow.log_artifact(
        model_info_file
    )


    # Log trained model

    mlflow.sklearn.log_model(
        model,
        name="ridge_model"
    )


    print("\n==============================")

    print(
        "MLFLOW EXPERIMENT COMPLETED"
    )

    print("==============================")

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R2   : {r2:.4f}"
    )

    print("\nRun ID:")

    print(
        mlflow.active_run().info.run_id
    )

    print("==============================")