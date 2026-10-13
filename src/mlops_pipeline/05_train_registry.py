
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path("data/processed/lab5_processed_online_retail.csv")
MODEL_DIR = Path("models")
REPORT_FILE = Path("artifacts/lab7_training_report.json")

MODEL_FILE = MODEL_DIR / "ridge_model_lab7.pkl"
SCALER_FILE = MODEL_DIR / "scaler_lab7.pkl"

FEATURES = [
    "TotalSpent",
    "NumberOfOrders",
    "TotalQuantity",
    "AverageOrderValue",
    "UniqueProducts",
    "AverageUnitPrice",
]
TARGET = "FutureSpending"


def main():
    print("\n====================================")
    print("LAB 7 - STEP 5: TRAIN RIDGE MODEL")
    print("====================================")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    required = FEATURES + [TARGET]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) < 2:
        raise ValueError("At least two rows are required for training.")

    if df[required].isnull().any().any():
        raise ValueError("Training data contains missing values.")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(mean_squared_error(y_test, predictions) ** 0.5),
        "R2": float(r2_score(y_test, predictions)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    # Record this training run in MLflow
    mlflow.set_experiment("Online_Retail_Lab7_Pipeline")

    with mlflow.start_run(run_name="Lab7_Ridge_Regression") as run:
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("alpha", 1.0)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_FILE))
        mlflow.log_artifact(str(SCALER_FILE))

        report = {
            "experiment": "Online_Retail_Lab7_Pipeline",
            "run_id": run.info.run_id,
            "model": str(MODEL_FILE),
            "scaler": str(SCALER_FILE),
            "training_rows": len(X_train),
            "testing_rows": len(X_test),
            "features": FEATURES,
            "target": TARGET,
            "metrics": metrics,
            "status": "success",
        }

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("Model saved:", MODEL_FILE)
    print("Scaler saved:", SCALER_FILE)
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Evaluation metrics:", metrics)
    print("MLflow run recorded.")
    print("Report saved:", REPORT_FILE)


if __name__ == "__main__":
    main()