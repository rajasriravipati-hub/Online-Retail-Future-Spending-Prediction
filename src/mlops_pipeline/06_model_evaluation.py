
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_FILE = Path("data/processed/lab5_processed_online_retail.csv")
MODEL_FILE = Path("models/ridge_model_lab7.pkl")
SCALER_FILE = Path("models/scaler_lab7.pkl")
REPORT_FILE = Path("reports/evaluation_report.json")

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
    print("LAB 7 - STEP 6: MODEL EVALUATION")
    print("====================================")

    for path in [DATA_FILE, MODEL_FILE, SCALER_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(DATA_FILE)
    X = df[FEATURES]
    y = df[TARGET]

    # Use the same split as the training script
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(mean_squared_error(y_test, predictions) ** 0.5),
        "R2": float(r2_score(y_test, predictions)),
    }

    report = {
        "model": str(MODEL_FILE),
        "test_rows": int(len(y_test)),
        "metrics": metrics,
        "status": "success",
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("Test rows:", len(y_test))
    print("MAE:", metrics["MAE"])
    print("RMSE:", metrics["RMSE"])
    print("R2:", metrics["R2"])
    print("Evaluation report:", REPORT_FILE)
    print("Model evaluation completed.")


if __name__ == "__main__":
    main()