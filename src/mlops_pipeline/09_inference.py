
import json
from pathlib import Path

import joblib
import pandas as pd

DATA_FILE = Path("data/processed/lab5_processed_online_retail.csv")
MODEL_FILE = Path("models/ridge_model_lab7.pkl")
SCALER_FILE = Path("models/scaler_lab7.pkl")

OUTPUT_FILE = Path("outputs/inference_predictions.csv")
REPORT_FILE = Path("artifacts/inference_report.json")

FEATURES = [
    "TotalSpent",
    "NumberOfOrders",
    "TotalQuantity",
    "AverageOrderValue",
    "UniqueProducts",
    "AverageUnitPrice",
]


def main():
    print("\n====================================")
    print("LAB 7 - STEP 9: INFERENCE")
    print("====================================")

    for path in [DATA_FILE, MODEL_FILE, SCALER_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    df = pd.read_csv(DATA_FILE)

    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing inference features: {missing}")

    if df.empty:
        raise ValueError("Inference dataset is empty.")

    X = df[FEATURES]

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    # Save actual values too, when the target is available
    results = df[FEATURES].copy()
    results["PredictedFutureSpending"] = predictions

    if "FutureSpending" in df.columns:
        results["ActualFutureSpending"] = df["FutureSpending"]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    report = {
        "input_rows": int(len(df)),
        "prediction_count": int(len(predictions)),
        "output_file": str(OUTPUT_FILE),
        "model_file": str(MODEL_FILE),
        "status": "success",
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("Predictions generated:", len(predictions))
    print("Predictions saved:", OUTPUT_FILE)
    print("Inference report:", REPORT_FILE)
    print("Inference completed successfully.")


if __name__ == "__main__":
    main()