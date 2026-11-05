import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "lab5_processed_online_retail.csv"
MODEL = ROOT / "models" / "lab9_recovery_model.pkl"
REPORT = ROOT / "reports" / "lab12_monitoring_report.json"
CANDIDATE = ROOT / "artifacts" / "lab12_candidate_model.pkl"

FEATURES = [
    "TotalSpent", "NumberOfOrders", "TotalQuantity",
    "AverageOrderValue", "UniqueProducts", "AverageUnitPrice"
]
TARGET = "FutureSpending"

def main():
    print("Loading Online Retail dataset...")
    df = pd.read_csv(DATA)
    df = df[FEATURES + [TARGET]].replace([np.inf, -np.inf], np.nan).dropna()

    reference, current = train_test_split(
        df, test_size=0.30, random_state=42
    )

    drift_results = {}
    for feature in FEATURES:
        statistic = ks_2samp(reference[feature], current[feature])
        drift_results[feature] = {
            "ks_statistic": float(statistic.statistic),
            "p_value": float(statistic.pvalue),
            "drift_detected": bool(statistic.pvalue < 0.05)
        }

    deployed_model = joblib.load(MODEL)

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.20, random_state=42
    )

    old_predictions = deployed_model.predict(X_test)
    old_metrics = {
        "mae": float(mean_absolute_error(y_test, old_predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, old_predictions))),
        "r2": float(r2_score(y_test, old_predictions))
    }

    candidate = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    candidate.fit(X_train, y_train)
    new_predictions = candidate.predict(X_test)

    new_metrics = {
        "mae": float(mean_absolute_error(y_test, new_predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, new_predictions))),
        "r2": float(r2_score(y_test, new_predictions))
    }

    old_prediction_mean = float(np.mean(old_predictions))
    new_prediction_mean = float(np.mean(new_predictions))

    prediction_drift = {
        "deployed_mean_prediction": old_prediction_mean,
        "candidate_mean_prediction": new_prediction_mean,
        "absolute_mean_difference": abs(old_prediction_mean - new_prediction_mean)
    }

    # Keep the candidate separate; do not automatically replace production.
    joblib.dump(candidate, CANDIDATE)

    report = {
        "experiment": "Lab 12 - Monitoring, Drift Detection and Retraining",
        "dataset_rows": int(len(df)),
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "feature_drift": drift_results,
        "feature_drift_count": sum(x["drift_detected"] for x in drift_results.values()),
        "deployed_model_metrics": old_metrics,
        "candidate_model_metrics": new_metrics,
        "prediction_drift": prediction_drift,
        "candidate_model_path": str(CANDIDATE.relative_to(ROOT)),
        "promotion_decision": (
            "Candidate performs better or equal on MAE and is eligible for review"
            if new_metrics["mae"] <= old_metrics["mae"]
            else "Keep deployed model; candidate did not improve MAE"
        )
    }

    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=4), encoding="utf-8")

    print("\n========== LAB 12 RESULTS ==========")
    print("Dataset rows:", len(df))
    print("Feature drift detected:", report["feature_drift_count"])
    print("Deployed model:", old_metrics)
    print("Candidate model:", new_metrics)
    print("Decision:", report["promotion_decision"])
    print("Report:", REPORT)
    print("Candidate:", CANDIDATE)

if __name__ == "__main__":
    main()
