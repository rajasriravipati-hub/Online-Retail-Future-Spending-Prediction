
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SOURCE_FILE = Path("data/processed/lab5_processed_online_retail.csv")
ERROR_FILE = Path("artifacts/corrupted_simulation_errors.csv")
REPORT_FILE = Path("artifacts/failure_simulation_report.json")

REQUIRED_COLUMNS = [
    "TotalSpent",
    "NumberOfOrders",
    "TotalQuantity",
    "AverageOrderValue",
    "UniqueProducts",
    "AverageUnitPrice",
    "FutureSpending",
]


def check_data(df):
    errors = []

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]
    if missing_columns:
        errors.append(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    if df.isnull().values.any():
        errors.append("Missing values detected")

    numeric_columns = df.select_dtypes(include="number").columns
    for column in numeric_columns:
        if (df[column] < 0).any():
            errors.append(f"Negative values detected in {column}")

    return errors


def main():
    print("\n====================================")
    print("LAB 7 - STEP 10: FAILURE INJECTION")
    print("====================================")

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE_FILE}")

    original = pd.read_csv(SOURCE_FILE)

    if original.empty:
        raise ValueError("Source dataset is empty.")

    scenarios = []

    # Scenario 1: remove a required column
    corrupted = original.drop(columns=["FutureSpending"])
    errors = check_data(corrupted)
    scenarios.append({
        "scenario": "missing_required_column",
        "expected_failure": True,
        "detected": len(errors) > 0,
        "errors": errors,
    })

    # Scenario 2: introduce a missing value
    corrupted = original.copy()
    corrupted.loc[corrupted.index[0], "TotalSpent"] = float("nan")
    errors = check_data(corrupted)
    scenarios.append({
        "scenario": "missing_value",
        "expected_failure": True,
        "detected": len(errors) > 0,
        "errors": errors,
    })

    # Scenario 3: introduce a negative value
    corrupted = original.copy()
    corrupted.loc[corrupted.index[0], "TotalSpent"] = -1
    errors = check_data(corrupted)
    scenarios.append({
        "scenario": "negative_value",
        "expected_failure": True,
        "detected": len(errors) > 0,
        "errors": errors,
    })

    for scenario in scenarios:
        scenario["status"] = (
            "PASS" if scenario["detected"] else "FAIL"
        )

    ERROR_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {
            "scenario": item["scenario"],
            "expected_failure": item["expected_failure"],
            "detected": item["detected"],
            "status": item["status"],
            "errors": "; ".join(item["errors"]),
        }
        for item in scenarios
    ]).to_csv(ERROR_FILE, index=False)

    all_passed = all(item["status"] == "PASS" for item in scenarios)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(SOURCE_FILE),
        "source_modified": False,
        "scenarios": scenarios,
        "status": "passed" if all_passed else "failed",
    }

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    for item in scenarios:
        print(f"{item['scenario']}: {item['status']}")

    print("Original dataset modified: No")
    print("Error details:", ERROR_FILE)
    print("Failure report:", REPORT_FILE)

    if not all_passed:
        raise SystemExit("Failure-injection tests failed.")

    print("All failure-injection tests passed.")


if __name__ == "__main__":
    main()