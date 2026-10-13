
import json
from pathlib import Path

import pandas as pd

DATA_FILE = Path("data/processed/lab5_processed_online_retail.csv")
CSV_REPORT = Path("artifacts/processed_output_validation.csv")
JSON_REPORT = Path("artifacts/processed_output_validation_report.json")

REQUIRED_COLUMNS = [
    "TotalSpent",
    "NumberOfOrders",
    "TotalQuantity",
    "AverageOrderValue",
    "UniqueProducts",
    "AverageUnitPrice",
    "FutureSpending",
]


def main():
    print("\n====================================")
    print("LAB 7 - STEP 4: VALIDATE OUTPUT")
    print("====================================")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]
    missing_values = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    numeric_columns = df.select_dtypes(include="number").columns
    non_finite_values = int(
        (~df[numeric_columns].apply(
            lambda column: column.map(lambda value: pd.notna(value) and abs(value) != float("inf"))
        )).sum().sum()
    )

    checks = [
        {
            "check": "required_columns",
            "status": "PASS" if not missing_columns else "FAIL",
            "details": str(missing_columns),
        },
        {
            "check": "missing_values",
            "status": "PASS" if missing_values == 0 else "FAIL",
            "details": str(missing_values),
        },
        {
            "check": "duplicate_rows",
            "status": "PASS" if duplicate_rows == 0 else "FAIL",
            "details": str(duplicate_rows),
        },
        {
            "check": "non_finite_values",
            "status": "PASS" if non_finite_values == 0 else "FAIL",
            "details": str(non_finite_values),
        },
        {
            "check": "non_empty_dataset",
            "status": "PASS" if not df.empty else "FAIL",
            "details": str(len(df)),
        },
    ]

    CSV_REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(CSV_REPORT, index=False)

    passed = all(item["status"] == "PASS" for item in checks)

    report = {
        "dataset": str(DATA_FILE),
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "missing_columns": missing_columns,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "non_finite_values": non_finite_values,
        "status": "success" if passed else "failed",
    }

    with JSON_REPORT.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("Rows:", len(df))
    print("Missing columns:", missing_columns)
    print("Missing values:", missing_values)
    print("Duplicate rows:", duplicate_rows)
    print("Validation report:", CSV_REPORT)

    if not passed:
        raise ValueError("Processed output validation failed. Check the reports.")

    print("Processed output validation passed.")


if __name__ == "__main__":
    main()