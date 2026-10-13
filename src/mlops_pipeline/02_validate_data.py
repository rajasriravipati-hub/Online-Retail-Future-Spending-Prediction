
import json
from pathlib import Path

import pandas as pd

DATA_FILE = Path("data/OnlineRetail.csv")
REPORT_FILE = Path("artifacts/baseline_validation.csv")
JSON_REPORT = Path("artifacts/data_validation_report.json")

REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def main():
    print("\n====================================")
    print("LAB 7 - STEP 2: VALIDATE RAW DATA")
    print("====================================")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, encoding="latin1")

    # Check required columns
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    # Check numeric columns
    numeric_columns = ["Quantity", "UnitPrice"]
    numeric_errors = []

    for column in numeric_columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        invalid_count = int(converted.isna().sum())

        if invalid_count:
            numeric_errors.append({
                "column": column,
                "invalid_values": invalid_count,
            })

    # Check dates
    invalid_dates = int(
        pd.to_datetime(df["InvoiceDate"], errors="coerce").isna().sum()
    )

    # Create a validation summary
    checks = [
        {
            "check": "required_columns",
            "status": "PASS" if not missing_columns else "FAIL",
            "details": ", ".join(missing_columns) if missing_columns else "All present",
        },
        {
            "check": "numeric_columns",
            "status": "PASS" if not numeric_errors else "FAIL",
            "details": json.dumps(numeric_errors) if numeric_errors else "Valid",
        },
        {
            "check": "invoice_dates",
            "status": "PASS" if invalid_dates == 0 else "FAIL",
            "details": f"Invalid dates: {invalid_dates}",
        },
    ]

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(REPORT_FILE, index=False)

    report = {
        "dataset": str(DATA_FILE),
        "rows": int(len(df)),
        "missing_columns": missing_columns,
        "numeric_errors": numeric_errors,
        "invalid_dates": invalid_dates,
        "status": (
            "success"
            if all(item["status"] == "PASS" for item in checks)
            else "failed"
        ),
    }

    with JSON_REPORT.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("Rows checked:", len(df))
    print("Missing columns:", missing_columns)
    print("Invalid dates:", invalid_dates)
    print("CSV report:", REPORT_FILE)
    print("JSON report:", JSON_REPORT)

    if missing_columns or numeric_errors or invalid_dates:
        raise ValueError("Raw dataset validation failed. Check the reports.")

    print("Raw dataset validation passed.")


if __name__ == "__main__":
    main()