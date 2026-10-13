
import json
from pathlib import Path

import pandas as pd

# Project paths
DATA_FILE = Path("data/OnlineRetail.csv")
REPORT_FILE = Path("artifacts/data_load_report.json")


def main():
    print("\n====================================")
    print("LAB 7 - STEP 1: LOAD RAW DATA")
    print("====================================")

    # Check whether the dataset exists
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Check that data/OnlineRetail.csv exists."
        )

    # Load the dataset
    df = pd.read_csv(DATA_FILE, encoding="latin1")

    if df.empty:
        raise ValueError("The raw dataset is empty.")

    # Generate a summary report
    report = {
        "dataset": str(DATA_FILE),
        "rows": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "missing_values": {
            str(column): int(count)
            for column, count in df.isnull().sum().items()
        },
        "status": "success"
    }

    # Save the report
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("Dataset loaded successfully.")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Report saved to:", REPORT_FILE)


if __name__ == "__main__":
    main()