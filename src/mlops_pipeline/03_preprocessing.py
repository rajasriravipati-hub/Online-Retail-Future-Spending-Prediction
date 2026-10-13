
import json
import runpy
from pathlib import Path

import pandas as pd

PREPROCESS_SCRIPT = Path("src/preprocess_pipeline.py")
OUTPUT_FILE = Path("data/processed/lab5_processed_online_retail.csv")
REPORT_FILE = Path("artifacts/preprocessing_summary_report.json")


def main():
    print("\n====================================")
    print("LAB 7 - STEP 3: PREPROCESS DATA")
    print("====================================")

    if not PREPROCESS_SCRIPT.exists():
        raise FileNotFoundError(
            f"Existing preprocessing script not found: {PREPROCESS_SCRIPT}"
        )

    # Reuse the preprocessing logic from Lab 5
    runpy.run_path(str(PREPROCESS_SCRIPT), run_name="__main__")

    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"Preprocessed output was not created: {OUTPUT_FILE}"
        )

    df = pd.read_csv(OUTPUT_FILE)

    report = {
        "output_file": str(OUTPUT_FILE),
        "rows_after_preprocessing": int(len(df)),
        "columns": df.columns.tolist(),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "status": "success",
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("Preprocessing completed.")
    print("Processed rows:", len(df))
    print("Processed columns:", len(df.columns))
    print("Summary report:", REPORT_FILE)


if __name__ == "__main__":
    main()