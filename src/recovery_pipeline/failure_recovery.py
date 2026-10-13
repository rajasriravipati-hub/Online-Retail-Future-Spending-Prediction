
import json
import shutil
import logging
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "lab5_processed_online_retail.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
ARTIFACT_DIR = ROOT / "artifacts"

MODEL_FILE = MODEL_DIR / "lab9_recovery_model.pkl"
BACKUP_MODEL_FILE = MODEL_DIR / "lab9_recovery_model_backup.pkl"
REPORT_FILE = REPORT_DIR / "lab9_recovery_report.json"

FEATURES = [
    "TotalSpent",
    "NumberOfOrders",
    "TotalQuantity",
    "AverageOrderValue",
    "UniqueProducts",
    "AverageUnitPrice",
]
TARGET = "FutureSpending"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_data():
    """Load the processed dataset and recover from missing data files."""
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Processed dataset is missing: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required = FEATURES + [TARGET]
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Schema mismatch. Missing columns: {missing_columns}"
        )

    missing_values = int(df[required].isna().sum().sum())
    if missing_values:
        logger.warning(
            "Found %s missing values; removing affected rows.",
            missing_values,
        )
        df = df.dropna(subset=required)

    if df.empty:
        raise ValueError("No valid rows remain after missing-value handling.")

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_rows = int(df[required].isna().any(axis=1).sum())
    if invalid_rows:
        logger.warning(
            "Removed %s rows with invalid numeric values.",
            invalid_rows,
        )
        df = df.dropna(subset=required)

    if df.empty:
        raise ValueError("No valid numeric rows remain.")

    if not (df[FEATURES + [TARGET]].apply(
        lambda column: column.map(lambda value: abs(value) != float("inf"))
    )).all().all():
        raise ValueError("Dataset contains infinite numeric values.")

    return df


def train_model(df):
    """Train and save a Ridge model, retaining a backup when possible."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_FILE.exists():
        shutil.copy2(MODEL_FILE, BACKUP_MODEL_FILE)

    X = df[FEATURES]
    y = df[TARGET]

    if len(df) < 5:
        raise ValueError("At least 5 valid rows are required for training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, predictions))

    temporary_model = MODEL_DIR / "lab9_recovery_model.tmp"
    joblib.dump(model, temporary_model)
    temporary_model.replace(MODEL_FILE)

    # Verify the saved artifact can actually be loaded.
    loaded_model = joblib.load(MODEL_FILE)
    loaded_model.predict(X_test.head(1))

    return {
        "model_path": str(MODEL_FILE.relative_to(ROOT)),
        "backup_path": (
            str(BACKUP_MODEL_FILE.relative_to(ROOT))
            if BACKUP_MODEL_FILE.exists()
            else None
        ),
        "test_rows": len(X_test),
        "mae": mae,
    }


def recover_missing_model(df):
    """Load the backup model if available; otherwise retrain."""
    if BACKUP_MODEL_FILE.is_file():
        try:
            model = joblib.load(BACKUP_MODEL_FILE)
            logger.warning("Recovered model from backup.")
            return {
                "recovery_action": "loaded_backup_model",
                "model_path": str(BACKUP_MODEL_FILE.relative_to(ROOT)),
                "model": model,
            }
        except Exception as exc:
            logger.error("Backup model is invalid: %s", exc)

    logger.warning("No usable backup found; retraining the model.")
    result = train_model(df)
    return {
        "recovery_action": "retrained_model",
        **result,
    }


def run_failure_tests(df):
    """Test five controlled failure scenarios without changing source data."""
    results = []

    # 1. Missing data
    try:
        damaged = df[FEATURES + [TARGET]].copy()
        damaged.loc[damaged.index[0], FEATURES[0]] = float("nan")
        if damaged.isna().any().any():
            damaged = damaged.dropna()
        assert not damaged.isna().any().any()
        results.append({
            "scenario": "missing_data",
            "status": "RECOVERED",
            "root_cause": "Required numeric value was missing.",
            "recovery": "Detected and removed the affected row.",
        })
    except Exception as exc:
        results.append({
            "scenario": "missing_data",
            "status": "FAILED",
            "error": str(exc),
        })

    # 2. Invalid datatype
    try:
        damaged = df[FEATURES + [TARGET]].head(10).copy()
        damaged[FEATURES[0]] = damaged[FEATURES[0]].astype(str)
        damaged[FEATURES[0]] = pd.to_numeric(
            damaged[FEATURES[0]], errors="raise"
        )
        results.append({
            "scenario": "invalid_datatype",
            "status": "RECOVERED",
            "root_cause": "Feature values arrived as strings.",
            "recovery": "Validated and converted numeric strings.",
        })
    except Exception as exc:
        results.append({
            "scenario": "invalid_datatype",
            "status": "DETECTED",
            "error": str(exc),
        })

    # 3. Schema mismatch
    try:
        damaged = df.drop(columns=[FEATURES[0]])
        missing = set(FEATURES + [TARGET]) - set(damaged.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        results.append({
            "scenario": "schema_mismatch",
            "status": "UNEXPECTED_PASS",
        })
    except ValueError as exc:
        results.append({
            "scenario": "schema_mismatch",
            "status": "RECOVERED",
            "root_cause": str(exc),
            "recovery": "Rejected the invalid schema before training.",
        })

    # 4. Preprocessing error
    try:
        damaged = df[FEATURES + [TARGET]].copy()
        damaged[FEATURES[0]] = "not-a-number"
        try:
            damaged[FEATURES[0]] = pd.to_numeric(
                damaged[FEATURES[0]], errors="raise"
            )
        except (ValueError, TypeError) as exc:
            logger.warning("Preprocessing failure caught: %s", exc)
            damaged[FEATURES[0]] = pd.to_numeric(
                damaged[FEATURES[0]], errors="coerce"
            )
            damaged = damaged.dropna(subset=[FEATURES[0]])

        assert pd.api.types.is_numeric_dtype(damaged[FEATURES[0]])
        results.append({
            "scenario": "preprocessing_error",
            "status": "RECOVERED",
            "root_cause": "A non-numeric value interrupted conversion.",
            "recovery": "Coerced invalid values to missing and removed affected rows.",
        })
    except Exception as exc:
        results.append({
            "scenario": "preprocessing_error",
            "status": "FAILED",
            "error": str(exc),
        })

    # 5. Missing model artifact
    try:
        missing_model = MODEL_DIR / "lab9_intentionally_missing_model.pkl"
        if missing_model.exists():
            missing_model.unlink()

        try:
            joblib.load(missing_model)
        except FileNotFoundError:
            recovery = recover_missing_model(df)
            results.append({
                "scenario": "missing_model_artifact",
                "status": "RECOVERED",
                "root_cause": "Expected model artifact was absent.",
                "recovery": recovery["recovery_action"],
            })
    except Exception as exc:
        results.append({
            "scenario": "missing_model_artifact",
            "status": "FAILED",
            "error": str(exc),
        })

    return results


def run_lab9():
    """Execute the recovery experiment and write its JSON report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": "Experiment 9: Pipeline Failure Handling and Recovery",
        "dataset": str(DATA_FILE.relative_to(ROOT)),
        "results": [],
        "model_training": {},
        "summary": {},
    }

    try:
        df = load_data()
        logger.info("Loaded %s valid rows.", len(df))

        report["model_training"] = train_model(df)
        report["results"] = run_failure_tests(df)

    except Exception as exc:
        logger.exception("Lab 9 pipeline stopped safely.")
        report["fatal_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    passed = sum(
        result["status"] == "RECOVERED"
        for result in report["results"]
    )
    total = len(report["results"])

    report["summary"] = {
        "total_scenarios": total,
        "recovered_scenarios": passed,
        "failed_scenarios": sum(
            result["status"] == "FAILED"
            for result in report["results"]
        ),
        "all_scenarios_recovered": total == 5 and passed == 5,
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
        
    )

    print("\n========== LAB 9 RECOVERY SUMMARY ==========")
    for result in report["results"]:
        print(f"{result['scenario']}: {result['status']}")
    print(f"Report saved to: {REPORT_FILE.relative_to(ROOT)}")
    print("============================================\n")

    if report.get("fatal_error"):
        raise RuntimeError(
            f"Lab 9 failed: {report['fatal_error']['message']}"
        )

    if not report["summary"]["all_scenarios_recovered"]:
        raise RuntimeError(
            "One or more failure scenarios did not recover successfully."
        )

    return report