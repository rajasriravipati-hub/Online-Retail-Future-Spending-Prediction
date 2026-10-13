
import json
import shutil
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "lab5_processed_online_retail.csv"
)

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
ARTIFACT_DIR = ROOT / "artifacts"

MODEL_FILE = MODEL_DIR / "lab9_recovery_model.pkl"
BACKUP_MODEL_FILE = MODEL_DIR / "lab9_recovery_model_backup.pkl"

REPORT_FILE = REPORT_DIR / "lab9_recovery_report.json"


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

FEATURES = [
    "TotalSpent",
    "NumberOfOrders",
    "TotalQuantity",
    "AverageOrderValue",
    "UniqueProducts",
    "AverageUnitPrice",
]

TARGET = "FutureSpending"
REQUIRED_COLUMNS = FEATURES + [TARGET]


# ============================================================
# 3. LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# 4. LOAD AND VALIDATE DATA
# ============================================================

def load_data():
    """
    Load the processed dataset.

    Validate required columns, handle missing values,
    convert numeric values, and reject invalid data.
    """

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Processed dataset is missing: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    missing_columns = sorted(
        set(REQUIRED_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Schema mismatch. Missing columns: {missing_columns}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    # Convert feature and target columns to numeric values.
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Remove rows with missing or invalid values.
    invalid_rows = int(
        df[REQUIRED_COLUMNS].isna().any(axis=1).sum()
    )

    if invalid_rows:
        logger.warning(
            "Removing %s rows containing missing or invalid values.",
            invalid_rows,
        )

        df = df.dropna(subset=REQUIRED_COLUMNS)

    # Reject infinite values.
    numeric_values = df[REQUIRED_COLUMNS]

    if not numeric_values.map(
        lambda value: abs(value) != float("inf")
    ).all().all():
        raise ValueError(
            "Dataset contains infinite numeric values."
        )

    if df.empty:
        raise ValueError(
            "No valid rows remain after data validation."
        )

    if len(df) < 5:
        raise ValueError(
            "At least 5 valid rows are required for training."
        )

    logger.info("Loaded %s valid rows.", len(df))

    return df


# ============================================================
# 5. TRAIN AND SAVE MODEL
# ============================================================

def train_model(df):
    """
    Train a Ridge regression model.

    Create a backup of the existing model when available.
    Save the new model and verify that it can make predictions.
    """

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Back up the previous model before replacing it.
    if MODEL_FILE.is_file():
        shutil.copy2(
            MODEL_FILE,
            BACKUP_MODEL_FILE,
        )

        logger.info("Existing model backup created.")

    X = df[FEATURES]
    y = df[TARGET]

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

    mae = float(
        mean_absolute_error(y_test, predictions)
    )

    # Save to a temporary file first.
    temporary_model = MODEL_DIR / "lab9_recovery_model.tmp"

    try:
        joblib.dump(model, temporary_model)

        # Replace the current model after saving succeeds.
        temporary_model.replace(MODEL_FILE)

    finally:
        if temporary_model.exists():
            temporary_model.unlink()

    # Verify the saved model can be loaded and used.
    loaded_model = joblib.load(MODEL_FILE)

    verification_prediction = loaded_model.predict(
        X_test.head(1)
    )

    if len(verification_prediction) != 1:
        raise RuntimeError(
            "Saved model verification failed."
        )

    result = {
        "model_path": str(
            MODEL_FILE.relative_to(ROOT)
        ),
        "backup_path": (
            str(BACKUP_MODEL_FILE.relative_to(ROOT))
            if BACKUP_MODEL_FILE.is_file()
            else None
        ),
        "test_rows": len(X_test),
        "mae": mae,
        "model_load_verified": True,
    }

    logger.info("Model training completed.")
    logger.info("Test MAE: %.6f", mae)

    return result


# ============================================================
# 6. RECOVER A MISSING MODEL
# ============================================================

def recover_missing_model(df):
    """
    Recover the model from a backup if possible.

    If the backup is missing or invalid, retrain the model.
    """

    # Attempt recovery from the backup.
    if BACKUP_MODEL_FILE.is_file():
        try:
            backup_model = joblib.load(
                BACKUP_MODEL_FILE
            )

            # Verify that the backup is usable.
            backup_model.predict(
                df[FEATURES].head(1)
            )

            logger.warning(
                "Recovered model from backup."
            )

            return {
                "recovery_action": "loaded_backup_model",
                "model_path": str(
                    BACKUP_MODEL_FILE.relative_to(ROOT)
                ),
                "model": backup_model,
            }

        except Exception as exc:
            logger.error(
                "Backup model could not be used: %s",
                exc,
            )

    # If no valid backup exists, retrain.
    logger.warning(
        "No usable backup found. Retraining the model."
    )

    training_result = train_model(df)

    recovered_model = joblib.load(MODEL_FILE)

    return {
        "recovery_action": "retrained_model",
        **training_result,
        "model": recovered_model,
    }


# ============================================================
# 7. TEST FIVE FAILURE SCENARIOS
# ============================================================

def run_failure_tests(df):
    """
    Simulate five failure scenarios.

    The tests use copies of the dataset so that the original
    processed dataset is not modified.
    """

    results = []

    # --------------------------------------------------------
    # SCENARIO 1: MISSING DATA
    # --------------------------------------------------------

    try:
        damaged = df[REQUIRED_COLUMNS].copy()

        # Simulate a missing feature value.
        damaged.loc[
            damaged.index[0],
            FEATURES[0],
        ] = float("nan")

        # Recovery: remove affected rows.
        recovered = damaged.dropna(
            subset=REQUIRED_COLUMNS
        )

        assert not recovered.empty

        assert not recovered[
            REQUIRED_COLUMNS
        ].isna().any().any()

        results.append({
            "scenario": "missing_data",
            "status": "RECOVERED",
            "root_cause": (
                "A required numeric value was missing."
            ),
            "recovery": (
                "Detected and removed the affected row."
            ),
        })

    except Exception as exc:
        results.append({
            "scenario": "missing_data",
            "status": "FAILED",
            "error": str(exc),
        })

    # --------------------------------------------------------
    # SCENARIO 2: INVALID DATATYPE
    # --------------------------------------------------------


    try:
        damaged = df[REQUIRED_COLUMNS].head(10).copy()

        # Allow both numbers and text in this test column.
        damaged[FEATURES[0]] = damaged[FEATURES[0]].astype(object)

        # Inject genuinely invalid text.
        damaged.loc[
            damaged.index[0],
            FEATURES[0],
        ] = "not-a-number"

        # Attempt strict numeric conversion.
        try:
            damaged[FEATURES[0]] = pd.to_numeric(
                damaged[FEATURES[0]],
                errors="raise",
            )
        except (ValueError, TypeError):
            # Recover by converting invalid text to missing values.
            damaged[FEATURES[0]] = pd.to_numeric(
                damaged[FEATURES[0]],
                errors="coerce",
            )

            damaged = damaged.dropna(
                subset=[FEATURES[0]]
            )

        assert not damaged.empty
        assert pd.api.types.is_numeric_dtype(
            damaged[FEATURES[0]]
        )
        assert not damaged[FEATURES[0]].isna().any()

        results.append({
            "scenario": "invalid_datatype",
            "status": "RECOVERED",
            "root_cause": "A feature contained non-numeric text.",
            "recovery": "Converted valid values and removed invalid rows.",
        })

    except Exception as exc:
        results.append({
            "scenario": "invalid_datatype",
            "status": "FAILED",
            "error": str(exc),
        })
    # --------------------------------------------------------
    # SCENARIO 3: SCHEMA MISMATCH
    # --------------------------------------------------------

    try:
        # Remove a required feature to simulate schema failure.
        damaged = df.drop(
            columns=[FEATURES[0]]
        )

        missing_columns = sorted(
            set(REQUIRED_COLUMNS) - set(damaged.columns)
        )

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                f"{missing_columns}"
            )

        # Reaching this point means detection failed.
        results.append({
            "scenario": "schema_mismatch",
            "status": "UNEXPECTED_PASS",
        })

    except ValueError as exc:
        results.append({
            "scenario": "schema_mismatch",
            "status": "RECOVERED",
            "root_cause": str(exc),
            "recovery": (
                "Detected and rejected the invalid schema."
            ),
        })

    # --------------------------------------------------------
    # SCENARIO 4: PREPROCESSING ERROR
    # --------------------------------------------------------

    try:
        damaged = df[REQUIRED_COLUMNS].copy()

        # Allow text injection into the numeric column.
        damaged[FEATURES[0]] = damaged[FEATURES[0]].astype(object)

        # Simulate a preprocessing failure.
        damaged.loc[
            damaged.index[0],
            FEATURES[0],
        ] = "not-a-number"

        try:
            damaged[FEATURES[0]] = pd.to_numeric(
                damaged[FEATURES[0]],
                errors="raise",
            )
        except (ValueError, TypeError) as exc:
            logger.warning(
                "Preprocessing error detected: %s",
                exc,
            )

            # Recovery: coerce invalid text and remove bad rows.
            damaged[FEATURES[0]] = pd.to_numeric(
                damaged[FEATURES[0]],
                errors="coerce",
            )

            damaged = damaged.dropna(
                subset=[FEATURES[0]]
            )

        assert not damaged.empty
        assert pd.api.types.is_numeric_dtype(
            damaged[FEATURES[0]]
        )
        assert not damaged[FEATURES[0]].isna().any()

        results.append({
            "scenario": "preprocessing_error",
            "status": "RECOVERED",
            "root_cause": "Non-numeric text interrupted preprocessing.",
            "recovery": "Coerced invalid values and removed affected rows.",
        })

    except Exception as exc:
        results.append({
            "scenario": "preprocessing_error",
            "status": "FAILED",
            "error": str(exc),
        })

    # --------------------------------------------------------
    # SCENARIO 5: MISSING MODEL ARTIFACT
    # --------------------------------------------------------

    try:
        # Use a separate expected path for the simulated failure.
        missing_model = (
            MODEL_DIR
            / "lab9_intentionally_missing_model.pkl"
        )

        if missing_model.exists():
            missing_model.unlink()

        try:
            joblib.load(missing_model)

        except FileNotFoundError:
            # Recover using a backup or retraining.
            recovery = recover_missing_model(df)

            recovered_model = recovery["model"]

            # Verify the recovered model can predict.
            predictions = recovered_model.predict(
                df[FEATURES].head(1)
            )

            assert len(predictions) == 1

            results.append({
                "scenario": "missing_model_artifact",
                "status": "RECOVERED",
                "root_cause": (
                    "The expected model artifact was absent."
                ),
                "recovery": recovery["recovery_action"],
                "prediction_verified": True,
            })

    except Exception as exc:
        results.append({
            "scenario": "missing_model_artifact",
            "status": "FAILED",
            "error": str(exc),
        })

    return results


# ============================================================
# 8. RUN LAB 9 AND GENERATE REPORT
# ============================================================

def run_lab9():
    """
    Run the Lab 9 recovery experiment and save a JSON report.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "experiment": (
            "Experiment 9: Pipeline Failure Handling and Recovery"
        ),
        "dataset": str(
            DATA_FILE.relative_to(ROOT)
        ),
        "results": [],
        "model_training": {},
        "summary": {},
    }

    try:
        # Load and validate the dataset.
        df = load_data()

        # Train and save the initial model.
        report["model_training"] = train_model(df)

        # Simulate and recover from five failures.
        report["results"] = run_failure_tests(df)

    except Exception as exc:
        logger.exception(
            "Lab 9 pipeline stopped safely."
        )

        report["fatal_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    # Count recovered scenarios.
    recovered_count = sum(
        result.get("status") == "RECOVERED"
        for result in report["results"]
    )

    failed_count = sum(
        result.get("status") == "FAILED"
        for result in report["results"]
    )

    total_scenarios = len(report["results"])

    all_recovered = (
        total_scenarios == 5
        and recovered_count == 5
    )

    report["summary"] = {
        "total_scenarios": total_scenarios,
        "recovered_scenarios": recovered_count,
        "failed_scenarios": failed_count,
        "all_scenarios_recovered": all_recovered,
    }

    # Save the report.
    REPORT_FILE.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    # Print a readable summary.
    print(
        "\n========== LAB 9 RECOVERY SUMMARY =========="
    )

    for result in report["results"]:
        print(
            f"{result['scenario']}: "
            f"{result['status']}"
        )

    print(
        f"Recovered: {recovered_count}/{total_scenarios}"
    )

    print(
        "All scenarios recovered: "
        f"{all_recovered}"
    )

    print(
        "Report saved to: "
        f"{REPORT_FILE.relative_to(ROOT)}"
    )

    print(
        "============================================\n"
    )

    if report.get("fatal_error"):
        raise RuntimeError(
            "Lab 9 failed: "
            f"{report['fatal_error']['message']}"
        )

    if not all_recovered:
        raise RuntimeError(
            "One or more failure scenarios did not recover."
        )

    return report


# ============================================================
# 9. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_lab9()