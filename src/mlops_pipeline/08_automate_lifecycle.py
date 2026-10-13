
import json
from datetime import datetime, timezone
from pathlib import Path

GATE_REPORT = Path("reports/quality_gate_report.json")
MODEL_FILE = Path("models/ridge_model_lab7.pkl")
SCALER_FILE = Path("models/scaler_lab7.pkl")
LIFECYCLE_REPORT = Path("artifacts/production_model_report.json")


def main():
    print("\n====================================")
    print("LAB 7 - STEP 8: MODEL LIFECYCLE")
    print("====================================")

    if not GATE_REPORT.exists():
        raise FileNotFoundError(
            f"Quality gate report not found: {GATE_REPORT}"
        )

    with GATE_REPORT.open("r", encoding="utf-8") as file:
        gate = json.load(file)

    if gate.get("status") != "passed":
        raise SystemExit(
            "Model did not pass the quality gate. Lifecycle stopped."
        )

    if not MODEL_FILE.exists() or not SCALER_FILE.exists():
        raise FileNotFoundError(
            "Lab 7 model or scaler artifact is missing."
        )

    report = {
        "model": str(MODEL_FILE),
        "scaler": str(SCALER_FILE),
        "quality_gate": "passed",
        "lifecycle_status": "eligible_for_promotion",
        "production_promotion_performed": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    LIFECYCLE_REPORT.parent.mkdir(parents=True, exist_ok=True)

    with LIFECYCLE_REPORT.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("Quality gate passed.")
    print("Model and scaler artifacts verified.")
    print("Lifecycle status: eligible_for_promotion")
    print("Production model was not changed.")
    print("Report saved:", LIFECYCLE_REPORT)


if __name__ == "__main__":
    main()