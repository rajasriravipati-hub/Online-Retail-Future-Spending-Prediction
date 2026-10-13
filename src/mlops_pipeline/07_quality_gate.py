
import json
from pathlib import Path

REPORT_FILE = Path("reports/evaluation_report.json")
GATE_REPORT = Path("reports/quality_gate_report.json")

# Minimum acceptable regression performance
MAX_MAE = 1.0
MAX_RMSE = 5.0
MIN_R2 = 0.90


def main():
    print("\n====================================")
    print("LAB 7 - STEP 7: MODEL QUALITY GATE")
    print("====================================")

    if not REPORT_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation report not found: {REPORT_FILE}"
        )

    with REPORT_FILE.open("r", encoding="utf-8") as file:
        report = json.load(file)

    metrics = report.get("metrics", {})
    required = ["MAE", "RMSE", "R2"]

    missing = [name for name in required if name not in metrics]
    if missing:
        raise ValueError(f"Missing evaluation metrics: {missing}")

    checks = {
        "MAE": metrics["MAE"] <= MAX_MAE,
        "RMSE": metrics["RMSE"] <= MAX_RMSE,
        "R2": metrics["R2"] >= MIN_R2,
    }

    passed = all(checks.values())

    result = {
        "thresholds": {
            "maximum_MAE": MAX_MAE,
            "maximum_RMSE": MAX_RMSE,
            "minimum_R2": MIN_R2,
        },
        "metrics": metrics,
        "checks": {
            name: "PASS" if status else "FAIL"
            for name, status in checks.items()
        },
        "status": "passed" if passed else "failed",
    }

    GATE_REPORT.parent.mkdir(parents=True, exist_ok=True)

    with GATE_REPORT.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    for name, status in result["checks"].items():
        print(f"{name}: {status}")

    print("Quality gate report:", GATE_REPORT)

    if not passed:
        raise SystemExit("Quality gate failed. Pipeline must stop.")

    print("Quality gate passed. Model can proceed.")


if __name__ == "__main__":
    main()