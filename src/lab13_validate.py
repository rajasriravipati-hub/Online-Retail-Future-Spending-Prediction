import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from api.main import app
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

results = []

def check(name, condition, details=""):
    status = bool(condition)
    results.append({
        "check": name,
        "status": "PASSED" if status else "FAILED",
        "details": details
    })
    print(f"{'PASSED' if status else 'FAILED'}: {name} {details}")

def main():
    print("\n========== LAB 13 END-TO-END VALIDATION ==========")

    # Check project files
    required_files = [
        "api/main.py",
        "api/model_loader.py",
        "api/schemas.py",
        "Dockerfile",
        "k8s/deployment.yaml",
        "k8s/service.yaml",
        "models/lab9_recovery_model.pkl",
        "reports/lab12_monitoring_report.json"
    ]

    for filename in required_files:
        check(
            f"File exists: {filename}",
            (ROOT / filename).is_file()
        )

    # Check Lab 12 report
    report_path = REPORT_DIR / "lab12_monitoring_report.json"
    if report_path.exists():
        try:
            lab12 = json.loads(report_path.read_text(encoding="utf-8"))
            check("Lab 12 report is valid JSON", isinstance(lab12, dict))
            check(
                "Lab 12 candidate model metrics available",
                "candidate_model_metrics" in lab12
            )
        except Exception as exc:
            check("Lab 12 report is valid JSON", False, str(exc))
    else:
        check("Lab 12 report is valid JSON", False, "Report missing")

    # Test API endpoints
    customer = {
        "TotalSpent": 1000,
        "NumberOfOrders": 5,
        "TotalQuantity": 100,
        "AverageOrderValue": 200,
        "UniqueProducts": 20,
        "AverageUnitPrice": 5
    }

    with TestClient(app) as client:
        response = client.get("/")
        check("GET /", response.status_code == 200)

        response = client.get("/health")
        check(
            "GET /health",
            response.status_code == 200,
            str(response.json())
        )

        response = client.get("/metadata")
        check("GET /metadata", response.status_code == 200)

        response = client.post("/predict", json=customer)
        check(
            "POST /predict",
            response.status_code == 200,
            str(response.json())
        )

        response = client.post(
            "/predict/batch",
            json={"customers": [customer, customer]}
        )
        batch_ok = (
            response.status_code == 200
            and response.json().get("total_customers") == 2
            and len(response.json().get("predictions", [])) == 2
        )
        check("POST /predict/batch", batch_ok)

    passed = sum(r["status"] == "PASSED" for r in results)
    failed = len(results) - passed

    output = {
        "experiment": "Lab 13 - End-to-End MLOps Validation",
        "timestamp": datetime.now().isoformat(),
        "total_checks": len(results),
        "passed": passed,
        "failed": failed,
        "overall_status": "PASSED" if failed == 0 else "FAILED",
        "checks": results
    }

    output_path = REPORT_DIR / "lab13_validation_report.json"
    output_path.write_text(json.dumps(output, indent=4), encoding="utf-8")

    print("\n========== LAB 13 SUMMARY ==========")
    print(f"Total checks : {len(results)}")
    print(f"Passed       : {passed}")
    print(f"Failed       : {failed}")
    print(f"Report       : {output_path}")

    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()

