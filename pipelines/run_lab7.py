
import subprocess
import sys

STEPS = [
    ("Load raw dataset", "src/mlops_pipeline/01_load_data.py"),
    ("Validate raw dataset", "src/mlops_pipeline/02_validate_data.py"),
    ("Preprocess dataset", "src/mlops_pipeline/03_preprocessing.py"),
    ("Validate processed output", "src/mlops_pipeline/04_validate_outputs.py"),
    ("Train Ridge model and log MLflow run", "src/mlops_pipeline/05_train_registry.py"),
    ("Evaluate trained model", "src/mlops_pipeline/06_model_evaluation.py"),
    ("Run model quality gate", "src/mlops_pipeline/07_quality_gate.py"),
    ("Automate model lifecycle", "src/mlops_pipeline/08_automate_lifecycle.py"),
    ("Generate inference predictions", "src/mlops_pipeline/09_inference.py"),
    ("Test failure handling", "src/mlops_pipeline/10_failure_injection.py"),
]


def main():
    print("\n========================================")
    print("ONLINE RETAIL - LAB 7 MLOPS PIPELINE")
    print("========================================")

    for number, (description, script) in enumerate(STEPS, start=1):
        print(f"\n[{number}/{len(STEPS)}] {description}")
        print("-" * 50)

        result = subprocess.run(
            [sys.executable, script],
            check=False,
        )

        if result.returncode != 0:
            print(f"\nPIPELINE FAILED at step {number}: {script}")
            sys.exit(result.returncode)

    print("\n========================================")
    print("LAB 7 PIPELINE COMPLETED SUCCESSFULLY")
    print("========================================")


if __name__ == "__main__":
    main()