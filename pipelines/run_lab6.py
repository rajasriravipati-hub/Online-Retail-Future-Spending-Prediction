import subprocess
import sys


print("\n========================================")
print("       EXPERIMENT 6 - MODEL REGISTRY")
print("========================================")


steps = [
    ("Training and registering Model V1",
     "src/train_registry.py"),

    ("Training and registering Model V2",
     "src/train_registry_v2.py"),

    ("Comparing model versions",
     "src/compare_models.py"),

    ("Validating Model V1",
     "src/validate_model.py"),

    ("Promoting Model V1",
     "src/promote_model.py"),

    ("Rolling back Model V1",
     "src/rollback_model.py"),

    ("Generating traceability report",
     "src/generate_registry_report.py")
]


for step_number, (description, script) in enumerate(
    steps,
    start=1
):

    print(
        f"\n[{step_number}/{len(steps)}] {description}..."
    )

    result = subprocess.run(
        [sys.executable, script]
    )

    if result.returncode != 0:

        print(
            f"\nExperiment 6 failed at: {script}"
        )

        sys.exit(1)

print("EXPERIMENT 6 SUCCESSFULLY COMPLETED")