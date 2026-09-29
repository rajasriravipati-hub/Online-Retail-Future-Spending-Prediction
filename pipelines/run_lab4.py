import subprocess
import sys


print("\n========================================")
print("        EXPERIMENT 4 - MLFLOW")
print("========================================")


print("\n[1/2] Running MLflow experiment...")

result1 = subprocess.run(
    [sys.executable, "src/train_mlflow.py"]
)

if result1.returncode != 0:
    print("\nExperiment 4 failed during MLflow experiment.")
    sys.exit(1)


print("\n[2/2] Running reproducibility validation...")

result2 = subprocess.run(
    [sys.executable, "src/validate_reproducibility.py"]
)

if result2.returncode != 0:
    print("\nExperiment 4 failed during reproducibility validation.")
    sys.exit(1)



print("EXPERIMENT 4 SUCCESSFULLY COMPLETED")
