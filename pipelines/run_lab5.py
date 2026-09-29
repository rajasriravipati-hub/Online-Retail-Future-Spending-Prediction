import subprocess
import sys

print("\n========================================")
print("      EXPERIMENT 5 - DATA VALIDATION")
print("========================================")


print("\n[1/3] Running data validation...")

result1 = subprocess.run(
    [sys.executable, "src/validate_data.py"]
)

if result1.returncode != 0:
    print("\nExperiment 5 failed during data validation.")
    sys.exit(1)


print("\n[2/3] Running preprocessing pipeline...")

result2 = subprocess.run(
    [sys.executable, "src/preprocess_pipeline.py"]
)

if result2.returncode != 0:
    print("\nExperiment 5 failed during preprocessing.")
    sys.exit(1)


print("\n[3/3] Running output validation...")

result3 = subprocess.run(
    [sys.executable, "src/validate_outputs.py"]
)

if result3.returncode != 0:
    print("\nExperiment 5 failed during output validation.")
    sys.exit(1)

print("EXPERIMENT 5 SUCCESSFULLY COMPLETED")
