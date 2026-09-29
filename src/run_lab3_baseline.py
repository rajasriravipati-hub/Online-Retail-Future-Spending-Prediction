import subprocess
import sys


# Lab 3 pipeline steps
steps = [
    "src/preprocess.py",
    "src/train.py",
    "src/evaluate.py"
]


print("=" * 50)
print("LAB 3: ONLINE RETAIL ML PIPELINE")
print("=" * 50)


for script in steps:

    print("\n" + "-" * 50)
    print(f"Running: {script}")
    print("-" * 50)

    result = subprocess.run(
        [sys.executable, script]
    )

    if result.returncode != 0:
        print(f"\nERROR: {script} failed.")
        sys.exit(1)


print("\n" + "=" * 50)
print("LAB 3 PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 50)