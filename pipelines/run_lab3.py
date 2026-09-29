import subprocess
import sys



print("LAB 3: ONLINE RETAIL ML PIPELINE")



steps = [
    "src/preprocess.py",
    "src/train.py",
    "src/evaluate.py"
]


for script in steps:


    print("Running:", script)


    result = subprocess.run(
        [sys.executable, script]
    )

    if result.returncode != 0:
        print("\nERROR:", script, "failed.")
        sys.exit(1)


print("LAB 3 PIPELINE COMPLETED SUCCESSFULLY")
