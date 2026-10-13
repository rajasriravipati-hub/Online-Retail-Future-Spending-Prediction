import sys
from pathlib import Path

# Add the project root directory to Python's import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recovery_pipeline.failure_recovery import run_lab9


if __name__ == "__main__":
    run_lab9()
