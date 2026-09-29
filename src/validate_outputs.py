import hashlib
import os
import pandas as pd

OUTPUT_FILE = "data/processed/lab5_processed_online_retail.csv"

EXPECTED_COLUMNS = [
    "TotalSpent",
    "NumberOfOrders",
    "TotalQuantity",
    "AverageOrderValue",
    "UniqueProducts",
    "AverageUnitPrice",
    "FutureSpending"
]


def get_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            data = file.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


print("===================================")
print("LAB 5 OUTPUT VALIDATION")
print("===================================")

if not os.path.exists(OUTPUT_FILE):
    print("ERROR: Processed output file does not exist.")
    raise SystemExit(1)

df = pd.read_csv(OUTPUT_FILE)

print("\n1. Checking output shape...")
print("Shape:", df.shape)

print("\n2. Checking output columns...")

if list(df.columns) == EXPECTED_COLUMNS:
    print("Output columns are correct.")
else:
    print("Output column mismatch detected.")
    raise SystemExit(1)

print("\n3. Checking missing values...")

missing = df.isnull().sum().sum()

if missing == 0:
    print("No missing values found.")
else:
    print("Missing values detected:", missing)
    raise SystemExit(1)

print("\n4. Checking duplicate records...")

duplicates = df.duplicated().sum()

print("Duplicate records:", duplicates)

if duplicates == 0:
    print("No duplicate records in processed output.")
else:
    print("Duplicate records still exist.")
    raise SystemExit(1)

print("\n5. Checking invalid values...")

if (df < 0).any().any():
    print("Negative values detected.")
    raise SystemExit(1)
else:
    print("No negative values found.")

print("\n6. Checking reproducibility hash...")

file_hash = get_file_hash(OUTPUT_FILE)

print("Output SHA256:")
print(file_hash)

print("\n===================================")
print("OUTPUT VALIDATION PASSED")
print("===================================")