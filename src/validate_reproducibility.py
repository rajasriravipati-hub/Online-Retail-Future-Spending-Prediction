import pandas as pd
import hashlib


INPUT_FILE = "data/processed/processed_online_retail.csv"


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
print("REPRODUCIBILITY VALIDATION")
print("===================================")

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:")
print(df.shape)

# Check required columns
required_columns = [
    "TotalSpent",
    "NumberOfOrders",
    "TotalQuantity",
    "AverageOrderValue",
    "UniqueProducts",
    "AverageUnitPrice",
    "FutureSpending"
]

print("\nChecking required columns...")

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if len(missing_columns) == 0:
    print("All required columns are present.")
else:
    print("Missing columns:", missing_columns)

# Check missing values
print("\nChecking missing values...")

missing_values = df[required_columns].isnull().sum()

print(missing_values)

if missing_values.sum() == 0:
    print("No missing values found.")
else:
    print("Missing values detected.")

# Dataset hash
dataset_hash = get_file_hash(INPUT_FILE)

print("\nDataset SHA256:")
print(dataset_hash)

# Check data types
print("\nChecking data types:")

print(df[required_columns].dtypes)

print("\n===================================")
print("REPRODUCIBILITY CHECK COMPLETED")
print("===================================")