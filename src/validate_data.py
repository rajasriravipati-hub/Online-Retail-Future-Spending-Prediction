import pandas as pd


INPUT_FILE = "data/processed/processed_online_retail.csv"


EXPECTED_COLUMNS = {
    "TotalSpent": "float64",
    "NumberOfOrders": "int64",
    "TotalQuantity": "int64",
    "AverageOrderValue": "float64",
    "UniqueProducts": "int64",
    "AverageUnitPrice": "float64",
    "FutureSpending": "float64"
}


print("===================================")
print("DATA VALIDATION")
print("===================================")


# Load dataset
df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:")
print(df.shape)


# 1. Schema validation
print("\n1. Checking schema...")

actual_columns = list(df.columns)
expected_columns = list(EXPECTED_COLUMNS.keys())

if actual_columns == expected_columns:
    print("Schema columns are correct.")
else:
    print("Schema mismatch detected.")
    print("Expected:", expected_columns)
    print("Actual:", actual_columns)


# 2. Missing attributes
print("\n2. Checking missing attributes...")

missing_columns = [
    column
    for column in expected_columns
    if column not in df.columns
]

if len(missing_columns) == 0:
    print("No required attributes are missing.")
else:
    print("Missing attributes:", missing_columns)


# 3. Datatype validation
print("\n3. Checking datatypes...")

datatype_errors = []

for column, expected_type in EXPECTED_COLUMNS.items():

    if column in df.columns:

        actual_type = str(df[column].dtype)

        if actual_type != expected_type:

            datatype_errors.append(
                f"{column}: expected {expected_type}, got {actual_type}"
            )


if len(datatype_errors) == 0:
    print("All datatypes are correct.")
else:

    print("Datatype mismatches detected:")

    for error in datatype_errors:
        print("-", error)


# 4. Missing values
print("\n4. Checking missing values...")

missing_values = df.isnull().sum()

total_missing = missing_values.sum()

if total_missing == 0:
    print("No missing values found.")
else:

    print("Missing values detected:")

    print(
        missing_values[
            missing_values > 0
        ]
    )


# 5. Invalid numerical values
print("\n5. Checking invalid numerical values...")

numeric_columns = list(
    EXPECTED_COLUMNS.keys()
)

invalid_values = {}

for column in numeric_columns:

    if column in df.columns:

        invalid_count = (
            ~pd.to_numeric(
                df[column],
                errors="coerce"
            ).notna()
        ).sum()

        if invalid_count > 0:

            invalid_values[column] = invalid_count


if len(invalid_values) == 0:
    print("No invalid numerical values found.")
else:

    print("Invalid numerical values detected:")

    for column, count in invalid_values.items():

        print(
            f"- {column}: {count}"
        )


# 6. Negative value validation
print("\n6. Checking negative values...")

negative_columns = [
    "TotalSpent",
    "NumberOfOrders",
    "TotalQuantity",
    "AverageOrderValue",
    "UniqueProducts",
    "AverageUnitPrice",
    "FutureSpending"
]

negative_found = False

for column in negative_columns:

    if column in df.columns:

        count = (
            df[column] < 0
        ).sum()

        if count > 0:

            negative_found = True

            print(
                f"- {column}: {count} negative values"
            )


if not negative_found:
    print("No negative values found.")


# 7. Duplicate records
print("\n7. Checking duplicate records...")

duplicate_count = df.duplicated().sum()

print(
    "Duplicate records:",
    duplicate_count
)


# Final result
print("\n===================================")
print("DATA VALIDATION COMPLETED")
print("===================================")