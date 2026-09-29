import os
import pandas as pd

INPUT_FILE = "data/OnlineRetail.csv"

OUTPUT_DIR = "data/processed"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "lab5_processed_online_retail.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess_data():

    print("===================================")
    print("LAB 5 PREPROCESSING PIPELINE")
    print("===================================")

    print("\nLoading raw dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        encoding="latin1"
    )

    print("Original shape:", df.shape)

    # 1. Remove records without CustomerID
    df = df.dropna(
        subset=["CustomerID"]
    )

    # 2. Remove cancelled invoices
    df = df[
        ~df["InvoiceNo"]
        .astype(str)
        .str.startswith("C")
    ]

    # 3. Remove invalid quantities
    df = df[
        df["Quantity"] > 0
    ]

    # 4. Remove invalid prices
    df = df[
        df["UnitPrice"] > 0
    ]

    # 5. Convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    # 6. Remove corrupted date records
    df = df.dropna(
        subset=["InvoiceDate"]
    )

    # 7. Remove duplicate transaction records
    df = df.drop_duplicates()

    # 8. Calculate total price
    df["TotalPrice"] = (
        df["Quantity"] *
        df["UnitPrice"]
    )

    # 9. Create customer-level dataset
    customer_data = df.groupby(
        "CustomerID"
    ).agg(
        TotalSpent=("TotalPrice", "sum"),
        NumberOfOrders=("InvoiceNo", "nunique"),
        TotalQuantity=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        AverageUnitPrice=("UnitPrice", "mean")
    ).reset_index()

    # 10. Calculate average order value
    customer_data["AverageOrderValue"] = (
        customer_data["TotalSpent"]
        / customer_data["NumberOfOrders"]
    )

    # 11. Create target variable
    customer_data["FutureSpending"] = (
        customer_data["TotalSpent"] * 0.10
    )

    # 12. Select final columns
    final_columns = [
        "TotalSpent",
        "NumberOfOrders",
        "TotalQuantity",
        "AverageOrderValue",
        "UniqueProducts",
        "AverageUnitPrice",
        "FutureSpending"
    ]

    customer_data = customer_data[
        final_columns
    ]

    # 13. Remove duplicate customer-level records
    customer_data = customer_data.drop_duplicates()

    # 14. Replace infinite values
    customer_data = customer_data.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    # 15. Remove missing values
    customer_data = customer_data.dropna()

    # 16. Save processed dataset
    customer_data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nProcessed shape:",
        customer_data.shape
    )

    print(
        "Processed dataset saved to:",
        OUTPUT_FILE
    )

    print("\nPipeline is idempotent:")
    print(
        "Running the pipeline again "
        "produces the same preprocessing result."
    )

    print("\n===================================")
    print("PREPROCESSING COMPLETED")
    print("===================================")


if __name__ == "__main__":
    preprocess_data()