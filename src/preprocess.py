import os
import pandas as pd


# ==============================
# File paths
# ==============================

INPUT_FILE = "data/OnlineRetail.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "processed_online_retail.csv"
)


# Create output folder if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# Preprocessing function
# ==============================

def preprocess_data():

    print("Loading Online Retail dataset...")

    # Read CSV using latin1 encoding
    df = pd.read_csv(
        INPUT_FILE,
        encoding="latin1"
    )

    print("Original shape:", df.shape)

    # ==============================
    # Data cleaning
    # ==============================

    # Remove rows where CustomerID is missing
    df = df.dropna(subset=["CustomerID"])

    # Remove cancelled transactions
    df = df[
        ~df["InvoiceNo"]
        .astype(str)
        .str.startswith("C")
    ]

    # Keep only positive quantities
    df = df[df["Quantity"] > 0]

    # Keep only positive unit prices
    df = df[df["UnitPrice"] > 0]

    # Convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    # Remove rows where InvoiceDate could not be converted
    df = df.dropna(subset=["InvoiceDate"])

    # ==============================
    # Feature engineering
    # ==============================

    # Calculate total price for each transaction
    df["TotalPrice"] = (
        df["Quantity"] * df["UnitPrice"]
    )

    # Create customer-level features
    customer_data = df.groupby("CustomerID").agg(

        TotalSpent=(
            "TotalPrice",
            "sum"
        ),

        NumberOfOrders=(
            "InvoiceNo",
            "nunique"
        ),

        TotalQuantity=(
            "Quantity",
            "sum"
        ),

        UniqueProducts=(
            "StockCode",
            "nunique"
        ),

        AverageUnitPrice=(
            "UnitPrice",
            "mean"
        )

    ).reset_index()

    # ==============================
    # Average Order Value
    # ==============================

    customer_data["AverageOrderValue"] = (
        customer_data["TotalSpent"]
        / customer_data["NumberOfOrders"]
    )

    # ==============================
    # Target variable
    # ==============================

    customer_data["FutureSpending"] = (
        customer_data["TotalSpent"] * 0.10
    )

    # ==============================
    # Select final features
    # ==============================

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

    # ==============================
    # Remove invalid values
    # ==============================

    customer_data = customer_data.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    customer_data = customer_data.dropna()

    # ==============================
    # Save processed dataset
    # ==============================

    customer_data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ==============================
    # Final information
    # ==============================

    print(
        "Processed shape:",
        customer_data.shape
    )

    print(
        "Processed dataset saved to:",
        OUTPUT_FILE
    )

    print("\nFinal features:")

    for column in final_columns:
        print("-", column)


# ==============================
# Main
# ==============================

if __name__ == "__main__":
    print("Preprocessing version: v2")
    preprocess_data()