import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


# ==============================
# File paths
# ==============================

INPUT_FILE = "data/processed/processed_online_retail.csv"

MODEL_DIR = "models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "ridge_model_lab3.pkl"
)

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "scaler_lab3.pkl"
)


# Create model directory
os.makedirs(MODEL_DIR, exist_ok=True)


# ==============================
# Load processed data
# ==============================

print("Loading processed dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# ==============================
# Features and target
# ==============================

features = [
    "TotalSpent",
    "NumberOfOrders",
    "TotalQuantity",
    "AverageOrderValue",
    "UniqueProducts",
    "AverageUnitPrice"
]

target = "FutureSpending"


X = df[features]
y = df[target]


# ==============================
# Train-test split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==============================
# Feature scaling
# ==============================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ==============================
# Train Ridge Regression
# ==============================

print("\nTraining Ridge Regression model...")

model = Ridge(alpha=1.0)

model.fit(
    X_train_scaled,
    y_train
)


print("Model training completed.")


# ==============================
# Save model and scaler
# ==============================

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    scaler,
    SCALER_FILE
)


print("\nModel saved to:")
print(MODEL_FILE)

print("\nScaler saved to:")
print(SCALER_FILE)


# ==============================
# Summary
# ==============================

print("\nTraining completed successfully.")

print("Features used:")

for feature in features:
    print("-", feature)

print("Target:", target)