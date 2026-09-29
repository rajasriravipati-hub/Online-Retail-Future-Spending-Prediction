import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==============================
# File paths
# ==============================

INPUT_FILE = "data/processed/processed_online_retail.csv"

MODEL_FILE = "models/ridge_model_lab3.pkl"

SCALER_FILE = "models/scaler_lab3.pkl"


# ==============================
# Load data
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


# ==============================
# Load scaler and model
# ==============================

print("Loading trained model...")

scaler = joblib.load(SCALER_FILE)
model = joblib.load(MODEL_FILE)


# ==============================
# Scale test data
# ==============================

X_test_scaled = scaler.transform(X_test)


# ==============================
# Make predictions
# ==============================

print("Generating predictions...")

y_pred = model.predict(X_test_scaled)


# ==============================
# Calculate metrics
# ==============================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


# ==============================
# Display results
# ==============================

print("\n==============================")
print("LAB 3 MODEL EVALUATION")
print("==============================")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

print("==============================")
print("Evaluation completed successfully.")