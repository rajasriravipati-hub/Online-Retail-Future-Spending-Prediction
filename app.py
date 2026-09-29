import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Online Retail Spending Predictor",
    page_icon="🛒",
    layout="centered"
)


# --------------------------------------------------
# Load Model and Scaler
# --------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/ridge_model.pkl"
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    return model, scaler


model, scaler = load_model()


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🛒 Online Retail Customer Spending Predictor")

st.write(
    "Enter customer purchasing information to predict "
    "future customer spending."
)


# --------------------------------------------------
# User Inputs
# --------------------------------------------------

st.subheader("Customer Information")


total_spent = st.number_input(
    "Total Spent",
    min_value=0.0,
    value=1000.0,
    step=100.0
)


number_of_orders = st.number_input(
    "Number of Orders",
    min_value=1,
    value=5,
    step=1
)


total_quantity = st.number_input(
    "Total Quantity",
    min_value=1,
    value=100,
    step=10
)


average_order_value = st.number_input(
    "Average Order Value",
    min_value=0.0,
    value=200.0,
    step=10.0
)


unique_products = st.number_input(
    "Unique Products",
    min_value=1,
    value=20,
    step=1
)


average_unit_price = st.number_input(
    "Average Unit Price",
    min_value=0.0,
    value=5.0,
    step=0.5
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Future Spending"):

    input_data = pd.DataFrame({
        "TotalSpent": [total_spent],
        "NumberOfOrders": [number_of_orders],
        "TotalQuantity": [total_quantity],
        "AverageOrderValue": [average_order_value],
        "UniqueProducts": [unique_products],
        "AverageUnitPrice": [average_unit_price]
    })

    # Apply the same scaling used during training
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)

    predicted_spending = prediction[0]

    st.success(
        f"Predicted Future Spending: ₹{predicted_spending:,.2f}"
    )


# --------------------------------------------------
# Model Information
# --------------------------------------------------

st.divider()

st.subheader("Model Information")

st.write(
    "Final Model: Tuned Ridge Regression"
)

st.write(
    "Target Variable: FutureSpending"
)

st.write(
    "Evaluation Metrics: MAE, RMSE and R² Score"
)