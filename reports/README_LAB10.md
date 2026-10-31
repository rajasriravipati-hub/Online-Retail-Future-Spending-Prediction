# Lab 10 – Online Retail Future Spending Prediction API

## Objective
To deploy the Online Retail Future Spending Prediction model using FastAPI and provide REST API endpoints for customer spending predictions.

## Model Information
- Model: Ridge Regression
- Model file: `models/lab9_recovery_model.pkl`
- Model type: Regression
- Target variable: FutureSpending
- Preprocessing: StandardScaler included in the trained Pipeline
- Features: TotalSpent, NumberOfOrders, TotalQuantity, AverageOrderValue, UniqueProducts, AverageUnitPrice

## API Endpoints
- GET `/` – API welcome message
- GET `/health` – Checks API and model health
- GET `/metadata` – Returns model information
- POST `/predict` – Predicts future spending for one customer
- POST `/predict/batch` – Predicts future spending for multiple customers

## Validation
Pydantic schemas validate the six customer input features and reject negative values.

## Testing Results
Command: `python -m pytest tests/test_api.py -v`

- Total tests: 6
- Passed: 6
- Failed: 0

Test cases:
1. Home endpoint
2. Health endpoint
3. Metadata endpoint
4. Single prediction
5. Batch prediction
6. Invalid input validation

## Sample Prediction
Input:
- TotalSpent: 1000
- NumberOfOrders: 5
- TotalQuantity: 100
- AverageOrderValue: 200
- UniqueProducts: 20
- AverageUnitPrice: 5

Predicted Future Spending: 99.76678583956921

## Conclusion
Successfully developed and tested a FastAPI application for Online Retail Future Spending Prediction. The API supports single and batch predictions, model health monitoring, metadata retrieval, and input validation.