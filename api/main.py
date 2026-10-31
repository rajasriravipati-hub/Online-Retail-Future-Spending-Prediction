from fastapi import FastAPI, HTTPException
import pandas as pd

from api.model_loader import get_model
from api.schemas import (
    CustomerFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse
)

app = FastAPI(
    title="Online Retail Future Spending Prediction API",
    description="API for predicting future customer spending",
    version="1.0.0"
)

FEATURE_NAMES = [
    "TotalSpent",
    "NumberOfOrders",
    "TotalQuantity",
    "AverageOrderValue",
    "UniqueProducts",
    "AverageUnitPrice"
]


@app.get("/")
def home():
    return {"message": "Online Retail Future Spending Prediction API is running"}


@app.get("/health")
def health():
    try:
        get_model()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metadata")
def metadata():
    return {
        "model_name": "Online Retail Ridge Regression",
        "model_version": "Lab 9 Recovery Model",
        "problem_type": "Regression",
        "target": "FutureSpending",
        "features": FEATURE_NAMES
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    try:
        model = get_model()

        input_data = pd.DataFrame(
            [customer.model_dump()],
            columns=FEATURE_NAMES
        )

        result = model.predict(input_data)

        return {"prediction": float(result[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    try:
        if not request.customers:
            raise HTTPException(
                status_code=400,
                detail="At least one customer is required"
            )

        model = get_model()

        input_data = pd.DataFrame(
            [customer.model_dump() for customer in request.customers],
            columns=FEATURE_NAMES
        )

        results = model.predict(input_data)

        return {
            "predictions": [float(value) for value in results],
            "total_customers": len(results)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))