from pydantic import BaseModel, Field
from typing import List


class CustomerFeatures(BaseModel):
    TotalSpent: float = Field(..., ge=0)
    NumberOfOrders: float = Field(..., ge=0)
    TotalQuantity: float = Field(..., ge=0)
    AverageOrderValue: float = Field(..., ge=0)
    UniqueProducts: float = Field(..., ge=0)
    AverageUnitPrice: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: float


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    total_customers: int