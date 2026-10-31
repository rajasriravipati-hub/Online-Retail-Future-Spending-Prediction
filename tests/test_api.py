from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "API is running" in response.json()["message"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_metadata():
    response = client.get("/metadata")
    assert response.status_code == 200
    assert response.json()["target"] == "FutureSpending"


def test_single_prediction():
    data = {
        "TotalSpent": 1000,
        "NumberOfOrders": 5,
        "TotalQuantity": 100,
        "AverageOrderValue": 200,
        "UniqueProducts": 20,
        "AverageUnitPrice": 5
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_batch_prediction():
    customer = {
        "TotalSpent": 1000,
        "NumberOfOrders": 5,
        "TotalQuantity": 100,
        "AverageOrderValue": 200,
        "UniqueProducts": 20,
        "AverageUnitPrice": 5
    }

    response = client.post(
        "/predict/batch",
        json={"customers": [customer, customer]}
    )

    assert response.status_code == 200
    assert response.json()["total_customers"] == 2
    assert len(response.json()["predictions"]) == 2


def test_invalid_input():
    data = {
        "TotalSpent": -100,
        "NumberOfOrders": 5,
        "TotalQuantity": 100,
        "AverageOrderValue": 200,
        "UniqueProducts": 20,
        "AverageUnitPrice": 5
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 422