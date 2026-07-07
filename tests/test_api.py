from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_home():

    response = client.get("/")

    assert response.status_code == 200

    assert "HariMLOps" in response.json()["message"]


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


def test_prediction():

    sample = {

        "age":63,

        "sex":1,

        "cp":3,

        "trestbps":145,

        "chol":233,

        "fbs":1,

        "restecg":0,

        "thalach":150,

        "exang":0,

        "oldpeak":2.3,

        "slope":0,

        "ca":0,

        "thal":1

    }

    response = client.post("/predict", json=sample)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result

    assert "probability" in result