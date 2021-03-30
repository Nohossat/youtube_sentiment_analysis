from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_infer():
    response = client.post("/",
                           json={"msg": "J'ai adoré ce restaurant"})

    assert response.status_code == 200
    assert response.json() == {"prediction": "Positive"}


def test_train():
    response = client.post("/train",
                           json={"name": "test_model",
                                 "estimator": "SVC",
                                 "cv": False})

    assert response.status_code == 200
    assert round(response.json()['accuracy'], 3) == 0.857


def test_train_false_model():
    response = client.post("/train",
                           json={"name": "test_model",
                                 "estimator": "SVC-fake",
                                 "cv": False})

    assert response.status_code == 200
    assert response.json() == {"res": "The model isn't registered in the API. You can choose between LGBM,SVC"}


def test_train_false_dataset():
    response = client.post("/train",
                           json={"name": "test_model",
                                 "estimator": "SVC",
                                 "data_path": "../../data/comments-fake.csv",
                                 "cv": False})

    assert response.status_code == 200
    assert response.json() == {"res": "The dataset doesn't exist"}


def test_models_available():
    response = client.get("/models")

    assert response.status_code == 200
    assert len(response.json()) == 11
