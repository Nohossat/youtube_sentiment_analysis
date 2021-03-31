from fastapi.testclient import TestClient
from nohossat_cas_pratique.main import app

client = TestClient(app)


def test_infer():
    response = client.post("/",
                           headers={"Authorization": "Basic Y2FzX3ByYXRpcXVlX25vbm86eW91dHViZQ=="},
                           json={"msg": "J'ai adoré ce restaurant"})

    assert response.status_code == 200
    assert response.json() == {"prediction": "Positive"}


def test_train():
    response = client.post("/train",
                           headers={"Authorization": "Basic Y2FzX3ByYXRpcXVlX25vbm86eW91dHViZQ=="},
                           json={"name": "test_model_api",
                                 "estimator": "SVC",
                                 "cv": False})

    assert response.status_code == 200
    assert round(response.json()['accuracy'], 3) == 1.0


def test_train_false_model():
    response = client.post("/train",
                           headers={"Authorization": "Basic Y2FzX3ByYXRpcXVlX25vbm86eW91dHViZQ=="},
                           json={"name": "test_model_api",
                                 "estimator": "SVC-fake",
                                 "cv": False})

    assert response.status_code == 200
    assert response.json() == {"res": "The model isn't registered in the API. You can choose between LGBM,SVC"}


def test_train_false_dataset():
    response = client.post("/train",
                           headers={"Authorization": "Basic Y2FzX3ByYXRpcXVlX25vbm86eW91dHViZQ=="},
                           json={"name": "test_model_api",
                                 "estimator": "SVC",
                                 "data_path": "../../data/comments-fake.csv",
                                 "cv": False})

    assert response.status_code == 200
    assert response.json() == {"res": "The dataset doesn't exist"}


def test_models_available():
    response = client.get("/models",
                          headers={"Authorization": "Basic Y2FzX3ByYXRpcXVlX25vbm86eW91dHViZQ=="})

    assert response.status_code == 200
    assert isinstance(response.json(), list), "It should be a list of current models available in the API"
