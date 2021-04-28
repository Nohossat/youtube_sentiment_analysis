from fastapi.testclient import TestClient
from nohossat_cas_pratique.main import app

client = TestClient(app)

HTTP_AUTH = "Basic Y2FzX3ByYXRpcXVlX25vbm86eW91dHViZQ=="
PATH_TRAIN = "/train"


def test_infer():
    response = client.post("/",
                           json={"msg": ["J'ai adoré ce restaurant", "J'ai détesté cet hôtel."]})

    assert response.status_code == 200
    assert response.json() == {"prediction": ["Positive", "Positive"]}


def test_train():
    response = client.post(PATH_TRAIN,
                           headers={"Authorization": HTTP_AUTH},
                           json={"model_name": "test_model_api",
                                 "estimator": "SVC",
                                 "cv": False,
                                 "neptune_log": False})

    assert response.status_code == 200
    assert round(response.json()['test/accuracy'], 3) == 0.857


def test_train_false_model():
    response = client.post(PATH_TRAIN,
                           headers={"Authorization": HTTP_AUTH},
                           json={"model_name": "test_model_api",
                                 "estimator": "SVC-fake",
                                 "cv": False})

    assert response.status_code == 200
    assert response.json() == {"res": "The model isn't registered in the API. You can choose between LGBM,SVC"}


def test_train_false_dataset():
    response = client.post(PATH_TRAIN,
                           headers={"Authorization": HTTP_AUTH},
                           json={"model_name": "test_model_api",
                                 "estimator": "SVC",
                                 "data_path": "../../data/comments-fake.csv",
                                 "cv": False})

    assert response.status_code == 200
    assert response.json() == {"res": "Can't load data"}


def test_grid_train_model():
    response = client.post("/grid_train",
                           headers={"Authorization": HTTP_AUTH},
                           json={"model_name": "test_model_api",
                                 "estimator": "SVC",
                                 "cv": False,
                                 "neptune_log": False})

    assert response.status_code == 200
    assert round(response.json()['test/accuracy'], 3) == 0.857


def test_models_available():
    response = client.get("/models",
                          headers={"Authorization": HTTP_AUTH})

    assert response.status_code == 200
    assert isinstance(response.json(), list), "It should be a list of current models available in the API"
