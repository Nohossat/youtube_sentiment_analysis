from fastapi.testclient import TestClient
from nohossat_cas_pratique.main import app

client = TestClient(app)

HTTP_AUTH = "Basic bm9ub19kZXY6MSNROFZDNjM4aTVeNnM="
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

    print(response.json())
    assert response.status_code == 200
    assert response.json() == {'res': 'The model is running. You will receive a mail if you provided your email address.'}


def test_train_false_model():
    response = client.post(PATH_TRAIN,
                           headers={"Authorization": HTTP_AUTH},
                           json={"model_name": "test_model_api",
                                 "estimator": "SVC-fake",
                                 "cv": False})

    assert response.status_code == 404
    assert response.json() == {"detail": "The model isn't registered in the API. You can choose between LGBM,SVC"}


def test_train_false_dataset():
    response = client.post(PATH_TRAIN,
                           headers={"Authorization": HTTP_AUTH},
                           json={"model_name": "test_model_api",
                                 "estimator": "SVC",
                                 "data_path": "../../data/comments-fake.csv",
                                 "cv": False})

    assert response.status_code == 400
    assert response.json() == {"detail": "The dataset doesn't exist."}


def test_grid_train_model():
    response = client.post("/grid_train",
                           headers={"Authorization": HTTP_AUTH},
                           json={"model_name": "test_model_api",
                                 "estimator": "SVC",
                                 "cv": False,
                                 "neptune_log": False})

    print(response.json())
    assert response.status_code == 200
    assert response.json() == {'res': 'The model is running. You will receive a mail if you provided your email address.'}


def test_models_available():
    response = client.get("/models",
                          headers={"Authorization": HTTP_AUTH})

    assert response.status_code == 200
    assert isinstance(response.json(), list), "It should be a list of current models available in the API"
