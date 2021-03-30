from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_read_main():
    response = client.post("/",
                           json={"msg": "J'ai adoré ce restaurant",
                                 "model_dirpath": "../../models"})

    assert response.status_code == 200
    assert response.json() == {"prediction": "Positive"}
