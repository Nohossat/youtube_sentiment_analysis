from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel
from fastapi.security import HTTPBasic, HTTPBasicCredentials

import os
import re
import joblib
import secrets

from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import neptune

import nohossat_cas_pratique
from nohossat_cas_pratique.preprocessing import split_data
from nohossat_cas_pratique.modeling import get_model, create_pipeline
from nohossat_cas_pratique.monitor import record_metadata
from nohossat_cas_pratique.logging_app import start_logging

app = FastAPI()
security = HTTPBasic()

# config logging
module_path = os.path.dirname(os.path.dirname(os.path.dirname(nohossat_cas_pratique.__file__)))
start_logging(module_path)


class Comment(BaseModel):
    msg: str
    model: str = "sentiment_pipe"


class Model(BaseModel):
    name: str
    estimator: str
    data_path: str = os.path.join(module_path, "data", "comments.csv")
    cv: bool = True


class Grid(BaseModel):
    data_path : str = os.path.join(module_path, "data", "comments.csv")
    base_model_name : str = "sentiment_pipe"
    grid_model_name : str = "grid_search_api_sentiment_pipe"
    model_estimator: str = "SVC"


def validate_access(credentials: HTTPBasicCredentials = Depends(security)):
    # export these into a MongoCollection
    correct_username = secrets.compare_digest(credentials.username, os.getenv('LOGIN'))
    correct_password = secrets.compare_digest(credentials.password, os.getenv('PASSWORD'))
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@app.post("/")
async def infer(comment: Comment, credentials: HTTPBasicCredentials = Depends(validate_access)):
    res = None

    try:
        # get a model
        model_dirpath = os.path.join(module_path, "models")
        model = joblib.load(f"{model_dirpath}/{comment.model}.joblib")
        pred = model.predict([comment.msg])[0]
        prediction = "Positive" if pred else "Negative"
        res = {"prediction": prediction}
    except FileNotFoundError as e:
        res = {"result": f"The model doesn't exist: {e}"}

    return res


@app.post("/train")
async def train(params: Model, credentials: HTTPBasicCredentials = Depends(validate_access)):
    try:
        data = pd.read_csv(params.data_path)
    except FileNotFoundError:
        return {"res": "The dataset doesn't exist"}

    X, y = split_data(data)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    models = {
        "LGBM": LGBMClassifier,
        "SVC": SVC
    }

    if params.estimator not in models.keys():
        return {"res": f"The model isn't registered in the API. You can choose between {','.join(list(models.keys()))}"}

    # start logging
    neptune.init(project_qualified_name='nohossat/youtube-sentiment-analysis')

    neptune.create_experiment(
        name='sentiment-analysis',
        upload_source_files=['*.py', 'requirements.txt'],
        tags=[params.estimator, "solo"],
        send_hardware_metrics=True
    )

    # run model
    # model = get_model(model_estimator=models[params.estimator], data=(X_train, y_train))

    hyper_params = {}
    if params.estimator == "SVC":
        hyper_params["probability"] = True
    model = create_pipeline(model_estimator=models[params.estimator], params=hyper_params)
    model.fit(X_train, y_train)

    # save model
    model_file = os.path.join(module_path, "models", f"{params.name}.joblib")
    joblib.dump(model, model_file)

    # get metrics and log them in Neptune
    metrics = record_metadata(X, y, model, cv=params.cv)

    # TODO return results by email
    return metrics


@app.get("/models")
async def get_available_models(credentials: HTTPBasicCredentials = Depends(validate_access)):
    """
        display available models
        :return: list
        """
    main_folder = os.path.join(module_path, "models")
    models = []
    for (_, _, filenames) in os.walk(main_folder):
        pattern = re.compile(r"(\w+)\.joblib")

        for filename in filenames:
            model = pattern.match(filename)

            if model is not None:
                models.append(model.group(1))
        break
    return models


@app.get("/report")
async def report(credentials: HTTPBasicCredentials = Depends(validate_access)):
    """
    Get Report Board as a Pandas Dataframe converted to JSON
    :return: JSON
    """
    project = neptune.init('nohossat/youtube-sentiment-analysis')

    data = project.get_leaderboard()
    result = data.to_json(orient="split")
    return result
