from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel
from fastapi.security import HTTPBasic, HTTPBasicCredentials

import os
import re
import joblib
import secrets
import logging

from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd

import nohossat_cas_pratique
from nohossat_cas_pratique.preprocessing import split_data
from nohossat_cas_pratique.modeling import create_pipeline
from nohossat_cas_pratique.monitor import record_metadata, activate_monitoring, create_exp, save_artifact
from nohossat_cas_pratique.logging_app import start_logging
from nohossat_cas_pratique.scoring import compute_metrics

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
    neptune_log: bool = True


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
        logging.error("Incorrect username or password")
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
        logging.error(e)
        res = {"result": f"The model doesn't exist: {e}"}

    return res


@app.post("/train")
async def train(params: Model, credentials: HTTPBasicCredentials = Depends(validate_access)):
    try:
        data = pd.read_csv(params.data_path)
    except FileNotFoundError as e:
        logging.error(e)
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
    run = None
    if params.neptune_log:
        run = activate_monitoring(os.getenv('NEPTUNE_USER'), os.getenv('NEPTUNE_PROJECT'))
        tags = [params.estimator, "solo"]

        create_exp(None, tags, run)

    # run model
    hyper_params = {}
    if params.estimator == "SVC":
        hyper_params["probability"] = True
    model = create_pipeline(model_estimator=models[params.estimator], params=hyper_params)
    model.fit(X_train, y_train)

    # save model
    model_file = os.path.join(module_path, "models", f"{params.name}.joblib")
    joblib.dump(model, model_file)

    # get metrics and log them in Neptune
    metrics = compute_metrics(X_train, y_train, model)

    if run is not None:
        record_metadata(metrics, run)
        save_artifact(data_path=params.data_path, model_file=model_file, run=run)

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
    project = activate_monitoring(os.getenv('NEPTUNE_USER'), os.getenv('NEPTUNE_PROJECT'))

    data = project.get_leaderboard()
    result = data.to_json(orient="split")
    return result
