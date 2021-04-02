from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict
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
from nohossat_cas_pratique.modeling import create_pipeline, run_grid_search
from nohossat_cas_pratique.monitor import record_metadata, activate_monitoring, create_exp, save_artifact
from nohossat_cas_pratique.logging_app import start_logging
from nohossat_cas_pratique.scoring import compute_metrics, compute_metrics_cv, get_grid_search_best_metrics

app = FastAPI()
security = HTTPBasic()

# config logging
module_path = os.path.dirname(os.path.dirname(os.path.dirname(nohossat_cas_pratique.__file__)))
start_logging(module_path)


class Comment(BaseModel):
    msg: str
    model: str = "sentiment_pipe"


class Model(BaseModel):
    model_name: str
    estimator: str
    data_path: str = os.path.join(module_path, "data", "comments.csv")
    cv: bool = False
    neptune_log: bool = True
    tags: List[str] = []


class Grid(BaseModel):
    data_path: str = os.path.join(module_path, "data", "comments.csv")
    model_name: str = "grid_search_api_sentiment_pipe"
    estimator: str = "SVC"
    parameters: Dict[str, list] = {}
    neptune_log: bool = True
    tags: list = []


MODELS = {
        "LGBM": {"fct": LGBMClassifier,
                 "default_hyperparams": {
                     "clf__max_depth": [3, 10, -1],
                     "clf__n_estimators": [50, 100, 200],
                     "clf__class_weight": ['balanced'],
                     "clf__random_state": [43]}},
        "SVC": {"fct": SVC,
                "default_hyperparams": {'clf__C': [5, 10, 100],
                                'clf__class_weight': ['balanced', {0: 0.37, 1: 0.63}],
                                'clf__kernel': ['poly', 'rbf', 'sigmoid'],
                                'clf__gamma': [0.001, "scale", "auto"]}}
    }

best_model_name = "grid_search_SVC"

def validate_access(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Check the validity of the user access
    :param credentials:
    :return: the name passed in the credentials
    """

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


def load_data(data_path):
    """
    Validate the existence of the dataset
    :param data_path:
    :return:
    """
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError as e:
        logging.error(e)
        return {"res": "The dataset doesn't exist"}

    X, y = split_data(data)
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


def get_models():
    """
    Fetch available models in the models folder
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


@app.post("/")
async def infer(comment: Comment, credentials: HTTPBasicCredentials = Depends(validate_access)):
    """
    Predict a comment polarity (Positive / Negative)
    :param comment:
    :param credentials:
    :return: The response from the model - Positive / Negative
    """
    model = None
    model_dirpath = os.path.join(module_path, "models")
    res = {}

    try:
        model = joblib.load(f"{model_dirpath}/{comment.model}.joblib")
    except FileNotFoundError as e:
        logging.error(e)
        if best_model_name in get_models():
            model = joblib.load(f"{model_dirpath}/{best_model_name}.joblib")
            res["message"] = "Prediction from default model"

    # inference
    if model is not None:
        pred = model.predict([comment.msg])[0]
        prediction = "Positive" if pred else "Negative"
        res["prediction"] = prediction
    else :
        res["error"] = "Couldn't make a prediction : the model hasn't been found and no default model"

    return res


@app.post("/train")
async def train(params: Model, credentials: HTTPBasicCredentials = Depends(validate_access)):
    """
    Choose an estimator, train it and collect metrics in Neptune.ai
    """
    X_train, X_test, y_train, y_test = load_data(params.data_path)

    if params.estimator not in MODELS.keys():
        return {"res": f"The model isn't registered in the API. You can choose between {','.join(list(MODELS.keys()))}"}

    # start logging
    run = None
    if params.neptune_log:
        run = activate_monitoring(os.getenv('NEPTUNE_USER'), os.getenv('NEPTUNE_PROJECT'))
        params.tags.extend([params.estimator, "solo"])
        create_exp(None, params.tags, run)

    # run model
    hyper_params = {}
    if params.estimator == "SVC":
        hyper_params["probability"] = True
    model = create_pipeline(model_estimator=MODELS[params.estimator]["fct"], params=hyper_params)
    model.fit(X_train, y_train)

    # save model
    model_file = os.path.join(module_path, "models", f"{params.model_name}.joblib")
    joblib.dump(model, model_file)

    # get CV metrics and test metrics and log them in Neptune
    if params.cv:
        cv_metrics = compute_metrics_cv(X_train, y_train, model)
    metrics = compute_metrics(X_test, y_test, model)

    if run is not None:
        if params.cv:
            record_metadata(cv_metrics, run)
        record_metadata(metrics, run)
        save_artifact(data_path=params.data_path, model_file=model_file, run=run)
        print(run.print_structure())
        run.stop()

    # TODO return results by email

    return metrics


@app.post("/grid_train")
async def grid_train(params: Grid, credentials: HTTPBasicCredentials = Depends(validate_access)):
    """
    Choose an estimator, and hyper-parameters to optimize for a GridSearchCV. Results can be recorded in Neptune.ai.
    """
    X_train, X_test, y_train, y_test = load_data(params.data_path)

    if params.estimator not in MODELS.keys():
        return {"res": f"The model isn't registered in the API. You can choose between {','.join(list(MODELS.keys()))}"}

    if params.parameters is None:
        params.parameters = MODELS[params.estimator]["default_hyperparams"]
    else:
        # ici il faut une validation des hyper paramètres
        params.parameters = {f"clf__{param}": liste for param, liste in params.parameters.items()}

    if params.estimator == "SVC":
        params.parameters["clf__probability"] = [True]

    # start logging
    run = None
    if params.neptune_log:
        run = activate_monitoring(os.getenv('NEPTUNE_USER'), os.getenv('NEPTUNE_PROJECT'))
        params.tags.extend([params.estimator, "grid"])
        create_exp(params.parameters, params.tags, run)

    # run model
    list_metrics = ['precision', 'recall', 'accuracy', 'f1_weighted', 'roc_auc']
    refit = "roc_auc"
    pipe = create_pipeline(model_estimator=MODELS[params.estimator]["fct"], params=None)

    model = run_grid_search(model=pipe,
                            params=params.parameters,
                            data=(X_train, y_train),
                            metrics=list_metrics,
                            refit=refit)

    # record best params
    if run is not None:
        run['best_params'] = model.best_params_

    # collect cv_results and test metrics
    cv_results = get_grid_search_best_metrics(model, list_metrics)
    metrics = compute_metrics(X_test, y_test, model)

    # save model
    model_file = os.path.join(module_path, "models", f"{params.model_name}.joblib")
    joblib.dump(model, model_file)

    if run is not None:
        record_metadata(cv_results, run)
        record_metadata(metrics, run)
        save_artifact(data_path=params.data_path, model_file=model_file, run=run)
        run.stop()

    # TODO return results by email
    return metrics


@app.get("/models")
async def get_available_models(credentials: HTTPBasicCredentials = Depends(validate_access)):
    """
    Get available models in the models folder
    """
    get_models()


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
