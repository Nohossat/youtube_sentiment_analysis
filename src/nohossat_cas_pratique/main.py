from fastapi import Depends, FastAPI, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from typing import List, Dict

from dotenv import load_dotenv
import joblib
import logging
import neptune.new as neptune
import os
import re
import pandas as pd
import secrets

from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import yaml

import nohossat_cas_pratique
from nohossat_cas_pratique.emailing import send_email, send_email_user_creation
from nohossat_cas_pratique.logging_app import start_logging
from nohossat_cas_pratique.modeling import create_pipeline, run_grid_search
from nohossat_cas_pratique.monitor import record_metadata, activate_monitoring, create_exp, save_artifact, get_project
from nohossat_cas_pratique.preprocessing import split_data
from nohossat_cas_pratique.scoring import compute_metrics, compute_metrics_cv, get_grid_search_best_metrics
from nohossat_cas_pratique.user_creation import get_existing_user

# load API description
description_file = os.path.join(os.path.dirname(nohossat_cas_pratique.__file__), "description.yml")
api_desc = yaml.load(open(description_file), Loader=yaml.FullLoader)

app = FastAPI(title=api_desc["title"],
              description=api_desc["description"],
              version=api_desc["version"],
              openapi_tags=api_desc["tags"])

security = HTTPBasic()

# config logging
module_path = os.path.dirname(os.path.dirname(os.path.dirname(nohossat_cas_pratique.__file__)))
start_logging(module_path)

# load environment variables
load_dotenv()


class Comment(BaseModel):
    msg: List[str]
    model: str = "grid_search_SVC"


class User(BaseModel):
    name: str
    email: str


class Model(BaseModel):
    model_name: str
    estimator: str
    data_path: str = os.path.join(module_path, "data", "comments.csv")
    comment_col : str = "comment"
    target_col: str = "sentiment"
    cv: bool = False
    neptune_log: bool = True
    tags: List[str] = []
    email_address: str = None


class Grid(BaseModel):
    data_path: str = os.path.join(module_path, "data", "comments.csv")
    comment_col : str = "comment"
    target_col: str = "sentiment"
    model_name: str = "grid_search_api_sentiment_pipe"
    estimator: str = "SVC"
    parameters: Dict[str, list] = {}
    neptune_log: bool = True
    tags: list = []
    email_address: str = None


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

    user = get_existing_user(credentials.username)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username doesn't exist",
            headers={"WWW-Authenticate": "Basic"},
        )

    correct_username = secrets.compare_digest(credentials.username, user[0])
    correct_password = secrets.compare_digest(credentials.password, user[2])

    if not (correct_username and correct_password):
        logging.error("Incorrect username or password")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def load_data(data_path, comment_col, target_col):
    """
    Validate the existence of the dataset
    :param data_path:
    :return:
    """
    try:
        data = pd.read_csv(data_path)
        X, y = split_data(data, comment_col=comment_col, target_col=target_col)
        return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    except FileNotFoundError as e:
        logging.error(e)
        raise FileNotFoundError("The dataset doesn't exist.")
    except ValueError as e:
        logging.error(e)
        raise ValueError("The comment or target column names are incorrect.")


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


def run_model(params, run, X_train, X_test, y_train, y_test):
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

    res = "Not sent"

    if run is not None:
        if params.cv:
            record_metadata(cv_metrics, run)
        record_metadata(metrics, run)
        save_artifact(data_path=params.data_path, model_file=model_file, run=run)

        # notify user
        if params.email_address is not None:
            url = f"{run._backend.get_display_address()}/{os.getenv('NEPTUNE_USER')}/{os.getenv('NEPTUNE_PROJECT')}/e/{run['sys/id'].fetch()}"
            res = send_email(url, params.email_address)

        run.stop()

    return {'metrics' : metrics, "email_sent" : res}


def grid_run_model(params, run, X_train, X_test, y_train, y_test):
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

    res = "Not sent"

    if run is not None:
        record_metadata(cv_results, run)
        record_metadata(metrics, run)
        save_artifact(data_path=params.data_path, model_file=model_file, run=run)

        # notify user
        if params.email_address is not None:
            url = f"{run._backend.get_display_address()}/{os.getenv('NEPTUNE_USER')}/{os.getenv('NEPTUNE_PROJECT')}/e/{run['sys/id'].fetch()}"
            res = send_email(url, params.email_address)

        run.stop()

    return {'metrics' : metrics, "email_sent" : res}


@app.post("/create_user", tags=["create new account"])
async def create(user: User):
    send_email_user_creation(user.name, user.email)


@app.post("/", tags=["predict"])
async def infer(comment: Comment):
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
        preds = model.predict(comment.msg)
        predictions = []

        for pred in preds:
            prediction = "Positive" if pred else "Negative"
            predictions.append(prediction)

        if len(predictions) == 1:
            predictions = predictions[0]
        res["prediction"] = predictions
    else :
        res["error"] = "Couldn't make a prediction : the model hasn't been found and no default model"

    return res


@app.post("/train", tags=["train"])
async def train(params: Model, background_tasks: BackgroundTasks, credentials: HTTPBasicCredentials = Depends(validate_access)):
    """
    Choose an estimator, train it and collect metrics in Neptune.ai
    """
    try:
        X_train, X_test, y_train, y_test = load_data(params.data_path, params.comment_col, params.target_col)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    params.estimator = params.estimator.upper()
    if params.estimator not in MODELS.keys():
        raise HTTPException(status_code=404, detail=f"The model isn't registered in the API. You can choose between {','.join(list(MODELS.keys()))}")

    # start logging
    run = None
    if params.neptune_log:
        try:
            run = activate_monitoring(os.getenv('NEPTUNE_USER'), os.getenv('NEPTUNE_PROJECT'))
            print(run['sys/id'].fetch())
            params.tags.extend([params.estimator, "solo"])
            create_exp(None, params.tags, run)
        except neptune.exceptions.NeptuneInvalidApiTokenException as e:
            raise HTTPException(status_code=400, detail="Not currently connected to NEPTUNE.ai. Ask the developer to provide its user access.")

    # run modeling in the background
    background_tasks.add_task(run_model, params, run, X_train, X_test, y_train, y_test)
    return {'res' : "The model is running. You will receive a mail if you provided your email address."}


@app.post("/grid_train", tags=["grid_train"])
async def grid_train(params: Grid, background_tasks: BackgroundTasks, credentials: HTTPBasicCredentials = Depends(validate_access)):
    """
    Choose an estimator, and hyper-parameters to optimize for a GridSearchCV. Results can be recorded in Neptune.ai.
    """

    try:
        X_train, X_test, y_train, y_test = load_data(params.data_path, params.comment_col, params.target_col)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    params.estimator = params.estimator.upper()
    if params.estimator not in MODELS.keys():
        raise HTTPException(status_code=400, detail=f"The model isn't registered in the API. You can choose between {','.join(list(MODELS.keys()))}")

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
        try:
            run = activate_monitoring(os.getenv('NEPTUNE_USER'), os.getenv('NEPTUNE_PROJECT'))
            params.tags.extend([params.estimator, "grid"])
            create_exp(params.parameters, params.tags, run)
        except neptune.exceptions.NeptuneInvalidApiTokenException as e:
            raise HTTPException(status_code=400, detail="Not currently connected to NEPTUNE.ai. Ask the developer to provide its user access.")

    # run modeling in the background
    background_tasks.add_task(grid_run_model, params, run, X_train, X_test, y_train, y_test)
    return {'res' : "The model is running. You will receive a mail if you provided your email address."}


@app.get("/models", tags=["models"])
async def get_available_models(credentials: HTTPBasicCredentials = Depends(validate_access)):
    """
    Get available models in the models folder
    """
    return get_models()


@app.get("/report", tags=["reports"])
async def report(credentials: HTTPBasicCredentials = Depends(validate_access)):
    """
    Get Report Board as a Pandas Dataframe converted to JSON
    :return: JSON
    """

    try:
        project = get_project(os.getenv('NEPTUNE_USER'), os.getenv('NEPTUNE_PROJECT'))

        # Get dashboard with runs contributed by 'sophia'
        df = project.get_runs_table().as_pandas()
        result = df.to_json(orient="split")
        return result
    except neptune.exceptions.NeptuneInvalidApiTokenException as e:
            raise HTTPException(status_code=400, detail="Not currently connected to NEPTUNE.ai. Ask the developer to provide its user access.")
