from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import neptune

import os
import re
import joblib

from preprocessing import split_data
from modeling import get_model
from scoring import compute_metrics_cv
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from monitoring import record_metadata

app = FastAPI()

class Comment(BaseModel):
    msg: str
    model_dirpath: str = "../models"
    model: str = "sentiment_pipe"


class Model(BaseModel):
    name: str
    estimator: str
    data_path: str = "../data/comments.csv"
    cv: bool = True


class Grid(BaseModel):
    data_path : str = "../data/comments.csv"
    base_model_name : str = "sentiment_pipe"
    grid_model_name : str = "grid_search_api_sentiment_pipe"
    model_estimator: str = "SVC"


@app.post("/")
async def infer(comment: Comment):
    res = None

    try:
        # get a model
        model = joblib.load(f"{comment.model_dirpath}/{comment.model}.joblib")
        pred = model.predict([comment.msg])[0]
        prediction = "Positive" if pred else "Negative"
        res = {"prediction": prediction}
    except FileNotFoundError as e:
        res = {"result": f"The model doesn't exist: {e}"}

    return res


@app.post("/train")
async def train(params: Model):
    neptune.init(project_qualified_name='nohossat/youtube-sentiment-analysis')

    neptune.create_experiment(
        name='sentiment-analysis',
        upload_source_files=['*.py', 'requirements.txt'],
        tags=[params.estimator, "solo"],
        send_hardware_metrics=True
    )

    # get data
    data = pd.read_csv(params.data_path)
    X, y = split_data(data)

    models = {
        "LGBM": LGBMClassifier,
        "SVC": SVC
    }

    if params.estimator not in models.keys():
        return {"res" : f"The model isn't registered in the API. You can choose between {list(models.keys())}"}

    # run model
    model = get_model(model_estimator=models[params.estimator], data=(X, y))

    # save model
    model_file = f"../models/{params.name}.joblib"
    joblib.dump(model, model_file)

    # get metrics and log them in Neptune
    metrics = record_metadata(X, y, model, params.data_path, params.name, cv=params.cv)

    # TODO return results by email
    return metrics


@app.get("/models")
async def get_available_models():
    """
        display available models
        :return: list
        """
    main_folder = "../models/"
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
async def report():
    """
    Get Report Board as a Pandas Dataframe converted to JSON
    :return: JSON
    """
    project = neptune.init('nohossat/youtube-sentiment-analysis')

    data = project.get_leaderboard()
    result = data.to_json(orient="split")
    return result
