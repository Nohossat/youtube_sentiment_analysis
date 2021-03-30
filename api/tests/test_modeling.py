import os

import pandas as pd
import joblib
from sklearn.svm import SVC

import api
from preprocessing import split_data
from modeling import get_model
from scoring import compute_metrics_cv, compute_metrics

module_path = os.path.dirname(os.path.dirname(api.__file__))
data_path = os.path.join(module_path, "data", "comments.csv")
model_path = os.path.join(module_path, "models", "sentiment_pipe.joblib")


def test_get_model():
    data = pd.read_csv(data_path)
    X, y = split_data(data)

    model = get_model(model_estimator=SVC, data=(X, y))

    assert model == None


def test_get_model_model_file():
    model = get_model(model_file=model_path)
    assert model == None


def test_get_model_fake_model_file():
    model_path = '../models/test_model.joblib'
    model = get_model(model_file=model_path)
    assert model == None
