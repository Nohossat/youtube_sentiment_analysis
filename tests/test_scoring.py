import os

import pandas as pd
import joblib

import nohossat_cas_pratique
from nohossat_cas_pratique.preprocessing import split_data
from nohossat_cas_pratique.scoring import compute_metrics_cv, compute_metrics

module_path = os.path.dirname(os.path.dirname(os.path.dirname(nohossat_cas_pratique.__file__)))
data_path = os.path.join(module_path, "data", "comments.csv")
model_path = os.path.join(module_path, "models", "sentiment_pipe.joblib")


def test_compute_metrics_cv():
    data = pd.read_csv(data_path)
    X, y = split_data(data)

    with open(model_path, "rb") as f:
        model = joblib.load(f)
        scores = compute_metrics_cv(X, y, model)

        assert round(scores['test_accuracy'], 3) == 0.865


def test_compute_metrics():
    data = pd.read_csv(data_path)
    X, y = split_data(data)

    with open(model_path, "rb") as f:
        model = joblib.load(f)
        scores = compute_metrics(X, y, model)

        assert round(scores['accuracy'], 3) == 0.717
