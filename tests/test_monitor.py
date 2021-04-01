import os

import joblib
import pandas as pd
import neptune

import nohossat_cas_pratique
from nohossat_cas_pratique.preprocessing import split_data
from nohossat_cas_pratique.monitor import activate_monitoring, create_exp, record_metadata, save_artifact
from nohossat_cas_pratique.scoring import compute_metrics

module_path = os.path.dirname(os.path.dirname(os.path.dirname(nohossat_cas_pratique.__file__)))
data_path = os.path.join(module_path, "data", "comments.csv")
model_path = os.path.join(module_path, "models", "sentiment_pipe.joblib")


def test_activate_monitoring():
    project = activate_monitoring(os.getenv('NEPTUNE_USER'), os.getenv('NEPTUNE_PROJECT'))
    assert isinstance(project, neptune.projects.Project), "This object should be an instance of Project"


def test_create_exp():
    activate_monitoring(os.getenv('NEPTUNE_USER'), os.getenv('NEPTUNE_PROJECT'))
    hyper_params = {"C": [10, 50]}
    tags = ["test"]
    exp = create_exp(hyper_params, tags)

    assert exp == None


def test_record_metadata():
    model_path = os.path.join(module_path, "models", "SVC_solo.joblib")

    with open(model_path, "rb") as f:
        model = joblib.load(f)
        data = pd.read_csv(data_path)
        X, y = split_data(data)
        metrics = compute_metrics(X, y, model)
        new_metrics = record_metadata(metrics)
        assert metrics == new_metrics


def test_save_artifact():
    activate_monitoring(os.getenv('NEPTUNE_USER'), os.getenv('NEPTUNE_PROJECT'))
    hyper_params = {"C": [10, 50]}
    tags = ["test"]
    create_exp(hyper_params, tags)
    res = save_artifact(data_path, model_path)

    assert res == None
