import os

import joblib
import pandas as pd
import neptune.new as neptune

import nohossat_cas_pratique
from nohossat_cas_pratique.preprocessing import split_data
from nohossat_cas_pratique.monitor import activate_monitoring, create_exp, record_metadata, save_artifact
from nohossat_cas_pratique.scoring import compute_metrics

module_path = os.path.dirname(os.path.dirname(os.path.dirname(nohossat_cas_pratique.__file__)))
data_path = os.path.join(module_path, "data", "comments.csv")
model_path = os.path.join(module_path, "models", "sentiment_pipe.joblib")


def test_activate_monitoring():
    project = activate_monitoring(os.getenv('NEPTUNE_USER'), os.getenv('NEPTUNE_PROJECT'))
    assert isinstance(project, neptune.run.Run), "This object should be an instance of Run"


def test_create_exp():
    run = activate_monitoring(os.getenv('NEPTUNE_USER'), os.getenv('NEPTUNE_PROJECT'))
    run["name"] = "test_create_exp"
    hyper_params = {"C": [10, 50]}
    tags = ["test"]
    exp = create_exp(hyper_params, tags, run)

    assert exp == None


def test_record_metadata():
    model_path = os.path.join(module_path, "models", "SVC_solo.joblib")
    run = activate_monitoring(os.getenv('NEPTUNE_USER'), os.getenv('NEPTUNE_PROJECT'))
    run["name"] = "test_record_metadata"
    run['sys/tags'].add(["test", "pytest"])

    with open(model_path, "rb") as f:
        model = joblib.load(f)
        data = pd.read_csv(data_path)
        X, y = split_data(data)
        metrics = compute_metrics(X, y, model)
        recording = record_metadata(metrics, run)
        assert recording == None


def test_save_artifact():
    run = activate_monitoring(os.getenv('NEPTUNE_USER'), os.getenv('NEPTUNE_PROJECT'))
    run["name"] = "test_save_artifact"
    hyper_params = {"C": [10, 50]}
    tags = ["test"]
    create_exp(hyper_params, tags, run)
    res = save_artifact(data_path, model_path, run)

    assert res == None
