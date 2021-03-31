import os

import pandas as pd
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import pytest

import nohossat_cas_pratique
from nohossat_cas_pratique.preprocessing import split_data, NLPCleaner
from nohossat_cas_pratique.modeling import get_model, create_pipeline, run_model, run_grid_search

module_path = os.path.dirname(os.path.dirname(os.path.dirname(nohossat_cas_pratique.__file__)))
data_path = os.path.join(module_path, "data", "comments.csv")
model_path = os.path.join(module_path, "models", "sentiment_pipe.joblib")


def test_get_model():
    data = pd.read_csv(data_path)
    X, y = split_data(data)
    model = get_model(model_estimator=SVC, data=(X, y))

    assert isinstance(model.steps[0][1], NLPCleaner), "the first step should be a cleaner"
    assert isinstance(model.steps[1][1], TfidfVectorizer), "the second step should be a tf-idf vectorizer"
    assert isinstance(model.steps[2][1], SVC), "the third step should be a SVC classifier"


def test_get_model_model_file():
    model = get_model(model_file=model_path)

    assert isinstance(model.steps[0][1], TfidfVectorizer), "the first step should be a tf-idf vectorizer"
    assert isinstance(model.steps[1][1], SVC), "the second step should be a SVC"


def test_get_model_fake_model_file():
    fake_model_path = '../models/test_model.joblib'
    with pytest.raises(FileNotFoundError, match="The model doesn't exist"):
        get_model(model_file=fake_model_path)


def test_get_model_none_data():
    data = None

    with pytest.raises(ValueError, match="Missing data values - please provide X and y values as a tuple"):
        get_model(model_estimator=SVC, data=data)


def test_get_model_no_model():
    data = pd.read_csv(data_path)
    X, y = split_data(data)

    with pytest.raises(ValueError, match="Please provide a valid model joblib or the name of a Scikit Learn Estimator"):
        get_model(model_file=None, model_estimator=None, data=(X, y))


def test_create_pipeline():
    pipe = create_pipeline()

    assert isinstance(pipe.steps[0][1], NLPCleaner), "the first step should be a cleaner"
    assert isinstance(pipe.steps[1][1], TfidfVectorizer), "the second step should be a tf-idf vectorizer"
    assert isinstance(pipe.steps[2][1], SVC), "the third step should be a SVC classifier"


def test_create_pipeline_params():
    params = {'C': 50, 'gamma': 0.01}
    pipe = create_pipeline(params=params)

    assert isinstance(pipe.steps[0][1], NLPCleaner), "the first step should be a cleaner"
    assert isinstance(pipe.steps[1][1], TfidfVectorizer), "the second step should be a tf-idf vectorizer"
    assert isinstance(pipe.steps[2][1], SVC), "the third step should be a SVC classifier"


def test_run_model():
    data = pd.read_csv(data_path)
    X, y = split_data(data)
    pipe = run_model(params=None, data=(X, y), model_estimator=LGBMClassifier)

    assert isinstance(pipe.steps[0][1], NLPCleaner), "the first step should be a cleaner"
    assert isinstance(pipe.steps[1][1], TfidfVectorizer), "the second step should be a tf-idf vectorizer"
    assert isinstance(pipe.steps[2][1], LGBMClassifier), "the third step should be a lgbm classifier"


def test_run_model_params():
    data = pd.read_csv(data_path)
    X, y = split_data(data)
    params = {'n_estimators': 50, 'random_state': 0}
    pipe = run_model(params=params, data=(X, y), model_estimator=LGBMClassifier)

    assert isinstance(pipe.steps[0][1], NLPCleaner), "the first step should be a cleaner"
    assert isinstance(pipe.steps[1][1], TfidfVectorizer), "the second step should be a tf-idf vectorizer"
    assert isinstance(pipe.steps[2][1], LGBMClassifier), "the third step should be a lgbm classifier"


def test_run_grid_search():
    data = pd.read_csv(data_path)
    X, y = split_data(data)
    params = {
        "clf__max_depth": [3],
        "clf__n_estimators": [50],
        "clf__class_weight": ['balanced'],
        "clf__random_state": [43]}

    pipe = create_pipeline(model_estimator=LGBMClassifier)

    grid_pipe = run_grid_search(model=pipe, params=params, data=(X, y))

    assert isinstance(grid_pipe[0], GridSearchCV), "Should be a grid search"
    assert isinstance(grid_pipe[0].estimator, Pipeline)
    assert grid_pipe[0].param_grid == {'clf__class_weight': ['balanced'],
                                    'clf__max_depth': [3],
                                    'clf__n_estimators': [50],
                                    'clf__random_state': [43]}

    assert grid_pipe[0].refit == 'roc_auc'
