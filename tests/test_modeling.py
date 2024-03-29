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
from nohossat_cas_pratique.modeling import get_model, create_pipeline, run_grid_search

module_path = os.path.dirname(os.path.dirname(os.path.dirname(nohossat_cas_pratique.__file__)))
data_path = os.path.join(module_path, "data", "comments.csv")
model_path = os.path.join(module_path, "models", "sentiment_pipe.joblib")
FIRST_MSG = "the first step should be a cleaner"
SECOND_MSG = "the second step should be a tf-idf vectorizer"
THIRD_MSG = "the third step should be a SVC classifier"
FOURTH_MSG = "the third step should be a lgbm classifier"


def test_get_model_model_file():
    model = get_model(model_file=model_path)

    assert isinstance(model.steps[0][1], TfidfVectorizer), SECOND_MSG
    assert isinstance(model.steps[1][1], SVC), THIRD_MSG


def test_get_model_fake_model_file():
    fake_model_path = '../models/test_model.joblib'
    with pytest.raises(FileNotFoundError, match="The model doesn't exist"):
        get_model(model_file=fake_model_path)


def test_create_pipeline():
    pipe = create_pipeline()

    assert isinstance(pipe.steps[0][1], NLPCleaner), FIRST_MSG
    assert isinstance(pipe.steps[1][1], TfidfVectorizer), SECOND_MSG
    assert isinstance(pipe.steps[2][1], SVC), THIRD_MSG


def test_create_pipeline_params():
    params = {'C': 50, 'gamma': 0.01}
    pipe = create_pipeline(params=params)

    assert isinstance(pipe.steps[0][1], NLPCleaner), FIRST_MSG
    assert isinstance(pipe.steps[1][1], TfidfVectorizer), SECOND_MSG
    assert isinstance(pipe.steps[2][1], SVC), THIRD_MSG


def test_run_grid_search():
    data = pd.read_csv(data_path)
    X, y = split_data(data)
    params = {
        "clf__max_depth": [3],
        "clf__n_estimators": [50],
        "clf__class_weight": ['balanced'],
        "clf__random_state": [43]}

    pipe = create_pipeline(model_estimator=LGBMClassifier)
    list_metrics = ['precision', 'recall']
    refit = "precision"

    grid_pipe = run_grid_search(model=pipe, params=params, data=(X, y), metrics=list_metrics, refit=refit)
    print(grid_pipe)

    assert isinstance(grid_pipe, GridSearchCV), "Should be a grid search"
    assert isinstance(grid_pipe.estimator, Pipeline)
    assert grid_pipe.param_grid == {'clf__class_weight': ['balanced'],
                                    'clf__max_depth': [3],
                                    'clf__n_estimators': [50],
                                    'clf__random_state': [43]}

    assert grid_pipe.refit == 'precision'
