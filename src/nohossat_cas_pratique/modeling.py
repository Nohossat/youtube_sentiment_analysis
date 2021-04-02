import joblib
import logging
import os

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import nohossat_cas_pratique
from nohossat_cas_pratique.preprocessing import NLPCleaner
from nohossat_cas_pratique.logging_app import start_logging

import spacy

nlp = spacy.load("fr_core_news_sm")

# config logging
module_path = os.path.dirname(os.path.dirname(os.path.dirname(nohossat_cas_pratique.__file__)))
start_logging(module_path)


def get_model(model_file: str):
    """
    Load a model
    :param model_file: path to the model joblib
    :return: a Scikit Learn estimator
    """

    try:
        with open(model_file, "rb") as f:
            model = joblib.load(f)
            return model
    except FileNotFoundError as e:
        logging.error(e)
        raise FileNotFoundError("The model doesn't exist")


def create_pipeline(params: dict = None, model_estimator: callable = SVC):
    """
    Create a Scikit Learn Pipeline with text preprocessing, tf-idf vectorizer and final estimator
    :param params: Parameters to pass to the estimator
    :param model_estimator: Scikit Learn Estimator
    :return: a full-fledged Scikit Learn pipeline
    """
    if params is None:
        params = {}

    pipe = Pipeline([('nlp_clean', NLPCleaner()),
                     ('tf-idf', TfidfVectorizer()),
                     ('clf', model_estimator(**params))])

    return pipe


def run_grid_search(model: object, params: dict, data: tuple, metrics: list, refit: str):
    """
    Run a grid search model with a dict of hyperparameters
    :param model: Scikit Learn Estimator
    :param params: hyperparameters to test
    :param data: features and label
    :param metrics : list of metrics names to compute during CV
    :param refit : metric to use to choose the best model after grid search
    :return: a Scikit Learn Pipeline
    """

    X, y = data
    grid_model = GridSearchCV(model, params, scoring=metrics, refit=refit, n_jobs=-1, verbose=1)
    grid_model.fit(X, y)

    return grid_model
