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


def get_model(model_file: str = None,
              model_estimator: object = SVC,
              model_hyper_params: dict = None,
              data: tuple = None):
    """
    Load or create a model
    :param model_file: path to the model joblib
    :param model_name: model name used for saving
    :param model_estimator: Scikit Learn estimator
    :param model_hyper_params: hyper parameters to pass to the model_fct
    :param data : features and label to have the model fitted on
    :return: a Scikit Learn estimator
    """

    if model_file is not None:
        try:
            with open(model_file, "rb") as f:
                model = joblib.load(f)
                return model
        except FileNotFoundError as e:
            logging.error(e)
            raise FileNotFoundError("The model doesn't exist")

    if data is None:
        logging.error("Missing data values - please provide X and y values as a tuple")
        raise ValueError("Missing data values - please provide X and y values as a tuple")

    if model_estimator is not None:
        try:
            model = run_model(params=model_hyper_params, data=data, model_estimator=model_estimator)
            return model
        except Exception as e:
            logging.error(e)

    if model_file is None and model_estimator is None:
        logging.error("Please provide a valid model joblib or the name of a Scikit Learn Estimator")
        raise ValueError("Please provide a valid model joblib or the name of a Scikit Learn Estimator")


def create_pipeline(params: dict = None, model_estimator: callable = SVC):
    if params is None:
        params = {}

    pipe = Pipeline([('nlp_clean', NLPCleaner()),
                     ('tf-idf', TfidfVectorizer()),
                     ('clf', model_estimator(**params))])

    return pipe


def run_model(params: dict = None, data: tuple = None, model_estimator: callable = SVC):
    """
    Run a TF-IDF + SVC pipeline with pre-defined hyper parameters
    :param params: Hyperparameters to the SVC estimator
    :param data: features and label to have the model fitted on
    :param model_estimator: Scikit Learn estimator to fit
    :param model_name: Model name used for saving
    :return: a Scikit Learn Pipeline
    """

    X, y = data

    pipe = Pipeline([('nlp_clean', NLPCleaner()),
                     ('tf-idf', TfidfVectorizer()),
                     ('clf', model_estimator())])

    if params is not None:
        pipe = Pipeline([('nlp_clean', NLPCleaner()),
                         ('tf-idf', TfidfVectorizer()),
                         ('clf', model_estimator(**params))])

    # run model
    pipe.fit(X, y)
    return pipe


def run_grid_search(model: object, params: dict, data: tuple, metrics: list, refit: str):
    """
    Run a grid search model with a dict of hyperparameters
    :param model: Scikit Learn Estimator
    :param params: hyperparameters to test
    :param data: features and label
    :param model_name: Model name used for saving
    :return: a Scikit Learn Pipeline
    """

    X, y = data
    grid_model = GridSearchCV(model, params, scoring=metrics, refit=refit, n_jobs=-1, verbose=1)
    grid_model.fit(X, y)

    print(grid_model.cv_results_)

    return grid_model
