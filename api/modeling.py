import joblib
import logging

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from preprocessing import clean_sentences

import re

from stop_words import get_stop_words
from nltk.stem.snowball import SnowballStemmer
import spacy

nlp = spacy.load("fr_core_news_sm")


class NLPCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def clean_sentences(self, sentence):
        # stemming : it is done during tokenization
        stemmer = SnowballStemmer(language='french')

        def remove_stop_words(phrase):
            # tokenize
            doc = nlp(phrase)
            pattern_token = re.compile(r"(?u)\b\w\w+\b")
            tokens = [token.text for token in doc if pattern_token.match(token.text)]

            # get stop words
            french_stop_words = get_stop_words('fr')

            # remove stop_words
            clean_sentence = []
            for token in tokens:
                if token not in french_stop_words:
                    clean_sentence.append(token)
            return clean_sentence

        def stem_sentences(tokens):
            return ' '.join([stemmer.stem(token) for token in tokens])

        sent = remove_stop_words(sentence)
        return stem_sentences(sent)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = [self.clean_sentences(sent) for sent in X]
        return X_


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
            logging.info(e)

    if data is None:
        raise Exception("Missing data values - please prove X and y values as a tuple")

    if model_estimator is not None:
        try:
            model = run_model(params=model_hyper_params, data=data, model_estimator=model_estimator)
            return model
        except Exception as e:
            print(e)
            logging.info(e)

    if model_file is None and model_estimator is None:
        raise Exception("Please provid a valid model joblib or the name of a Scikit Learn Estimator")


def create_pipeline(params: dict = None, model_estimator: object = SVC):
    if params is None:
        params = {}

    pipe = Pipeline([('tf-idf', TfidfVectorizer(analyzer=clean_sentences)),
                     ('clf', model_estimator(**params))])

    return pipe


def run_model(params: dict = None, data: tuple = None, model_estimator: object = SVC):
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


def run_grid_search(model: object, params: dict, data: tuple):
    """
    Run a grid search model with a dict of hyperparameters
    :param model: Scikit Learn Estimator
    :param params: hyperparameters to test
    :param data: features and label
    :param model_name: Model name used for saving
    :return: a Scikit Learn Pipeline
    """

    X, y = data
    grid_model = GridSearchCV(model, params, scoring=['precision', 'recall', 'roc_auc'], refit="roc_auc", n_jobs=-1, verbose=1)
    grid_model.fit(X, y)

    return grid_model, grid_model.best_params_
