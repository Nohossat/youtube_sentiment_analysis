import argparse
import logging

import neptune
import pandas as pd
import os

import nohossat_cas_pratique
from nohossat_cas_pratique.preprocessing import split_data
from nohossat_cas_pratique.modeling import get_model, run_grid_search, create_pipeline
from nohossat_cas_pratique.monitor import save_artifact, record_metadata, create_exp

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import joblib

module_path = os.path.dirname(os.path.dirname(os.path.dirname(nohossat_cas_pratique.__file__)))

if __name__ == "__main__":

    # config logging
    try:
        logging.basicConfig(filename=os.path.join(module_path, "logs", "monitoring.log"), level=logging.DEBUG)
    except FileNotFoundError:
        logs_folder = os.path.join(module_path, "logs")
        filename = os.path.join(logs_folder, "monitoring.log")
        os.mkdir(logs_folder)
        with open(filename, "w") as f:
            f.write("")

        logging.basicConfig(filename=filename, level=logging.DEBUG)

    # parser config
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file",
                        help="Path to the Joblib model",
                        type=str, default=None)
    parser.add_argument("--model_name",
                        help="Model name that will be used to save the model",
                        type=str,
                        default="sentiment_analysis_model")
    parser.add_argument("--estimator",
                        help="Name of the estimator to use for the modeling pipeline",
                        type=str,
                        default=None)
    parser.add_argument("--data_path", help="data", type=str, default=os.path.join(module_path, "data", "comments.csv"))
    parser.add_argument("--tags",
                        help="tags to be passed to the Neptune AI, separated by commas",
                        type=str,
                        default=None)
    parser.add_argument("--grid_search",
                        help="should run a grid search ?",
                        type=bool,
                        default=False)
    parser.add_argument("--cv",
                        help="should we cross-validate ?",
                        type=bool,
                        default=False)
    args = parser.parse_args()
    model_file = args.model_file
    model_name = args.model_name
    tags = args.tags
    data_path = args.data_path
    grid_search = args.grid_search
    estimator = args.estimator
    cv = args.cv

    estimators = {
        "LGBM": {"name": LGBMClassifier,
                 "hyperparams": {
                     "clf__max_depth": [3, 10, -1],
                     "clf__n_estimators": [50, 100, 200],
                     "clf__class_weight": ['balanced'],
                     "clf__random_state": [43]}},
        "SVC": {"name": SVC,
                "hyperparams": {'clf__C': [5, 10, 100],
                                'clf__class_weight': ['balanced', {0: 0.37, 1: 0.63}],
                                'clf__kernel': ['poly', 'rbf', 'sigmoid'],
                                'clf__gamma': [0.001, "scale", "auto"]}}
    }

    if tags:
        tags = args.tags.split(",")

    # monitoring config
    neptune.init(project_qualified_name='nohossat/youtube-sentiment-analysis')

    try:
        df = pd.read_csv(data_path)
        X, y = split_data(df)
    except FileNotFoundError:
        print("Data path not correct")
        raise

    hyper_params = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    # get model
    if model_file:
        model = get_model(model_file=model_file)
        hyper_params = model.steps[1][1].get_params()
    else:
        if estimator is None and estimator not in estimators.keys():
            estimator = "SVC"

        model = estimators[estimator]["name"]
        model = create_pipeline(model_estimator=model)

    # run grid search or not ?
    if grid_search:
        model_name = f"grid_search_{estimator}"
        hyper_params = estimators[estimator]["hyperparams"]
        create_exp(hyper_params, tags)
        logging.info(hyper_params)
        model, best_params = run_grid_search(model=model, params=hyper_params, data=(X_train, y_train))

        # record best params
        for param, value in best_params.items():
            neptune.log_text(f'best_{param}', str(value))
    else:
        create_exp(hyper_params, tags)

        if model_file is None:
            model = get_model(model_estimator=estimators[estimator]['name'], data=(X_train, y_train))

    # compute metrics and logs data, model and metrics to Neptune.ai
    metrics = record_metadata(X_test, y_test, model, cv=cv)

    if not model_file:
        model_file = os.path.join(module_path, "models", f"{model_name}.joblib")

    joblib.dump(model, model_file)
    save_artifact(data_path=data_path, model_file=model_file)
