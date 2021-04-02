from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import os

import nohossat_cas_pratique
from nohossat_cas_pratique.logging_app import start_logging

# config logging
module_path = os.path.dirname(os.path.dirname(os.path.dirname(nohossat_cas_pratique.__file__)))
start_logging(module_path)

def get_grid_search_best_metrics(model, metrics):
    scores = {}

    for metric in metrics:
        means_test = model.cv_results_[f'mean_test_{metric}']
        stds_test = model.cv_results_[f'std_test_{metric}']

        for mean, std, params in zip(means_test, stds_test, model.cv_results_['params']):
            if params == model.best_params_:
                scores[f'mean_test_{metric}'] = mean
                scores[f'std_test_{metric}'] = std

    return scores


def compute_metrics_cv(X, y, model):
    """
    Compute main classification metrics with a previous 5-fold cross-validation step
    :param X: features. list
    :param y: label. list
    :param model : Scikit-Learn Classification Estimator
    :return: a dictionary with classification metrics
    """
    scores = cross_validate(model,
                            X,
                            y,
                            cv=5,
                            scoring=('accuracy', 'precision', 'recall', 'f1_weighted', 'roc_auc'),
                            return_train_score = True)
    final_scores = {f"mean_{metric}": round(np.mean(metric_scores), 3) for metric, metric_scores in scores.items()}
    std_final_scores = {f"std_{metric}": round(np.std(metric_scores), 3) for metric, metric_scores in scores.items()}
    final_scores = {**final_scores, **std_final_scores}
    return final_scores


def compute_metrics(X, y, model, dataset="test"):
    """
    Compute main classification metrics
    :param X: features. list
    :param y: label. list
    :param model : Scikit-Learn Classification Estimator
    :return: a dictionary with classification metrics
    """

    scores = classification_report(model.predict(X), y, output_dict=True)
    final_scores = scores['weighted avg']
    final_scores['accuracy'] = scores['accuracy']
    final_scores = {f"{dataset}/{metric}": value for metric, value in final_scores.items() if metric != "support"}

    # compute auc
    try:
        probas = model.predict_proba(X)
        auc = roc_auc_score(y, probas[:, 1])
        final_scores['auc'] = auc
    except AttributeError as e:
        logging.error(e)
        raise AttributeError("The probability param must be set before fitting model")

    return final_scores
