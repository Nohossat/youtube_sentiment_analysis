from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import train_test_split


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
    final_scores = {metric: round(np.mean(metric_scores), 3) for metric, metric_scores in scores.items()}
    return final_scores


def compute_metrics(X, y, model, random_state=0):
    """
    Compute main classification metrics
    :param X: features. list
    :param y: label. list
    :param model : Scikit-Learn Classification Estimator
    :return: a dictionary with classification metrics
    """

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    scores = classification_report(model.predict(X_test), y_test, output_dict=True)
    final_scores = scores['weighted avg']
    final_scores['accuracy'] = scores['accuracy']
    return final_scores
