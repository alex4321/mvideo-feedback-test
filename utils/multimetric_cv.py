from joblib import Parallel, delayed
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from .imbalanced_class_sample_weights import class_sample_weights


def _get_scoring(name):
    scorings = {
        "f1": lambda y_true, y_pred: f1_score(y_true, y_pred),
        "f1_micro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro"),
        "f1_macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
        "mae": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
        "mse": lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
        "mae_imb": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred,
                                                              sample_weight=class_sample_weights(y_true)),
        "mse_imb": lambda y_true, y_pred: mean_squared_error(y_true, y_pred,
                                                             sample_weight=class_sample_weights(y_true)),
    }
    assert name in scorings, "Invalid scoring name"
    return scorings[name]


def _cv_score(estimator, scorings, default_params, X_train, y_train, X_test, y_test):
    estimator.set_params(**default_params)
    estimator.fit(X_train, y_train)
    y_predicted = estimator.predict(X_test)
    scores = {}
    for scoring in scorings:
        scores[scoring] = _get_scoring(scoring)(y_test, y_predicted)
    return scores


def multimetric_cv(estimator, X, y, metrics, n_fold=3, n_jobs=2):
    kfold = StratifiedKFold(n_fold)
    params = estimator.get_params(True)
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(
            delayed(_cv_score)(
                estimator,
                metrics,
                params,
                X[train_index], y[train_index],
                X[test_index], y[test_index]
            )
            for train_index, test_index in kfold.split(X, y)
        )
    all_results = {}
    for metric in metrics:
        values = np.array([item.get(metric) for item in results])
        all_results[metric] = values
    return all_results
