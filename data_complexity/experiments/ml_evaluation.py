"""
Utilities for evaluating ML models on datasets.

Provides functions to train classifiers and compute performance metrics
for correlation with data complexity measures.

This module provides a functional interface that wraps the class-based
ml_models module for backwards compatibility.
"""
import numpy as np
from data_complexity.experiments.ml_models import (
    AbstractMLModel,
    LogisticRegressionModel,
    KNNModel,
    DecisionTreeModel,
    SVMModel,
    RandomForestModel,
    GradientBoostingModel,
    NaiveBayesModel,
    MLPModel,
    SCORING_METRICS,
    get_default_models,
    evaluate_models,
    get_best_metric,
    get_mean_metric,
    print_evaluation_results,
)


def get_default_classifiers():
    """
    Return a dict of classifiers for benchmarking.

    Returns
    -------
    dict
        Classifier name -> sklearn estimator (with scaling pipeline where needed).
    """
    models = get_default_models()
    return {model.name: model._create_model() for model in models}


def evaluate_classifiers(
    X, y, classifiers=None, cv_folds=5, random_state=42, scoring=None
):
    """
    Evaluate multiple classifiers using cross-validation with multiple metrics.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Target labels.
    classifiers : dict, optional
        Classifier name -> estimator. Default: get_default_classifiers()
    cv_folds : int
        Number of cross-validation folds. Default: 5
    random_state : int
        Random state for reproducibility. Default: 42
    scoring : dict, optional
        Scoring metrics. Default: SCORING_METRICS

    Returns
    -------
    dict
        Classifier name -> dict with metric names -> {'mean': float, 'std': float}
    """
    data = {"X": X, "y": y}

    if classifiers is not None:
        # Legacy path: use provided sklearn estimators directly
        from sklearn.model_selection import cross_validate, StratifiedKFold

        if scoring is None:
            scoring = SCORING_METRICS

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        results = {}

        for name, clf in classifiers.items():
            try:
                cv_results = cross_validate(
                    clf, X, y, cv=cv, scoring=scoring, return_train_score=False
                )
                model_results = {}
                for metric_name in scoring.keys():
                    scores = cv_results[f"test_{metric_name}"]
                    model_results[metric_name] = {
                        "mean": scores.mean(),
                        "std": scores.std(),
                    }
                results[name] = model_results
            except Exception as e:
                print(f"Warning: {name} failed: {e}")
                results[name] = {
                    m: {"mean": np.nan, "std": np.nan} for m in scoring.keys()
                }

        return results

    # New path: use model classes
    models = get_default_models()
    for model in models:
        model.random_state = random_state

    return evaluate_models(data, models=models, cv_folds=cv_folds)


def get_metric_summary(ml_results, metric="accuracy"):
    """
    Extract a specific metric across all models.

    Parameters
    ----------
    ml_results : dict
        Output from evaluate_classifiers().
    metric : str
        Metric name (accuracy, f1, precision, recall, balanced_accuracy).

    Returns
    -------
    dict
        Model name -> metric mean value.
    """
    return {
        model: results[metric]["mean"]
        for model, results in ml_results.items()
        if metric in results
    }


def get_model_metric(ml_results, model_name, metric="accuracy"):
    """Get a specific model's metric value."""
    if model_name in ml_results and metric in ml_results[model_name]:
        return ml_results[model_name][metric]["mean"]
    return np.nan


def print_model_results(ml_results, metric="accuracy"):
    """Print a formatted table of model results for a metric."""
    print_evaluation_results(ml_results, metric)


def print_all_metrics(ml_results):
    """Print results for all metrics."""
    for metric in SCORING_METRICS.keys():
        print_model_results(ml_results, metric)
        print()


# Legacy compatibility functions
def get_best_accuracy(ml_results):
    """Get the best accuracy across all classifiers."""
    return get_best_metric(ml_results, "accuracy")


def get_mean_accuracy(ml_results):
    """Get the mean accuracy across all classifiers."""
    return get_mean_metric(ml_results, "accuracy")


def get_linear_accuracy(ml_results):
    """Get logistic regression accuracy (linear separability indicator)."""
    return get_model_metric(ml_results, "LogisticRegression", "accuracy")
