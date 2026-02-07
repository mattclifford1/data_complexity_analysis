"""
High-level orchestration API for ML model evaluation.

Provides convenient functions for evaluating models with various
metrics and evaluation strategies.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

from .models import AbstractMLModel, get_default_models
from .classification_metrics import AbstractEvaluationMetric, get_default_metrics
from .evaluation import AbstractEvaluator, get_default_evaluator


def evaluate_single_model(
    model: AbstractMLModel,
    data: Dict,
    evaluator: Optional[AbstractEvaluator] = None,
    metrics: Optional[List[AbstractEvaluationMetric]] = None,
    cv_folds: int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a single model on a dataset.

    Parameters
    ----------
    model : AbstractMLModel
        Model to evaluate.
    data : dict
        Data dictionary with 'X' and 'y' keys.
    evaluator : AbstractEvaluator, optional
        Evaluation strategy. Default: CrossValidationEvaluator(cv_folds)
    metrics : list of AbstractEvaluationMetric, optional
        Metrics to compute. Default: get_default_metrics()
    cv_folds : int, optional
        Number of CV folds (used if evaluator is None). Default: 5

    Returns
    -------
    dict
        Metric name -> {'mean': float, 'std': float}
    """
    if evaluator is None:
        evaluator = get_default_evaluator(cv_folds=cv_folds)
    if metrics is None:
        metrics = get_default_metrics()

    X, y = data["X"], data["y"]
    return evaluator.evaluate(model, X, y, metrics=metrics)


def evaluate_models(
    data: Dict,
    models: Optional[List[AbstractMLModel]] = None,
    evaluator: Optional[AbstractEvaluator] = None,
    metrics: Optional[List[AbstractEvaluationMetric]] = None,
    cv_folds: int = 5,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate multiple models on a dataset.

    Parameters
    ----------
    data : dict
        Data dictionary with 'X' and 'y' keys.
    models : list of AbstractMLModel, optional
        Models to evaluate. Default: get_default_models()
    evaluator : AbstractEvaluator, optional
        Evaluation strategy. Default: CrossValidationEvaluator(cv_folds)
    metrics : list of AbstractEvaluationMetric, optional
        Metrics to compute. Default: get_default_metrics()
    cv_folds : int, optional
        Number of CV folds (used if evaluator is None). Default: 5

    Returns
    -------
    dict
        Model name -> metric results dict
        Example: {'LogisticRegression': {'accuracy': {'mean': 0.9, 'std': 0.02}}}
    """
    if models is None:
        models = get_default_models()
    if evaluator is None:
        evaluator = get_default_evaluator(cv_folds=cv_folds)
    if metrics is None:
        metrics = get_default_metrics()

    X, y = data["X"], data["y"]

    results = {}
    for model in models:
        model_results = evaluator.evaluate(model, X, y, metrics=metrics)
        results[model.name] = model_results

    return results


def evaluate_models_train_test(
    train_data: Dict,
    test_data: Dict,
    models: Optional[List[AbstractMLModel]] = None,
    metrics: Optional[List[AbstractEvaluationMetric]] = None,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Train each model on train_data, evaluate on both train and test.

    Parameters
    ----------
    train_data : dict
        Training data with 'X' and 'y' keys.
    test_data : dict
        Test data with 'X' and 'y' keys.
    models : list of AbstractMLModel, optional
        Models to evaluate. Default: get_default_models()
    metrics : list of AbstractEvaluationMetric, optional
        Metrics to compute. Default: get_default_metrics()

    Returns
    -------
    tuple of (train_results, test_results)
        Each: model_name -> metric_name -> {'mean': float, 'std': 0.0}
    """
    if models is None:
        models = get_default_models()
    if metrics is None:
        metrics = get_default_metrics()

    X_train, y_train = train_data["X"], train_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]

    train_results = {}
    test_results = {}

    for model in models:
        sklearn_model = model._create_model()
        sklearn_model.fit(X_train, y_train)

        y_pred_train = sklearn_model.predict(X_train)
        y_pred_test = sklearn_model.predict(X_test)

        train_metrics = {}
        test_metrics = {}
        for metric in metrics:
            train_metrics[metric.name] = {
                "mean": metric.compute(y_train, y_pred_train),
                "std": 0.0,
            }
            test_metrics[metric.name] = {
                "mean": metric.compute(y_test, y_pred_test),
                "std": 0.0,
            }

        train_results[model.name] = train_metrics
        test_results[model.name] = test_metrics

    return train_results, test_results


def get_best_metric(results: Dict, metric: str = "accuracy") -> float:
    """
    Get the best value of a metric across all models.

    Parameters
    ----------
    results : dict
        Output from evaluate_models().
    metric : str, optional
        Metric name. Default: 'accuracy'

    Returns
    -------
    float
        Best metric value across all models.
    """
    values = [
        r[metric]["mean"]
        for r in results.values()
        if metric in r and not np.isnan(r[metric]["mean"])
    ]
    return max(values) if values else np.nan


def get_mean_metric(results: Dict, metric: str = "accuracy") -> float:
    """
    Get the mean value of a metric across all models.

    Parameters
    ----------
    results : dict
        Output from evaluate_models().
    metric : str, optional
        Metric name. Default: 'accuracy'

    Returns
    -------
    float
        Mean metric value across all models.
    """
    values = [
        r[metric]["mean"]
        for r in results.values()
        if metric in r and not np.isnan(r[metric]["mean"])
    ]
    return np.mean(values) if values else np.nan


def get_model_metric(
    results: Dict, model_name: str, metric: str = "accuracy"
) -> float:
    """
    Get a specific metric value for a specific model.

    Parameters
    ----------
    results : dict
        Output from evaluate_models().
    model_name : str
        Name of the model.
    metric : str, optional
        Metric name. Default: 'accuracy'

    Returns
    -------
    float
        Metric value for the specified model, or NaN if not found.
    """
    if model_name not in results:
        return np.nan
    if metric not in results[model_name]:
        return np.nan
    return results[model_name][metric]["mean"]


def print_evaluation_results(results: Dict, metric: str = "accuracy"):
    """
    Print a formatted table of evaluation results.

    Parameters
    ----------
    results : dict
        Output from evaluate_models().
    metric : str, optional
        Metric to display. Default: 'accuracy'
    """
    print(f"\n{'Model':<20} {metric:>12} {'± std':>10}")
    print("-" * 44)

    # Sort by metric value (descending)
    sorted_models = sorted(
        results.items(),
        key=lambda x: (
            -x[1][metric]["mean"]
            if metric in x[1] and not np.isnan(x[1][metric]["mean"])
            else float("-inf")
        ),
    )

    for model_name, metrics_dict in sorted_models:
        if metric not in metrics_dict:
            continue
        value = metrics_dict[metric]["mean"]
        std = metrics_dict[metric]["std"]
        print(f"{model_name:<20} {value:>12.4f} {'±':>3} {std:.4f}")
