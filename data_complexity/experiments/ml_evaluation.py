"""
Utilities for evaluating ML models on datasets.

Provides functions to train classifiers and compute performance metrics
for correlation with data complexity measures.
"""
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# Scoring metrics to evaluate
SCORING_METRICS = {
    "accuracy": "accuracy",
    "f1": "f1_weighted",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "balanced_accuracy": "balanced_accuracy",
}


def get_default_classifiers():
    """
    Return a dict of classifiers for benchmarking.

    Returns
    -------
    dict
        Classifier name -> sklearn estimator (with scaling pipeline where needed).
    """
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "KNN-5": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)
        ),
        "KNN-3": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=3)
        ),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "SVM-RBF": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", random_state=42)
        ),
        "SVM-Linear": make_pipeline(
            StandardScaler(), SVC(kernel="linear", random_state=42)
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42
        ),
        "NaiveBayes": GaussianNB(),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
        ),
    }


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
    if classifiers is None:
        classifiers = get_default_classifiers()
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


def get_best_metric(ml_results, metric="accuracy"):
    """Get the best value of a metric across all classifiers."""
    values = get_metric_summary(ml_results, metric)
    valid = [v for v in values.values() if not np.isnan(v)]
    return max(valid) if valid else np.nan


def get_mean_metric(ml_results, metric="accuracy"):
    """Get the mean value of a metric across all classifiers."""
    values = get_metric_summary(ml_results, metric)
    valid = [v for v in values.values() if not np.isnan(v)]
    return np.mean(valid) if valid else np.nan


def get_model_metric(ml_results, model_name, metric="accuracy"):
    """Get a specific model's metric value."""
    if model_name in ml_results and metric in ml_results[model_name]:
        return ml_results[model_name][metric]["mean"]
    return np.nan


def print_model_results(ml_results, metric="accuracy"):
    """Print a formatted table of model results for a metric."""
    print(f"\n{'Model':<20} {metric:>12} {'± std':>10}")
    print("-" * 44)

    # Sort by metric value
    summary = get_metric_summary(ml_results, metric)
    sorted_models = sorted(summary.items(), key=lambda x: -x[1] if not np.isnan(x[1]) else float('-inf'))

    for model, value in sorted_models:
        std = ml_results[model][metric]["std"]
        print(f"{model:<20} {value:>12.4f} {'±':>3} {std:.4f}")


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
