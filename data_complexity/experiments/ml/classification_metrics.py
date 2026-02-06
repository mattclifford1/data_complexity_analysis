"""
Evaluation metrics and evaluator classes for ML model assessment.

Provides abstract base classes and concrete implementations for:
- Evaluation metrics (accuracy, F1, precision, recall, balanced accuracy)
- Evaluators (cross-validation, train-test split)
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)

# ============================================================================
# Evaluation Metrics
# ============================================================================


class AbstractEvaluationMetric(ABC):
    """
    Abstract base class for evaluation metrics.

    Evaluation metrics define how to score model predictions.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name (e.g., 'accuracy')."""
        pass

    @property
    @abstractmethod
    def sklearn_name(self) -> str:
        """Return the sklearn scoring name (e.g., 'accuracy')."""
        pass

    @abstractmethod
    def compute(self, y_true, y_pred) -> float:
        """
        Compute the metric value.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.

        Returns
        -------
        float
            Metric value.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class AccuracyMetric(AbstractEvaluationMetric):
    """Classification accuracy metric."""

    @property
    def name(self) -> str:
        return "accuracy"

    @property
    def sklearn_name(self) -> str:
        return "accuracy"

    def compute(self, y_true, y_pred) -> float:
        return accuracy_score(y_true, y_pred)


class F1Metric(AbstractEvaluationMetric):
    """Weighted F1 score metric."""

    @property
    def name(self) -> str:
        return "f1"

    @property
    def sklearn_name(self) -> str:
        return "f1_weighted"

    def compute(self, y_true, y_pred) -> float:
        return f1_score(y_true, y_pred, average="weighted")


class PrecisionMetric(AbstractEvaluationMetric):
    """Weighted precision metric."""

    @property
    def name(self) -> str:
        return "precision"

    @property
    def sklearn_name(self) -> str:
        return "precision_weighted"

    def compute(self, y_true, y_pred) -> float:
        return precision_score(y_true, y_pred, average="weighted", zero_division=0)


class RecallMetric(AbstractEvaluationMetric):
    """Weighted recall metric."""

    @property
    def name(self) -> str:
        return "recall"

    @property
    def sklearn_name(self) -> str:
        return "recall_weighted"

    def compute(self, y_true, y_pred) -> float:
        return recall_score(y_true, y_pred, average="weighted", zero_division=0)


class BalancedAccuracyMetric(AbstractEvaluationMetric):
    """Balanced accuracy metric."""

    @property
    def name(self) -> str:
        return "balanced_accuracy"

    @property
    def sklearn_name(self) -> str:
        return "balanced_accuracy"

    def compute(self, y_true, y_pred) -> float:
        return balanced_accuracy_score(y_true, y_pred)


# ============================================================================
# Metric Factory Functions
# ============================================================================


def get_default_metrics() -> List[AbstractEvaluationMetric]:
    """
    Return a list of default evaluation metrics.

    Returns
    -------
    list of AbstractEvaluationMetric
        Default set of metrics: accuracy, F1, precision, recall, balanced accuracy.
    """
    return [
        AccuracyMetric(),
        F1Metric(),
        PrecisionMetric(),
        RecallMetric(),
        BalancedAccuracyMetric(),
    ]


def get_metrics_dict(metrics: Optional[List[AbstractEvaluationMetric]] = None) -> Dict[str, str]:
    """
    Convert metrics list to sklearn scoring dict.

    Parameters
    ----------
    metrics : list of AbstractEvaluationMetric, optional
        Metrics to convert. Default: get_default_metrics()

    Returns
    -------
    dict
        Metric name -> sklearn scoring name mapping.
    """
    if metrics is None:
        metrics = get_default_metrics()
    return {metric.name: metric.sklearn_name for metric in metrics}

