"""
Evaluation metric classes for ML model assessment.

Provides abstract base classes and concrete implementations for:
- Evaluation metrics (accuracy, F1, precision, recall, balanced accuracy)
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    balanced_accuracy_score,
    precision_recall_fscore_support
)
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import get_scorer_names, make_scorer


# ============================================================================
# Metrics Abstract
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

    def __call__(self, y_true, y_pred) -> float:
        """Allow metric to be called directly."""
        return self.compute(y_true, y_pred)
    
    @property
    def greater_is_better(self) -> bool:
        """Indicate whether higher metric values are better."""
        return True

    def get_scorer(self):
        """
        Return an sklearn-compatible scorer for this metric.

        Returns either a string (for built-in sklearn metrics) or a callable
        scorer (for custom metrics). Custom metrics are wrapped with make_scorer
        to convert from (y_true, y_pred) signature to (estimator, X, y) signature.

        Returns
        -------
        str or callable
            If sklearn_name is a built-in scorer, returns the string name.
            Otherwise, returns a make_scorer wrapped callable.
        """
        # Check if this is a built-in sklearn metric
        # if self.sklearn_name in get_scorer_names():
        #     return self.sklearn_name

        # Custom metric: wrap with make_scorer
        # greater_is_better=True for all our metrics (accuracy-like)
        return make_scorer(self.compute, greater_is_better=self.greater_is_better)

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


# ============================================================================
# Classification Metrics
# ============================================================================

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
    

class AccuracyBalancedMetric(AbstractEvaluationMetric):
    """Balanced accuracy metric."""

    @property
    def name(self) -> str:
        return "balanced_accuracy"

    @property
    def sklearn_name(self) -> str:
        return "balanced_accuracy"

    def compute(self, y_true, y_pred) -> float:
        return balanced_accuracy_score(y_true, y_pred)


class AccuracyMinorityMetric(AbstractEvaluationMetric):
    """Minority class accuracy metric."""

    @property
    def name(self) -> str:
        return "minority_accuracy"

    @property
    def sklearn_name(self) -> str:
        return "minority_accuracy"

    def compute(self, y_true, y_pred) -> float:
        return confusion_matrix(y_true, y_pred, normalize="true").diagonal()[1]
    
    
class AccuracyMajorityMetric(AbstractEvaluationMetric):
    """Majority class accuracy metric."""

    @property
    def name(self) -> str:
        return "majority_accuracy"

    @property
    def sklearn_name(self) -> str:
        return "majority_accuracy"

    def compute(self, y_true, y_pred) -> float:
        return confusion_matrix(y_true, y_pred, normalize="true").diagonal()[0]


class GeometricMeanMetric(AbstractEvaluationMetric):
    """Geometric mean score metric."""

    @property
    def name(self) -> str:
        return "geometric_mean"

    @property
    def sklearn_name(self) -> str:
        return "geometric_mean"

    def compute(self, y_true, y_pred) -> float:
        return geometric_mean_score(y_true, y_pred)


class GeometricMeanWeightedMetric(AbstractEvaluationMetric):
    """Geometric mean score metric."""

    @property
    def name(self) -> str:
        return "geometric_mean_weighted"

    @property
    def sklearn_name(self) -> str:
        return "geometric_mean_weighted"

    def compute(self, y_true, y_pred) -> float:
        return geometric_mean_score(y_true, y_pred, average="weighted")


class F1Metric(AbstractEvaluationMetric):
    """Weighted F1 score metric."""

    @property
    def name(self) -> str:
        return "f1"

    @property
    def sklearn_name(self) -> str:
        return "f1"

    def compute(self, y_true, y_pred) -> float:
        return f1_score(y_true, y_pred, zero_division=0)


class F1WeightedMetric(AbstractEvaluationMetric):
    """Weighted F1 score metric."""

    @property
    def name(self) -> str:
        return "f1_weighted"

    @property
    def sklearn_name(self) -> str:
        return "f1_weighted"

    def compute(self, y_true, y_pred) -> float:
        return f1_score(y_true, y_pred, average="weighted", zero_division=0)


class PrecisionMetric(AbstractEvaluationMetric):
    """Precision metric. Pos is taken as 1 by default."""

    @property
    def name(self) -> str:
        return "precision"

    @property
    def sklearn_name(self) -> str:
        return "precision"

    def compute(self, y_true, y_pred) -> float:
        prec, recal, fscor, sup = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
        # Return class 1 precision (positive class) as scalar
        return float(prec[1])

    
class Precision0Metric(AbstractEvaluationMetric):
    """Precision for class 0 metric."""

    @property
    def name(self) -> str:
        return "precision_class_0"

    @property
    def sklearn_name(self) -> str:
        return "precision_class_0"

    def compute(self, y_true, y_pred) -> float:
        return precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    

class Precision1Metric(AbstractEvaluationMetric):
    """Precision for class 1 metric."""

    @property
    def name(self) -> str:
        return "precision_class_1"

    @property
    def sklearn_name(self) -> str:
        return "precision_class_1"

    def compute(self, y_true, y_pred) -> float:
        return precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    

class PrecisionWeightedMetric(AbstractEvaluationMetric):
    """Weighted precision metric."""

    @property
    def name(self) -> str:
        return "precision_weighted"

    @property
    def sklearn_name(self) -> str:
        return "precision_weighted"

    def compute(self, y_true, y_pred) -> float:
        return precision_score(y_true, y_pred, average="weighted", zero_division=0)
    

class FScoreMetric(AbstractEvaluationMetric):
    """F-score metric. Pos is taken as 1 by default."""

    @property
    def name(self) -> str:
        return "fscore"

    @property
    def sklearn_name(self) -> str:
        return "fscore"

    def compute(self, y_true, y_pred) -> float:
        prec, recal, fscor, sup = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
        # Return class 1 F-score (positive class) as scalar
        return float(fscor[1])
    

class RecallMetric(AbstractEvaluationMetric):
    """Recall metric. Pos is taken as 1 by default."""

    @property
    def name(self) -> str:
        return "recall"

    @property
    def sklearn_name(self) -> str:
        return "recall"

    def compute(self, y_true, y_pred) -> float:
        prec, recal, fscor, sup = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
        # Return class 1 recall (positive class) as scalar
        return float(recal[1])


class RecallWeightedMetric(AbstractEvaluationMetric):
    """Weighted recall metric."""

    @property
    def name(self) -> str:
        return "recall_weighted"

    @property
    def sklearn_name(self) -> str:
        return "recall_weighted"

    def compute(self, y_true, y_pred) -> float:
        return recall_score(y_true, y_pred, average="weighted", zero_division=0)
    

class AucMetric(AbstractEvaluationMetric):
    """AUC metric."""

    @property
    def name(self) -> str:
        return "auc"

    @property
    def sklearn_name(self) -> str:
        return "auc"

    def compute(self, y_true, y_pred) -> float:
        return auc(y_true, y_pred)
    

class RocAucMetric(AbstractEvaluationMetric):
    """ROC AUC metric."""

    @property
    def name(self) -> str:
        return "roc_auc"

    @property
    def sklearn_name(self) -> str:
        return "roc_auc"

    def compute(self, y_true, y_pred) -> float:
        return roc_auc_score(y_true, y_pred)


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
        AccuracyBalancedMetric(),
        F1Metric(),
        # PrecisionMetric(),
        # RecallMetric(),
    ]


def get_metrics_dict(
    metrics: Optional[List[AbstractEvaluationMetric]] = None
) -> Dict[str, Any]:
    """
    Convert metrics list to sklearn scoring dict.

    Returns a dictionary suitable for sklearn's cross_validate scoring parameter.
    For custom metrics, returns make_scorer wrapped callables.

    Parameters
    ----------
    metrics : list of AbstractEvaluationMetric, optional
        Metrics to convert. Default: get_default_metrics()

    Returns
    -------
    dict
        Metric name -> scorer (str or callable) mapping.

    Examples
    --------
    >>> custom_metrics = [AccuracyMetric(), AccuracyMinorityMetric()]
    >>> scoring = get_metrics_dict(custom_metrics)
    >>> scoring  # doctest: +SKIP
    {'accuracy': 'accuracy', 'minority_accuracy': <callable>}
    """
    if metrics is None:
        metrics = get_default_metrics()
    return {metric.name: metric.get_scorer() for metric in metrics}


def get_metric_by_name(name: str) -> AbstractEvaluationMetric:
    """
    Get a metric instance by its string name.

    Parameters
    ----------
    name : str
        Metric name (e.g., 'accuracy', 'f1', 'precision', 'recall')

    Returns
    -------
    AbstractEvaluationMetric
        Metric instance

    Raises
    ------
    ValueError
        If metric name is not recognized

    Examples
    --------
    >>> metric = get_metric_by_name('accuracy')
    >>> metric.name
    'accuracy'

    >>> metrics = [get_metric_by_name(name) for name in ['accuracy', 'f1']]
    """
    metric_map = {
        'accuracy': AccuracyMetric,
        'balanced_accuracy': AccuracyBalancedMetric,
        'minority_accuracy': AccuracyMinorityMetric,
        'majority_accuracy': AccuracyMajorityMetric,
        'geometric_mean': GeometricMeanMetric,
        'geometric_mean_weighted': GeometricMeanWeightedMetric,
        'f1': F1Metric,
        'f1_weighted': F1WeightedMetric,
        'precision': PrecisionMetric,
        'precision_class_0': Precision0Metric,
        'precision_class_1': Precision1Metric,
        'precision_weighted': PrecisionWeightedMetric,
        'fscore': FScoreMetric,
        'recall': RecallMetric,
        'recall_weighted': RecallWeightedMetric,
        'auc': AucMetric,
        'roc_auc': RocAucMetric,
    }

    if name not in metric_map:
        available = ', '.join(sorted(metric_map.keys()))
        raise ValueError(
            f"Unknown metric name: '{name}'. "
            f"Available metrics: {available}"
        )

    return metric_map[name]()


def get_metrics_from_names(names: List[str]) -> List[AbstractEvaluationMetric]:
    """
    Convert a list of metric names to metric instances.

    Parameters
    ----------
    names : list of str
        Metric names

    Returns
    -------
    list of AbstractEvaluationMetric
        Metric instances

    Examples
    --------
    >>> metrics = get_metrics_from_names(['accuracy', 'f1', 'precision'])
    >>> [m.name for m in metrics]
    ['accuracy', 'f1', 'precision']
    """
    return [get_metric_by_name(name) for name in names]

