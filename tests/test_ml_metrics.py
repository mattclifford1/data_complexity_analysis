"""Tests for ML evaluation metrics and evaluators."""
import pytest
import numpy as np
from sklearn.datasets import make_classification

from data_complexity.experiments.ml import (
    # Metrics
    AbstractEvaluationMetric,
    AccuracyMetric,
    F1Metric,
    PrecisionMetric,
    RecallMetric,
    get_default_metrics,
    get_metrics_dict,
    # Models
    LogisticRegressionModel,
    KNNModel,
)


@pytest.fixture
def simple_xy():
    """Generate simple classification data."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )
    return X, y


@pytest.fixture
def simple_model():
    """Return a simple model for testing."""
    return LogisticRegressionModel()


# ============================================================================
# Metric Tests
# ============================================================================

class TestAbstractEvaluationMetric:
    """Tests for AbstractEvaluationMetric."""

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            AbstractEvaluationMetric()

    def test_subclass_must_implement_name(self):
        class IncompleteMetric(AbstractEvaluationMetric):
            @property
            def sklearn_name(self):
                return "test"

            def compute(self, y_true, y_pred):
                return 0.0

        with pytest.raises(TypeError):
            IncompleteMetric()

    def test_subclass_must_implement_sklearn_name(self):
        class IncompleteMetric(AbstractEvaluationMetric):
            @property
            def name(self):
                return "test"

            def compute(self, y_true, y_pred):
                return 0.0

        with pytest.raises(TypeError):
            IncompleteMetric()

    def test_subclass_must_implement_compute(self):
        class IncompleteMetric(AbstractEvaluationMetric):
            @property
            def name(self):
                return "test"

            @property
            def sklearn_name(self):
                return "test"

        with pytest.raises(TypeError):
            IncompleteMetric()


class TestAccuracyMetric:
    """Tests for AccuracyMetric."""

    def test_name(self):
        metric = AccuracyMetric()
        assert metric.name == "accuracy"

    def test_sklearn_name(self):
        metric = AccuracyMetric()
        assert metric.sklearn_name == "accuracy"

    def test_compute(self):
        metric = AccuracyMetric()
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert metric.compute(y_true, y_pred) == 1.0

        y_pred = np.array([0, 0, 0, 0])
        assert metric.compute(y_true, y_pred) == 0.5

    def test_repr(self):
        metric = AccuracyMetric()
        assert "AccuracyMetric" in repr(metric)
        assert "accuracy" in repr(metric)


class TestF1Metric:
    """Tests for F1Metric."""

    def test_name(self):
        metric = F1Metric()
        assert metric.name == "f1"

    def test_sklearn_name(self):
        metric = F1Metric()
        assert metric.sklearn_name == "f1"

    def test_compute(self):
        metric = F1Metric()
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert metric.compute(y_true, y_pred) == 1.0


class TestPrecisionMetric:
    """Tests for PrecisionMetric."""

    def test_name(self):
        metric = PrecisionMetric()
        assert metric.name == "precision"

    def test_sklearn_name(self):
        metric = PrecisionMetric()
        assert metric.sklearn_name == "precision"


class TestRecallMetric:
    """Tests for RecallMetric."""

    def test_name(self):
        metric = RecallMetric()
        assert metric.name == "recall"

    def test_sklearn_name(self):
        metric = RecallMetric()
        assert metric.sklearn_name == "recall"


class TestMetricFactoryFunctions:
    """Tests for metric factory functions."""

    def test_get_default_metrics(self):
        metrics = get_default_metrics()
        assert all(isinstance(m, AbstractEvaluationMetric) for m in metrics)

    def test_get_default_metrics_return_scalar(self):
        metrics = get_default_metrics()
        y = np.array([0, 0, 1, 1])
        pred = np.array([0, 0, 0, 1])
        for metric in metrics:
            score = metric.compute(y, pred)
            assert isinstance(score, float), f"Expected np.float64, got {type(score)} from {metric.name}"

    def test_get_metrics_dict_default(self):
        metrics_dict = get_metrics_dict()
        assert metrics_dict["accuracy"] == "accuracy"
        assert metrics_dict["f1"] == "f1"
        assert metrics_dict["balanced_accuracy"] == "balanced_accuracy"


