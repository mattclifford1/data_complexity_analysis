"""Tests for ML evaluation metrics and evaluators."""
import pytest
import numpy as np
from sklearn.datasets import make_classification

from data_complexity.model_experiments.classification import (
    # Metrics
    AbstractEvaluationMetric,
    AccuracyMetric,
    AccuracyMinorityMetric,
    F1Metric,
    PrecisionMetric,
    RecallMetric,
    get_default_metrics,
    get_metrics_dict,
    get_metric_by_name,
    get_metrics_from_names,
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
        """Test get_metrics_dict returns appropriate scorer types."""
        metrics_dict = get_metrics_dict()
        # All values should be strings or callables
        for name, scorer in metrics_dict.items():
            assert isinstance(name, str), f"Expected metric name to be str, got {type(name)}"
            assert callable(scorer)

    def test_get_metrics_dict_custom_metric(self):
        """Test custom metrics return callables."""
        from sklearn.metrics import get_scorer_names

        # Mix built-in and custom metrics
        metrics = [
            AccuracyMetric(),  # Built-in
            AccuracyMinorityMetric(),  # Custom
        ]

        metrics_dict = get_metrics_dict(metrics)

        assert callable(metrics_dict["minority_accuracy"])

    def test_get_metrics_dict_works_with_cross_validate(self, simple_xy):
        """Test that returned dict works with sklearn cross_validate."""
        from sklearn.model_selection import cross_validate
        from sklearn.linear_model import LogisticRegression

        X, y = simple_xy
        metrics = [AccuracyMetric(), AccuracyMinorityMetric()]
        scoring = get_metrics_dict(metrics)

        model = LogisticRegression(max_iter=1000)
        results = cross_validate(model, X, y, cv=3, scoring=scoring)

        # Check both metrics produced results
        assert "test_accuracy" in results
        assert "test_minority_accuracy" in results
        assert len(results["test_accuracy"]) == 3
        assert len(results["test_minority_accuracy"]) == 3


class TestMetricFactory:
    """Tests for metric factory functions."""

    def test_get_metric_by_name_accuracy(self):
        metric = get_metric_by_name('accuracy')
        assert isinstance(metric, AccuracyMetric)
        assert metric.name == 'accuracy'

    def test_get_metric_by_name_f1(self):
        metric = get_metric_by_name('f1')
        assert isinstance(metric, F1Metric)
        assert metric.name == 'f1'

    def test_get_metric_by_name_precision(self):
        metric = get_metric_by_name('precision')
        assert isinstance(metric, PrecisionMetric)
        assert metric.name == 'precision'

    def test_get_metric_by_name_recall(self):
        metric = get_metric_by_name('recall')
        assert isinstance(metric, RecallMetric)
        assert metric.name == 'recall'

    def test_get_metric_by_name_invalid(self):
        with pytest.raises(ValueError, match="Unknown metric name"):
            get_metric_by_name('invalid_metric')

    def test_get_metric_by_name_error_message(self):
        try:
            get_metric_by_name('foo')
        except ValueError as e:
            assert 'foo' in str(e)
            assert 'Available metrics:' in str(e)
            assert 'accuracy' in str(e)

    def test_get_metrics_from_names(self):
        metrics = get_metrics_from_names(['accuracy', 'f1', 'precision'])
        assert len(metrics) == 3
        assert isinstance(metrics[0], AccuracyMetric)
        assert isinstance(metrics[1], F1Metric)
        assert isinstance(metrics[2], PrecisionMetric)

    def test_get_metrics_from_names_empty(self):
        metrics = get_metrics_from_names([])
        assert metrics == []

    def test_get_metrics_from_names_all_requested(self):
        """Test that all metrics in a typical config can be created."""
        names = ['accuracy', 'f1', 'precision', 'recall', 'balanced_accuracy']
        metrics = get_metrics_from_names(names)
        assert len(metrics) == 5
        metric_names = [m.name for m in metrics]
        assert metric_names == names


