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
    BalancedAccuracyMetric,
    get_default_metrics,
    get_metrics_dict,
    # Evaluators
    AbstractEvaluator,
    CrossValidationEvaluator,
    TrainTestSplitEvaluator,
    get_default_evaluator,
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
        assert metric.sklearn_name == "f1_weighted"

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
        assert metric.sklearn_name == "precision_weighted"


class TestRecallMetric:
    """Tests for RecallMetric."""

    def test_name(self):
        metric = RecallMetric()
        assert metric.name == "recall"

    def test_sklearn_name(self):
        metric = RecallMetric()
        assert metric.sklearn_name == "recall_weighted"


class TestBalancedAccuracyMetric:
    """Tests for BalancedAccuracyMetric."""

    def test_name(self):
        metric = BalancedAccuracyMetric()
        assert metric.name == "balanced_accuracy"

    def test_sklearn_name(self):
        metric = BalancedAccuracyMetric()
        assert metric.sklearn_name == "balanced_accuracy"


class TestMetricFactoryFunctions:
    """Tests for metric factory functions."""

    def test_get_default_metrics(self):
        metrics = get_default_metrics()
        assert len(metrics) == 5
        assert all(isinstance(m, AbstractEvaluationMetric) for m in metrics)

        names = [m.name for m in metrics]
        assert "accuracy" in names
        assert "f1" in names
        assert "precision" in names
        assert "recall" in names
        assert "balanced_accuracy" in names

    def test_get_metrics_dict_default(self):
        metrics_dict = get_metrics_dict()
        assert len(metrics_dict) == 5
        assert metrics_dict["accuracy"] == "accuracy"
        assert metrics_dict["f1"] == "f1_weighted"
        assert metrics_dict["precision"] == "precision_weighted"
        assert metrics_dict["recall"] == "recall_weighted"
        assert metrics_dict["balanced_accuracy"] == "balanced_accuracy"

    def test_get_metrics_dict_custom(self):
        metrics = [AccuracyMetric(), F1Metric()]
        metrics_dict = get_metrics_dict(metrics)
        assert len(metrics_dict) == 2
        assert "accuracy" in metrics_dict
        assert "f1" in metrics_dict


# ============================================================================
# Evaluator Tests
# ============================================================================


class TestAbstractEvaluator:
    """Tests for AbstractEvaluator."""

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            AbstractEvaluator()

    def test_subclass_must_implement_evaluate(self):
        class IncompleteEvaluator(AbstractEvaluator):
            pass

        with pytest.raises(TypeError):
            IncompleteEvaluator()


class TestCrossValidationEvaluator:
    """Tests for CrossValidationEvaluator."""

    def test_default_parameters(self):
        evaluator = CrossValidationEvaluator()
        assert evaluator.cv_folds == 5
        assert evaluator.random_state == 42

    def test_custom_parameters(self):
        evaluator = CrossValidationEvaluator(cv_folds=3, random_state=123)
        assert evaluator.cv_folds == 3
        assert evaluator.random_state == 123

    def test_evaluate(self, simple_xy, simple_model):
        X, y = simple_xy
        evaluator = CrossValidationEvaluator(cv_folds=3)
        results = evaluator.evaluate(simple_model, X, y)

        assert "accuracy" in results
        assert "f1" in results
        assert 0 <= results["accuracy"]["mean"] <= 1
        assert results["accuracy"]["std"] >= 0

    def test_evaluate_with_custom_metrics(self, simple_xy, simple_model):
        X, y = simple_xy
        metrics = [AccuracyMetric(), F1Metric()]
        evaluator = CrossValidationEvaluator(cv_folds=3)
        results = evaluator.evaluate(simple_model, X, y, metrics=metrics)

        assert len(results) == 2
        assert "accuracy" in results
        assert "f1" in results

    def test_evaluate_handles_errors(self, simple_xy):
        X, y = simple_xy
        # Create a model that will fail
        from data_complexity.experiments.ml import SVMModel

        model = SVMModel(C=-1)  # Invalid C parameter
        evaluator = CrossValidationEvaluator(cv_folds=3)
        results = evaluator.evaluate(model, X, y)

        # Should return NaN for all metrics on error
        assert all(np.isnan(v["mean"]) for v in results.values())

    def test_repr(self):
        evaluator = CrossValidationEvaluator(cv_folds=3)
        assert "CrossValidationEvaluator" in repr(evaluator)
        assert "3" in repr(evaluator)


class TestTrainTestSplitEvaluator:
    """Tests for TrainTestSplitEvaluator."""

    def test_default_parameters(self):
        evaluator = TrainTestSplitEvaluator()
        assert evaluator.test_size == 0.2
        assert evaluator.random_state == 42

    def test_custom_parameters(self):
        evaluator = TrainTestSplitEvaluator(test_size=0.3, random_state=123)
        assert evaluator.test_size == 0.3
        assert evaluator.random_state == 123

    def test_evaluate(self, simple_xy, simple_model):
        X, y = simple_xy
        evaluator = TrainTestSplitEvaluator(test_size=0.3)
        results = evaluator.evaluate(simple_model, X, y)

        assert "accuracy" in results
        assert "f1" in results
        assert 0 <= results["accuracy"]["mean"] <= 1
        assert results["accuracy"]["std"] == 0.0  # Single split has no std

    def test_evaluate_with_custom_metrics(self, simple_xy, simple_model):
        X, y = simple_xy
        metrics = [AccuracyMetric()]
        evaluator = TrainTestSplitEvaluator()
        results = evaluator.evaluate(simple_model, X, y, metrics=metrics)

        assert len(results) == 1
        assert "accuracy" in results

    def test_repr(self):
        evaluator = TrainTestSplitEvaluator(test_size=0.3)
        assert "TrainTestSplitEvaluator" in repr(evaluator)
        assert "0.3" in repr(evaluator)


class TestEvaluatorFactoryFunctions:
    """Tests for evaluator factory functions."""

    def test_get_default_evaluator(self):
        evaluator = get_default_evaluator()
        assert isinstance(evaluator, CrossValidationEvaluator)
        assert evaluator.cv_folds == 5

    def test_get_default_evaluator_custom_folds(self):
        evaluator = get_default_evaluator(cv_folds=3)
        assert isinstance(evaluator, CrossValidationEvaluator)
        assert evaluator.cv_folds == 3
