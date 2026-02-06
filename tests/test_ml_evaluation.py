"""Tests for ML evaluation metrics and evaluators."""
import pytest
import numpy as np
from sklearn.datasets import make_classification

from data_complexity.model_experiments.ml import (
    # Metrics
    AccuracyMetric,
    F1Metric,
    # Evaluators
    AbstractEvaluator,
    CrossValidationEvaluator,
    TrainTestSplitEvaluator,
    get_default_evaluator,
    # Models
    LogisticRegressionModel,
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
