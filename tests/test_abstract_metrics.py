"""
Tests for the AbstractMetric base class.

Verifies that the abstract base class enforces the expected interface
for custom metric implementations.
"""
import pytest
import numpy as np
from data_complexity.abstract_metrics import AbstractMetric


class TestAbstractMetricInterface:
    """Tests verifying AbstractMetric enforces proper interface."""

    def test_cannot_instantiate_directly(self):
        """Verify AbstractMetric cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractMetric()

    def test_subclass_must_implement_compute(self):
        """Verify subclass without compute method cannot be instantiated."""
        class IncompleteMetric(AbstractMetric):
            pass

        with pytest.raises(TypeError):
            IncompleteMetric()

    def test_subclass_with_compute_can_instantiate(self):
        """Verify subclass implementing compute can be instantiated."""
        class CompleteMetric(AbstractMetric):
            def compute(self, X, y):
                return 0.5

        metric = CompleteMetric()
        assert metric is not None

    def test_compute_receives_correct_arguments(self):
        """Verify compute method receives X and y as expected."""
        class RecordingMetric(AbstractMetric):
            def __init__(self):
                self.received_X = None
                self.received_y = None

            def compute(self, X, y):
                self.received_X = X
                self.received_y = y
                return 0.0

        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        metric = RecordingMetric()
        metric.compute(X, y)

        np.testing.assert_array_equal(metric.received_X, X)
        np.testing.assert_array_equal(metric.received_y, y)

    def test_compute_can_return_float(self):
        """Verify compute can return a float value."""
        class FloatMetric(AbstractMetric):
            def compute(self, X, y):
                return 0.75

        metric = FloatMetric()
        result = metric.compute(np.array([[1]]), np.array([0]))
        assert result == 0.75
        assert isinstance(result, float)

    def test_compute_can_return_dict(self):
        """Verify compute can return a dictionary of metrics."""
        class DictMetric(AbstractMetric):
            def compute(self, X, y):
                return {'metric_a': 0.5, 'metric_b': 0.8}

        metric = DictMetric()
        result = metric.compute(np.array([[1]]), np.array([0]))
        assert result == {'metric_a': 0.5, 'metric_b': 0.8}


class TestCustomMetricExample:
    """Example showing how to implement a custom metric."""

    def test_custom_class_imbalance_metric(self):
        """
        Example: A simple class imbalance metric.

        Demonstrates implementing a concrete metric that calculates
        the ratio of minority to majority class samples.
        """
        class ClassImbalanceRatio(AbstractMetric):
            """Computes minority/majority class ratio (0 to 1)."""

            def compute(self, X, y):
                unique, counts = np.unique(y, return_counts=True)
                if len(counts) < 2:
                    return 1.0  # Single class = "balanced"
                return counts.min() / counts.max()

        metric = ClassImbalanceRatio()

        # Balanced dataset (50/50)
        X_balanced = np.random.randn(100, 2)
        y_balanced = np.array([0] * 50 + [1] * 50)
        assert metric.compute(X_balanced, y_balanced) == 1.0

        # Imbalanced dataset (80/20)
        y_imbalanced = np.array([0] * 80 + [1] * 20)
        assert metric.compute(X_balanced, y_imbalanced) == 0.25
