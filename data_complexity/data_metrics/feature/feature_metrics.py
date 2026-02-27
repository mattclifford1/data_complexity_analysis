# author: Matt Clifford <matt.clifford@bristol.ac.uk>
"""
Feature overlap complexity metrics.
"""
from data_complexity.data_metrics.abstract_metrics import PyColAbstractMetric


class F1Metric(PyColAbstractMetric):
    """Maximum Fisher's Discriminant Ratio."""

    @property
    def metric_name(self) -> str:
        return 'F1'

    def compute_from_complexity(self, complexity):
        return complexity.F1()


class F1vMetric(PyColAbstractMetric):
    """Directional-vector Maximum Fisher's Discriminant Ratio."""

    @property
    def metric_name(self) -> str:
        return 'F1v'

    def compute_from_complexity(self, complexity):
        return complexity.F1v()


class F2Metric(PyColAbstractMetric):
    """Volume of Overlapping Region."""

    @property
    def metric_name(self) -> str:
        return 'F2'

    def compute_from_complexity(self, complexity):
        return complexity.F2()


class F3Metric(PyColAbstractMetric):
    """Maximum Individual Feature Efficiency."""

    @property
    def metric_name(self) -> str:
        return 'F3'

    def compute_from_complexity(self, complexity):
        return complexity.F3()


class F4Metric(PyColAbstractMetric):
    """Collective Feature Efficiency."""

    @property
    def metric_name(self) -> str:
        return 'F4'

    def compute_from_complexity(self, complexity):
        return complexity.F4()


class INMetric(PyColAbstractMetric):
    """Input Noise measure."""

    @property
    def metric_name(self) -> str:
        return 'IN'

    def compute_from_complexity(self, complexity):
        return complexity.input_noise()


FEATURE_METRICS: list = [
    F1Metric(), F1vMetric(), F2Metric(), F3Metric(), F4Metric(), INMetric()
]
