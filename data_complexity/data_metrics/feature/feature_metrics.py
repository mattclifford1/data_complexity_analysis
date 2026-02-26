# author: Matt Clifford <matt.clifford@bristol.ac.uk>
"""
Feature overlap complexity metrics.
"""
from data_complexity.data_metrics.abstract_metrics import AbstractComplexityMetric


class F1Metric(AbstractComplexityMetric):
    """Maximum Fisher's Discriminant Ratio."""

    @property
    def metric_name(self) -> str:
        return 'F1'

    def compute_from_complexity(self, complexity):
        return complexity.F1()


class F1vMetric(AbstractComplexityMetric):
    """Directional-vector Maximum Fisher's Discriminant Ratio."""

    @property
    def metric_name(self) -> str:
        return 'F1v'

    def compute_from_complexity(self, complexity):
        return complexity.F1v()


class F2Metric(AbstractComplexityMetric):
    """Volume of Overlapping Region."""

    @property
    def metric_name(self) -> str:
        return 'F2'

    def compute_from_complexity(self, complexity):
        return complexity.F2()


class F3Metric(AbstractComplexityMetric):
    """Maximum Individual Feature Efficiency."""

    @property
    def metric_name(self) -> str:
        return 'F3'

    def compute_from_complexity(self, complexity):
        return complexity.F3()


class F4Metric(AbstractComplexityMetric):
    """Collective Feature Efficiency."""

    @property
    def metric_name(self) -> str:
        return 'F4'

    def compute_from_complexity(self, complexity):
        return complexity.F4()


FEATURE_METRICS: list = [
    F1Metric(), F1vMetric(), F2Metric(), F3Metric(), F4Metric()
]
