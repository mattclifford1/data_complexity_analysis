# author: Matt Clifford <matt.clifford@bristol.ac.uk>
"""
Structural overlap complexity metrics.
"""
from data_complexity.data_metrics.abstract_metrics import PyColAbstractMetric


class N1Metric(PyColAbstractMetric):
    """Fraction of borderline points (N1)."""

    @property
    def metric_name(self) -> str:
        return 'N1'

    def compute_from_complexity(self, complexity):
        return complexity.N1()


class T1Metric(PyColAbstractMetric):
    """Fraction of hyperspheres covering data (T1)."""

    @property
    def metric_name(self) -> str:
        return 'T1'

    def compute_from_complexity(self, complexity):
        return complexity.T1()


class ClustMetric(PyColAbstractMetric):
    """Number of clusters per class (Clust)."""

    @property
    def metric_name(self) -> str:
        return 'Clust'

    def compute_from_complexity(self, complexity):
        return complexity.Clust()


STRUCTURAL_METRICS: list = [
    N1Metric(), T1Metric(), ClustMetric()
]
