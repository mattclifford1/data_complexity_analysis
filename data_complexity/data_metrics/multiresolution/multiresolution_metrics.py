# author: Matt Clifford <matt.clifford@bristol.ac.uk>
"""
Multiresolution overlap complexity metrics.
"""
from data_complexity.data_metrics.abstract_metrics import AbstractComplexityMetric


class MRCAMetric(AbstractComplexityMetric):
    """Multiresolution Class Aggregate (MRCA)."""

    @property
    def metric_name(self) -> str:
        return 'MRCA'

    def compute_from_complexity(self, complexity):
        return complexity.MRCA()


class C1Metric(AbstractComplexityMetric):
    """Entropy of class proportions (C1)."""

    @property
    def metric_name(self) -> str:
        return 'C1'

    def compute_from_complexity(self, complexity):
        return complexity.C1()


class PurityMetric(AbstractComplexityMetric):
    """Purity measure."""

    @property
    def metric_name(self) -> str:
        return 'Purity'

    def compute_from_complexity(self, complexity):
        return complexity.purity()


MULTIRESOLUTION_METRICS: list = [
    MRCAMetric(), C1Metric(), PurityMetric()
]
