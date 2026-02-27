# author: Matt Clifford <matt.clifford@bristol.ac.uk>
"""
Multiresolution overlap complexity metrics.
"""
from data_complexity.data_metrics.abstract_metrics import PyColAbstractMetric


class MRCAMetric(PyColAbstractMetric):
    """Multiresolution Class Aggregate (MRCA)."""

    @property
    def metric_name(self) -> str:
        return 'MRCA'

    def compute_from_complexity(self, complexity):
        return complexity.MRCA()


class C1Metric(PyColAbstractMetric):
    """Entropy of class proportions (C1)."""

    @property
    def metric_name(self) -> str:
        return 'C1'

    def compute_from_complexity(self, complexity):
        return complexity.C1()


class PurityMetric(PyColAbstractMetric):
    """Purity measure."""

    @property
    def metric_name(self) -> str:
        return 'Purity'

    def compute_from_complexity(self, complexity):
        return complexity.purity()


class C2Metric(PyColAbstractMetric):
    """Multi-resolution class complexity (C2)."""

    @property
    def metric_name(self) -> str:
        return 'C2'

    def compute_from_complexity(self, complexity):
        return complexity.C2()


class NeighbourhoodSeparabilityMetric(PyColAbstractMetric):
    """Neighbourhood Separability measure."""

    @property
    def metric_name(self) -> str:
        return 'NeighbourhoodSeparability'

    def compute_from_complexity(self, complexity):
        return complexity.neighbourhood_separability()


MULTIRESOLUTION_METRICS: list = [
    MRCAMetric(), C1Metric(), PurityMetric(), C2Metric(), NeighbourhoodSeparabilityMetric()
]
