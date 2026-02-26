# author: Matt Clifford <matt.clifford@bristol.ac.uk>
"""
Instance overlap complexity metrics.
"""
from data_complexity.data_metrics.abstract_metrics import PyColAbstractMetric


class RaugMetric(PyColAbstractMetric):
    """Augmented R value (instance hardness)."""

    @property
    def metric_name(self) -> str:
        return 'Raug'

    def compute_from_complexity(self, complexity):
        return complexity.R_value()


class DegOverlapMetric(PyColAbstractMetric):
    """Degree of overlap."""

    @property
    def metric_name(self) -> str:
        return 'deg_overlap'

    def compute_from_complexity(self, complexity):
        return complexity.deg_overlap()


class N3Metric(PyColAbstractMetric):
    """Error rate of the 1-nearest neighbour classifier."""

    @property
    def metric_name(self) -> str:
        return 'N3'

    def compute_from_complexity(self, complexity):
        return complexity.N3()


class SIMetric(PyColAbstractMetric):
    """Separability Index."""

    @property
    def metric_name(self) -> str:
        return 'SI'

    def compute_from_complexity(self, complexity):
        return complexity.SI()


class N4Metric(PyColAbstractMetric):
    """Non-linearity of the 1-NN classifier."""

    @property
    def metric_name(self) -> str:
        return 'N4'

    def compute_from_complexity(self, complexity):
        return complexity.N4()


class KDNMetric(PyColAbstractMetric):
    """K-Disagreeing Neighbours."""

    @property
    def metric_name(self) -> str:
        return 'kDN'

    def compute_from_complexity(self, complexity):
        return complexity.kDN()


class D3Metric(PyColAbstractMetric):
    """Disjunct size (D3 value)."""

    @property
    def metric_name(self) -> str:
        return 'D3'

    def compute_from_complexity(self, complexity):
        return complexity.D3_value()


class CMMetric(PyColAbstractMetric):
    """Class Complexity Measure."""

    @property
    def metric_name(self) -> str:
        return 'CM'

    def compute_from_complexity(self, complexity):
        return complexity.CM()


INSTANCE_METRICS: list = [
    RaugMetric(), DegOverlapMetric(), N3Metric(), SIMetric(),
    N4Metric(), KDNMetric(), D3Metric(), CMMetric()
]
