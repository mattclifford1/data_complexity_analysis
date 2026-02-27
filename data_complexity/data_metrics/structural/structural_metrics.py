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


class N2Metric(PyColAbstractMetric):
    """Ratio of intra/inter class nearest neighbour distance (N2)."""

    @property
    def metric_name(self) -> str:
        return 'N2'

    def compute_from_complexity(self, complexity):
        return complexity.N2()


class ONBMetric(PyColAbstractMetric):
    """Overlap of Neighbourhoods (ONB)."""

    @property
    def metric_name(self) -> str:
        return 'ONB'

    def compute_from_complexity(self, complexity):
        return complexity.ONB()


class LSCAvgMetric(PyColAbstractMetric):
    """Local Set Average Cardinality (LSCAvg)."""

    @property
    def metric_name(self) -> str:
        return 'LSCAvg'

    def compute_from_complexity(self, complexity):
        return complexity.LSC()


class DBCMetric(PyColAbstractMetric):
    """Decision Boundary Complexity (DBC)."""

    @property
    def metric_name(self) -> str:
        return 'DBC'

    def compute_from_complexity(self, complexity):
        return complexity.DBC()


class NSGMetric(PyColAbstractMetric):
    """Number of Same-class Groups (NSG)."""

    @property
    def metric_name(self) -> str:
        return 'NSG'

    def compute_from_complexity(self, complexity):
        return complexity.NSG()


class ICSVMetric(PyColAbstractMetric):
    """Intra-class Spatial Variability (ICSV)."""

    @property
    def metric_name(self) -> str:
        return 'ICSV'

    def compute_from_complexity(self, complexity):
        return complexity.ICSV()


STRUCTURAL_METRICS: list = [
    N1Metric(), T1Metric(), ClustMetric(),
    N2Metric(), ONBMetric(), LSCAvgMetric(), DBCMetric(), NSGMetric(), ICSVMetric()
]
