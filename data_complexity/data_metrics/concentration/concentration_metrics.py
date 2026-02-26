# author: Matt Clifford <matt.clifford@bristol.ac.uk>
"""
Concentration dataset measures.
"""
import numpy as np
from data_complexity.data_metrics.abstract_metrics import BaseAbstractMetric


class Deltas(BaseAbstractMetric):
    """
    Deltas:
        - https://github.com/mattclifford1/linear_confidence/tree/main
        - https://arxiv.org/abs/2407.11878
    """

    @property
    def metric_name(self) -> str:
        return 'Deltas'

    def compute(self, X: np.ndarray, y: np.ndarray) -> float:
        # TODO: Implement Deltas metric from linear_confidence
        raise NotImplementedError("Deltas metric is not yet implemented.")




CONCENTRATION_METRICS: list = [
    Deltas()
]
