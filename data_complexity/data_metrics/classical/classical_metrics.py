# author: Matt Clifford <matt.clifford@bristol.ac.uk>
"""
Classical dataset measures.
"""
import numpy as np
from data_complexity.data_metrics.abstract_metrics import AbstractComplexityMetric


class ImbalanceRatioMetric(AbstractComplexityMetric):
    """
    Imbalance Ratio (IR).

    IR = n_majority / n_minority

    For binary classification, this is the ratio of the majority class
    to the minority class. For multiclass, it's the ratio of the largest
    class to the smallest class.
    """

    @property
    def metric_name(self) -> str:
        return 'IR'

    def compute_from_complexity(self, complexity) -> float:
        class_count = complexity.class_count

        if len(class_count) < 2:
            return 1.0

        min_count = np.min(class_count)
        max_count = np.max(class_count)

        if min_count == 0:
            return np.inf

        return float(max_count / min_count)


CLASSICAL_METRICS: list = [
    ImbalanceRatioMetric()
]
