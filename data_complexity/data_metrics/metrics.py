# author: Matt Clifford <matt.clifford@bristol.ac.uk>
"""
All complexity metrics and visualisation methods.
"""
import numpy as np
from data_complexity.data_metrics.pycol import Complexity
from data_complexity.data_metrics.feature import FEATURE_METRICS
from data_complexity.data_metrics.instance import INSTANCE_METRICS
from data_complexity.data_metrics.structural import STRUCTURAL_METRICS
from data_complexity.data_metrics.multiresolution import MULTIRESOLUTION_METRICS
from data_complexity.data_metrics.classical import CLASSICAL_METRICS
from data_complexity.data_metrics.distributional import DISTRIBUTIONAL_METRICS


def _to_scalar(value) -> float:
    """Convert array-like metric values to a single float via mean aggregation."""
    if isinstance(value, (np.ndarray, list)):
        return float(np.mean(value))
    return float(value)


class ComplexityMetrics:
    def __init__(
            self,
            dataset: dict,
            distance_func: str = "default"
            ):
        """
        Get all complexity metrics using pycol on a dataset.

        Parameters
        ----------
        dataset : dict
            Dict with keys 'X' and 'y' and numpy array values.
        distance_func : str
            Distance function to use (default: "default").
        """
        self.pycol_complexity = Complexity(
            dataset=dataset, file_type="array", distance_func=distance_func
        )

    def _compute_pycol(self, metrics: list, scalar: bool) -> dict:
        """Compute a list of PyColAbstractMetric instances."""
        if scalar:
            return {m.metric_name: _to_scalar(m.compute_from_complexity(self.pycol_complexity))
                    for m in metrics}
        return {m.metric_name: m.compute_from_complexity(self.pycol_complexity)
                for m in metrics}

    def _compute_base(self, metrics: list) -> dict:
        """Compute a list of BaseAbstractMetric instances (use X, y directly)."""
        return {m.metric_name: m.compute(self.pycol_complexity.X, self.pycol_complexity.y)
                for m in metrics}

    def get_all_metrics_scalar(self) -> dict:
        all_metrics = {}
        categories = [
            (self.feature_overlap_scalar,         FEATURE_METRICS),
            (self.instance_overlap_scalar,        INSTANCE_METRICS),
            (self.structural_overlap_scalar,      STRUCTURAL_METRICS),
            (self.multiresolution_overlap_scalar, MULTIRESOLUTION_METRICS),
            (self.classical_measures_scalar,      CLASSICAL_METRICS),
            (self.distributional_measures_scalar, DISTRIBUTIONAL_METRICS),
        ]
        for fn, fallback_metrics in categories:
            try:
                all_metrics.update(fn())
            except Exception as e:
                print(f"Warning: {fn.__name__} failed: {e}")
                for m in fallback_metrics:
                    all_metrics[m.metric_name] = np.nan
        return all_metrics

    def get_all_metrics_full(self) -> dict:
        all_metrics = {}
        all_metrics.update(self.feature_overlap_full())
        all_metrics.update(self.instance_overlap_full())
        all_metrics.update(self.structural_overlap_full())
        all_metrics.update(self.multiresolution_overlap_full())
        all_metrics.update(self.classical_measures_full())
        all_metrics.update(self.distributional_measures_scalar())
        return all_metrics

    '''FEATURE OVERLAP MEASURES'''
    def feature_overlap_scalar(self) -> dict:
        return self._compute_pycol(FEATURE_METRICS, scalar=True)

    def feature_overlap_full(self) -> dict:
        return self._compute_pycol(FEATURE_METRICS, scalar=False)

    '''INSTANCE OVERLAP MEASURES'''
    def instance_overlap_scalar(self) -> dict:
        return self._compute_pycol(INSTANCE_METRICS, scalar=True)

    def instance_overlap_full(self) -> dict:
        return self._compute_pycol(INSTANCE_METRICS, scalar=False)

    '''STRUCTURAL OVERLAP MEASURES'''
    def structural_overlap_scalar(self) -> dict:
        return self._compute_pycol(STRUCTURAL_METRICS, scalar=True)

    def structural_overlap_full(self) -> dict:
        return self._compute_pycol(STRUCTURAL_METRICS, scalar=False)

    '''MULTIRESOLUTION OVERLAP MEASURES'''
    def multiresolution_overlap_scalar(self) -> dict:
        return self._compute_pycol(MULTIRESOLUTION_METRICS, scalar=True)

    def multiresolution_overlap_full(self) -> dict:
        return self._compute_pycol(MULTIRESOLUTION_METRICS, scalar=False)

    '''CLASSICAL MEASURES'''
    def classical_measures_scalar(self) -> dict:
        """
        Classical dataset measures.

        Returns basic dataset statistics that characterize the data distribution
        but are not complexity measures based on class overlap.

        Returns
        -------
        dict
            Dictionary containing:
            - 'IR': Imbalance Ratio (majority/minority class ratio)
        """
        return self._compute_base(CLASSICAL_METRICS)

    def classical_measures_full(self) -> dict:
        """
        Classical dataset measures (full version).

        For classical measures, the full and scalar versions return the same values
        since these are dataset-level statistics, not per-feature or per-instance.

        Returns
        -------
        dict
            Dictionary containing:
            - 'IR': Imbalance Ratio (majority/minority class ratio)
        """
        return self.classical_measures_scalar()

    '''DISTRIBUTIONAL MEASURES'''
    def distributional_measures_scalar(self) -> dict:
        """
        Distributional and boundary complexity measures.

        Returns statistical and geometric measures of class separation.

        Returns
        -------
        dict
            Dictionary containing Silhouette, Bhattacharyya, Wasserstein,
            SVM_SVR, and TwoNN_ID values.
        """
        return self._compute_base(DISTRIBUTIONAL_METRICS)
