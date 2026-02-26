# author: Matt Clifford <matt.clifford@bristol.ac.uk>
"""
All complexity metrics and visualisation methods.
"""
import numpy as np
from data_complexity.data_metrics.pycol import Complexity


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

    def get_all_metrics_scalar(self) -> dict:
        all_metrics = {}

        try:
            all_metrics.update(self.feature_overlap_scalar())
        except Exception as e:
            print(f"Warning: Feature overlap metrics failed: {e}")
            all_metrics.update({
                'F1': np.nan, 'F1v': np.nan, 'F2': np.nan,
                'F3': np.nan, 'F4': np.nan, 'IN': np.nan
            })

        try:
            all_metrics.update(self.instance_overlap_scalar())
        except Exception as e:
            print(f"Warning: Instance overlap metrics failed: {e}")
            instance_names = ['R-value', 'Raug', 'degOver', 'N3', 'SI', 'N4',
                            'kDN', 'D3', 'CM', 'wCM', 'dwCM',
                            'Borderline Examples', 'IPoints']
            for name in instance_names:
                all_metrics[name] = np.nan

        try:
            all_metrics.update(self.structural_overlap_scalar())
        except Exception as e:
            print(f"Warning: Structural overlap metrics failed: {e}")
            structural_names = ['N1', 'T1', 'Clust', 'ONB', 'LSCAvg',
                              'DBC', 'N2', 'NSG', 'ICSV']
            for name in structural_names:
                all_metrics[name] = np.nan

        try:
            all_metrics.update(self.classical_measures_scalar())
        except Exception as e:
            print(f"Warning: Classical measures failed: {e}")
            classical_names = ['MRCA', 'C1', 'C2', 'Purity', 'Neighbourhood Separability']
            for name in classical_names:
                all_metrics[name] = np.nan

        return all_metrics

    def get_all_metrics_full(self) -> dict:
        all_metrics = {}
        all_metrics.update(self.feature_overlap_full())
        all_metrics.update(self.instance_overlap_full())
        all_metrics.update(self.structural_overlap_full())
        all_metrics.update(self.multiresolution_overlap_full())
        all_metrics.update(self.classical_measures_full())
        return all_metrics

    '''FEATURE OVERLAP MEASURES'''
    def feature_overlap_scalar(self) -> dict:
        feature_overlap, f_names = self.pycol_complexity.feature_overlap(viz=False)
        return dict(zip(f_names, feature_overlap))

    def feature_overlap_full(self) -> dict:
        from data_complexity.data_metrics.feature import FEATURE_METRICS
        return {m.metric_name: m.compute_from_complexity(self.pycol_complexity)
                for m in FEATURE_METRICS}

    '''INSTANCE OVERLAP MEASURES'''
    def instance_overlap_scalar(self) -> dict:
        instance_overlap, i_names = self.pycol_complexity.instance_overlap(viz=False)
        return dict(zip(i_names, instance_overlap))

    def instance_overlap_full(self) -> dict:
        from data_complexity.data_metrics.instance import INSTANCE_METRICS
        return {m.metric_name: m.compute_from_complexity(self.pycol_complexity)
                for m in INSTANCE_METRICS}

    '''STRUCTURAL OVERLAP MEASURES'''
    def structural_overlap_scalar(self) -> dict:
        structural_overlap, s_names = self.pycol_complexity.structure_overlap(viz=False)
        return dict(zip(s_names, structural_overlap))

    def structural_overlap_full(self) -> dict:
        from data_complexity.data_metrics.structural import STRUCTURAL_METRICS
        return {m.metric_name: m.compute_from_complexity(self.pycol_complexity)
                for m in STRUCTURAL_METRICS}

    '''MULTIRESOLUTION OVERLAP MEASURES'''
    def multiresolution_overlap_full(self) -> dict:
        from data_complexity.data_metrics.multiresolution import MULTIRESOLUTION_METRICS
        return {m.metric_name: m.compute_from_complexity(self.pycol_complexity)
                for m in MULTIRESOLUTION_METRICS}

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
        from data_complexity.data_metrics.classical import CLASSICAL_METRICS
        return {m.metric_name: m.compute(self.pycol_complexity.X, self.pycol_complexity.y)
                for m in CLASSICAL_METRICS}

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
