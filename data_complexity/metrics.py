# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
All complexity metrics and visualisation methods
'''
import numpy as np
from data_complexity.pycol import Complexity


class complexity_metrics:
    def __init__(
            self,
            dataset: dict,
            distance_func="default"
            ):
        '''
        get all complexity metrics using pycol on a dataset
            dataset: dict with keys 'X' and 'y' and numpy array values
        '''
        self.pycol_complexity = Complexity(
            dataset=dataset, file_type="array", distance_func=distance_func
        )

    def _compute_imbalance_ratio(self):
        """
        Calculate the Imbalance Ratio (IR).

        IR = n_majority / n_minority

        For binary classification, this is the ratio of the majority class
        to the minority class. For multiclass, it's the ratio of the largest
        class to the smallest class.

        Returns
        -------
        float
            Imbalance ratio (>= 1.0). Returns np.inf if minority class is empty,
            1.0 if single class.
        """
        class_count = self.pycol_complexity.class_count

        # Handle edge case: single class
        if len(class_count) < 2:
            return 1.0

        min_count = np.min(class_count)
        max_count = np.max(class_count)

        # Handle edge case: empty minority class
        if min_count == 0:
            return np.inf

        return max_count / min_count


    def get_all_metrics_scalar(self):
        all_metrics = {}
        all_metrics.update(self.feature_overlap_scalar())
        all_metrics.update(self.instance_overlap_scalar())
        all_metrics.update(self.structural_overlap_scalar())
        all_metrics.update(self.classical_measures_scalar())
        return all_metrics
    
    def get_all_metrics_full(self):
        all_metrics = {}
        all_metrics.update(self.feature_overlap_full())
        all_metrics.update(self.instance_overlap_full())
        all_metrics.update(self.structural_overlap_full())
        all_metrics.update(self.multiresolution_overlap_full())
        all_metrics.update(self.classical_measures_full())
        return all_metrics
    

    '''FEATURE OVERLAP MEASURES'''
    def feature_overlap_scalar(self):
        feature_overlap, f_names = self.pycol_complexity.feature_overlap(viz=False)
        metrics = {}
        for measure, value in zip(f_names, feature_overlap):
            metrics[measure] = value
        return metrics

    def feature_overlap_full(self):
        return {
            'F1': self.pycol_complexity.F1(),
            'F1v': self.pycol_complexity.F1v(),
            'F2': self.pycol_complexity.F2(),
            'F3': self.pycol_complexity.F3(),
            'F4': self.pycol_complexity.F4()
        }

    '''INSTANCE OVERLAP MEASURES'''
    def instance_overlap_scalar(self):
        instance_overlap, i_names = self.pycol_complexity.instance_overlap(viz=False)
        metrics = {}
        for measure, value in zip(i_names, instance_overlap):
            metrics[measure] = value
        return metrics
    
    def instance_overlap_full(self):
        '''Instance Overlap Measures'''
        return {
            'Raug': self.pycol_complexity.R_value(),
            'deg_overlap': self.pycol_complexity.deg_overlap(),
            'N3': self.pycol_complexity.N3(),
            'SI': self.pycol_complexity.SI(),
            'N4': self.pycol_complexity.N4(),
            'kDN': self.pycol_complexity.kDN(),
            'D3': self.pycol_complexity.D3_value(),
            'CM': self.pycol_complexity.CM()
        }
    
    '''STRUCTURAL OVERLAP MEASURES'''
    def structural_overlap_scalar(self):
        structural_overlap, s_names = self.pycol_complexity.structure_overlap(viz=False)
        metrics = {}
        for measure, value in zip(s_names, structural_overlap):
            metrics[measure] = value
        return metrics
    
    def structural_overlap_full(self):
        '''Structural Overlap Measures'''
        return {
            'N1': self.pycol_complexity.N1(),
            'T1': self.pycol_complexity.T1(),
            'Clust': self.pycol_complexity.Clust()
        }
    
    '''MULTIRESOLUTION OVERLAP MEASURES'''
    def multiresolution_overlap_full(self):
        '''Multiresolution Overlap Measures'''
        return {
            'MRCA': self.pycol_complexity.MRCA(),
            'C1': self.pycol_complexity.C1(),
            'Purity': self.pycol_complexity.purity()
        }

    '''CLASSICAL MEASURES'''
    def classical_measures_scalar(self):
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
        return {
            'IR': self._compute_imbalance_ratio()
        }

    def classical_measures_full(self):
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
        return {
            'IR': self._compute_imbalance_ratio()
        }


if __name__ == "__main__":
    from data_loaders import get_dataset
    dataset = get_dataset("Wine")
    print(f"Dataset loaded: {dataset.name}\n")
    dataset.plot_dataset(terminal_plot=True)

    complexity = complexity_metrics(dataset=dataset.get_data_dict())
    all_metrics = complexity.get_all_metrics_scalar()
    for metric_name, metric_value in all_metrics.items():
        print(f'{metric_name}: {metric_value}')

