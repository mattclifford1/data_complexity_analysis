# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
All complexity metrics and visualisation methods
'''
from pycol import Complexity


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
        

    def get_all_metrics_scalar(self):
        all_metrics = {}
        all_metrics.update(self.feature_overlap_scalar(viz=False))
        all_metrics.update(self.instance_overlap_scalar())
        all_metrics.update(self.structural_overlap_scalar())
        return all_metrics
    
    def get_all_metrics_full(self):
        all_metrics = {}
        all_metrics.update(self.feature_overlap_full())
        all_metrics.update(self.instance_overlap_full())
        all_metrics.update(self.structural_overlap_full())
        all_metrics.update(self.multiresolution_overlap_full())
        return all_metrics
    

    '''FEATURE OVERLAP MEASURES'''
    def feature_overlap_scalar(self, viz=True):
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
    

if __name__ == "__main__":
    from data_loaders import get_dataset
    dataset = get_dataset("Wine")
    print(f"Dataset loaded: {dataset.name}\n")
    dataset.plot_dataset(terminal_plot=True)

    complexity = complexity_metrics(dataset=dataset.get_data_dict())
    all_metrics = complexity.get_all_metrics_scalar()
    for metric_name, metric_value in all_metrics.items():
        print(f'{metric_name}: {metric_value}')

