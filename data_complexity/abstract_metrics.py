# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Generic class for metrics to inherit functionality from
'''
from abc import ABC, abstractmethod


class abstract_metric(ABC):
    @abstractmethod
    def compute(self, X, y):
        '''
        Compute the metric

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data
        y : array-like, shape (n_samples,)
            The target labels

        Returns
        -------
        score : float
            The computed metric score
        '''
        pass