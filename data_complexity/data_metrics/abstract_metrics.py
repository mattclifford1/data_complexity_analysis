# author: Matt Clifford <matt.clifford@bristol.ac.uk>
"""
Abstract base classes for complexity metrics.
"""
from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class BaseAbstractMetric(ABC):
    @abstractmethod
    def compute(self, X: np.ndarray, y: np.ndarray) -> Union[float, dict]:
        """
        Compute the metric.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data
        y : array-like, shape (n_samples,)
            The target labels

        Returns
        -------
        score : float or dict
            The computed metric score
        """
        pass


class PyColAbstractMetric(BaseAbstractMetric):
    """Base class for PyCol-backed complexity metrics.

    Subclasses implement compute_from_complexity(). The compute(X, y) method
    is provided for standalone use and creates a PyCol Complexity object internally.
    ComplexityMetrics creates one shared Complexity object and calls
    compute_from_complexity() directly for efficiency.
    """

    def compute(self, X: np.ndarray, y: np.ndarray) -> Union[float, dict]:
        from data_complexity.data_metrics.pycol import Complexity
        comp = Complexity(dataset={'X': X, 'y': y}, file_type='array')
        return self.compute_from_complexity(comp)

    @abstractmethod
    def compute_from_complexity(self, complexity) -> Union[float, dict]:
        """Compute using a pre-built PyCol Complexity object (efficient path)."""
        pass

    @property
    def metric_name(self) -> str:
        return self.__class__.__name__
