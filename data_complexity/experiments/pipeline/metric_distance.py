"""
Distance measures between metrics for correlation analysis.

Provides an abstract base class and concrete implementations for measuring
the relationship between two 1D arrays of values. The default is Pearson
correlation, but Spearman, Kendall tau, mutual information, and Euclidean
distance are also available.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats


class DistanceBetweenMetrics(ABC):
    """Abstract base class for measuring relationships between metric arrays."""

    @abstractmethod
    def compute(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float | None]:
        """
        Compute the distance/association between two arrays.

        Parameters
        ----------
        x, y : np.ndarray
            1D arrays of the same length.

        Returns
        -------
        tuple[float, float | None]
            (distance_value, p_value_or_None). p_value is None when the measure
            does not produce a significance test (e.g. mutual information).
        """
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable label for plot titles (e.g. 'Pearson r')."""
        ...

    @property
    def name(self) -> str:
        """Filesystem-safe identifier derived from display_name (e.g. 'pearson_r')."""
        n = self.display_name.lower().replace("ρ", "rho").replace("τ", "tau")
        return re.sub(r"[^a-z0-9]+", "_", n).strip("_")

    @property
    def signed(self) -> bool:
        """True if values range from -1 to 1 (correlation-like). False for 0…∞."""
        return False


class PearsonCorrelation(DistanceBetweenMetrics):
    """Pearson product-moment correlation coefficient."""

    def compute(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float | None]:
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)

    @property
    def display_name(self) -> str:
        return "Pearson r"

    @property
    def signed(self) -> bool:
        return True


class SpearmanCorrelation(DistanceBetweenMetrics):
    """Spearman rank-order correlation coefficient."""

    def compute(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float | None]:
        r, p = stats.spearmanr(x, y)
        return float(r), float(p)

    @property
    def display_name(self) -> str:
        return "Spearman ρ"

    @property
    def signed(self) -> bool:
        return True


class KendallTau(DistanceBetweenMetrics):
    """Kendall tau rank correlation coefficient."""

    def compute(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float | None]:
        tau, p = stats.kendalltau(x, y)
        return float(tau), float(p)

    @property
    def display_name(self) -> str:
        return "Kendall τ"

    @property
    def signed(self) -> bool:
        return True


class MutualInformation(DistanceBetweenMetrics):
    """Mutual information between two continuous variables (via k-NN estimation)."""

    def compute(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float | None]:
        from sklearn.feature_selection import mutual_info_regression

        mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=0)[0]
        return float(mi), None

    @property
    def display_name(self) -> str:
        return "Mutual Information"


class EuclideanDistance(DistanceBetweenMetrics):
    """Euclidean distance between z-score-normalised arrays."""

    def compute(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float | None]:
        x_std = float(np.std(x))
        y_std = float(np.std(y))
        x_norm = (x - np.mean(x)) / x_std if x_std > 0 else x - np.mean(x)
        y_norm = (y - np.mean(y)) / y_std if y_std > 0 else y - np.mean(y)
        return float(np.linalg.norm(x_norm - y_norm)), None

    @property
    def display_name(self) -> str:
        return "Euclidean Distance"
