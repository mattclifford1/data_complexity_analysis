# author: Matt Clifford <matt.clifford@bristol.ac.uk>
"""
Distributional and boundary complexity metrics.

These measures quantify class overlap using statistical distribution properties,
decision boundary geometry, and intrinsic data dimensionality.
"""
from typing import Union
import numpy as np
from data_complexity.data_metrics.abstract_metrics import BaseAbstractMetric


class SilhouetteScoreMetric(BaseAbstractMetric):
    """Mean Silhouette Coefficient — measures cluster cohesion vs separation.

    Higher values indicate better-separated classes (lower complexity).
    Range: [-1, 1].
    """

    @property
    def metric_name(self) -> str:
        return 'Silhouette'

    def compute(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import silhouette_score
        return float(silhouette_score(X, y))


class BhattacharyyaMetric(BaseAbstractMetric):
    """Bhattacharyya coefficient — measures distributional overlap between classes.

    Estimated per feature via histogram overlap, then averaged over all feature-pair
    combinations. Higher values indicate more overlap (higher complexity).
    Range: [0, 1].
    """

    @property
    def metric_name(self) -> str:
        return 'Bhattacharyya'

    def compute(self, X: np.ndarray, y: np.ndarray) -> float:
        classes = np.unique(y)
        n_features = X.shape[1]
        eps = 1e-10
        coeffs = []
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                Xi = X[y == classes[i]]
                Xj = X[y == classes[j]]
                for f in range(n_features):
                    # Shared bin edges over both class distributions for this feature
                    combined = np.concatenate([Xi[:, f], Xj[:, f]])
                    bins = np.linspace(combined.min(), combined.max(), 11)
                    hi, _ = np.histogram(Xi[:, f], bins=bins, density=True)
                    hj, _ = np.histogram(Xj[:, f], bins=bins, density=True)
                    bin_width = bins[1] - bins[0]
                    # Bhattacharyya coefficient: integral of sqrt(p*q) dx ≈ sum(sqrt(pi*qi)) * dx
                    bc = float(np.sum(np.sqrt((hi + eps) * (hj + eps))) * bin_width)
                    coeffs.append(bc)
        return float(np.mean(coeffs))


class WassersteinDistanceMetric(BaseAbstractMetric):
    """Mean Wasserstein (Earth Mover's) distance between class distributions.

    Computed per feature for each class pair, then averaged. Scale-normalised
    by feature standard deviation so features contribute equally.
    Higher values indicate more separation (lower complexity).
    Range: [0, ∞).
    """

    @property
    def metric_name(self) -> str:
        return 'Wasserstein'

    def compute(self, X: np.ndarray, y: np.ndarray) -> float:
        from scipy.stats import wasserstein_distance
        classes = np.unique(y)
        n_features = X.shape[1]
        distances = []
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                Xi = X[y == classes[i]]
                Xj = X[y == classes[j]]
                for f in range(n_features):
                    std = X[:, f].std()
                    scale = std if std > 1e-10 else 1.0
                    d = wasserstein_distance(Xi[:, f] / scale, Xj[:, f] / scale)
                    distances.append(d)
        return float(np.mean(distances))


class SVMSupportVectorRatioMetric(BaseAbstractMetric):
    """SVM support vector ratio — fraction of training samples that are support vectors.

    Fit an RBF-kernel SVM and return n_support_vectors / n_samples.
    Higher ratio indicates a more complex decision boundary.
    Range: [0, 1].
    """

    @property
    def metric_name(self) -> str:
        return 'SVM_SVR'

    def compute(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.svm import SVC
        clf = SVC(kernel='rbf', C=1.0)
        clf.fit(X, y)
        return float(clf.support_vectors_.shape[0] / X.shape[0])


class IntrinsicDimensionalityMetric(BaseAbstractMetric):
    """TwoNN intrinsic dimensionality estimate.

    Uses the TwoNN estimator (Facco et al., 2017) to estimate the intrinsic
    dimension of the data manifold. Higher values indicate a more complex
    data geometry. Range: [1, n_features].
    """

    @property
    def metric_name(self) -> str:
        return 'TwoNN_ID'

    def compute(self, X: np.ndarray, y: np.ndarray) -> float:
        import skdim
        if X.shape[0] < 5:
            return float('nan')
        estimator = skdim.id.TwoNN()
        estimator.fit(X)
        return float(estimator.dimension_)


DISTRIBUTIONAL_METRICS: list = [
    SilhouetteScoreMetric(),
    BhattacharyyaMetric(),
    WassersteinDistanceMetric(),
    SVMSupportVectorRatioMetric(),
    IntrinsicDimensionalityMetric(),
]
