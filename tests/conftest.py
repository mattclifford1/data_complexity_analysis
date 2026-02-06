"""
Pytest fixtures for data_complexity tests.

Provides reusable test datasets with varying complexity characteristics.
"""
import pytest
import numpy as np
from sklearn.datasets import make_moons, make_blobs, make_classification


@pytest.fixture
def simple_linearly_separable():
    """
    Two well-separated Gaussian blobs - low complexity dataset.

    Expected characteristics:
    - Low feature overlap (classes easily separable)
    - Low instance overlap (few ambiguous points)
    """
    X, y = make_blobs(
        n_samples=100,
        n_features=2,
        centers=2,
        cluster_std=0.5,
        random_state=42
    )
    return {'X': X, 'y': y}


@pytest.fixture
def moons_dataset():
    """
    Two interleaving half-moons - moderate complexity dataset.

    Expected characteristics:
    - Moderate feature overlap
    - Non-linear decision boundary required
    """
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    return {'X': X, 'y': y}


@pytest.fixture
def high_overlap_dataset():
    """
    Two heavily overlapping Gaussian blobs - high complexity dataset.

    Expected characteristics:
    - High feature overlap (classes hard to separate)
    - High instance overlap (many ambiguous points)
    """
    X, y = make_blobs(
        n_samples=100,
        n_features=2,
        centers=2,
        cluster_std=3.0,
        random_state=42
    )
    return {'X': X, 'y': y}


@pytest.fixture
def multiclass_dataset():
    """
    Three-class classification problem.

    Tests that metrics work with more than two classes.
    """
    X, y = make_blobs(
        n_samples=150,
        n_features=3,
        centers=3,
        cluster_std=1.0,
        random_state=42
    )
    return {'X': X, 'y': y}


@pytest.fixture
def high_dimensional_dataset():
    """
    Dataset with more features than typical 2D visualizations.

    Tests metric computation on higher-dimensional data.
    """
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    return {'X': X, 'y': y}


@pytest.fixture
def balanced_binary_dataset():
    """
    Perfectly balanced binary classification dataset (50/50).

    Expected IR = 1.0
    """
    X, y = make_blobs(
        n_samples=100,
        n_features=2,
        centers=2,
        cluster_std=1.0,
        random_state=42
    )
    return {'X': X, 'y': y}


@pytest.fixture
def imbalanced_binary_dataset():
    """
    Imbalanced binary classification dataset (80/20).

    Expected IR ≈ 4.0
    """
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        weights=[0.8, 0.2],
        random_state=42
    )
    return {'X': X, 'y': y}


@pytest.fixture
def highly_imbalanced_dataset():
    """
    Highly imbalanced binary classification dataset (90/10).

    Expected IR ≈ 9.0
    """
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42
    )
    return {'X': X, 'y': y}


@pytest.fixture
def single_class_dataset():
    """
    Degenerate case: all samples from one class.

    Expected IR = 1.0 (by convention)
    """
    X = np.random.randn(50, 2)
    y = np.zeros(50)
    return {'X': X, 'y': y}
