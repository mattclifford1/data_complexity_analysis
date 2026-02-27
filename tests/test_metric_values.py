"""
Tests that complexity metric values fall within expected numerical ranges
and that "low complexity" datasets score lower/higher than "high complexity"
datasets in the correct direction for each metric.

Fixtures are module-scoped so that the expensive PyCol computations run
only once per test session rather than once per test function.
"""
import numpy as np
import pytest
from sklearn.datasets import make_blobs

from data_complexity.data_metrics.metrics import ComplexityMetrics


# ---------------------------------------------------------------------------
# Module-scoped dataset fixtures — same parameters as conftest
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def simple_data():
    """Well-separated blobs — low complexity."""
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=0.5, random_state=42)
    return {"X": X, "y": y}


@pytest.fixture(scope="module")
def complex_data():
    """Heavily overlapping blobs — high complexity."""
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=3.0, random_state=42)
    return {"X": X, "y": y}


# ---------------------------------------------------------------------------
# Module-scoped metric fixtures — computed once, shared across all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def simple_cm(simple_data):
    return ComplexityMetrics(dataset=simple_data)


@pytest.fixture(scope="module")
def complex_cm(complex_data):
    return ComplexityMetrics(dataset=complex_data)


@pytest.fixture(scope="module")
def simple_feature(simple_cm):
    return simple_cm.feature_overlap_scalar()


@pytest.fixture(scope="module")
def complex_feature(complex_cm):
    return complex_cm.feature_overlap_scalar()


@pytest.fixture(scope="module")
def simple_instance(simple_cm):
    return simple_cm.instance_overlap_scalar()


@pytest.fixture(scope="module")
def complex_instance(complex_cm):
    return complex_cm.instance_overlap_scalar()


@pytest.fixture(scope="module")
def simple_structural(simple_cm):
    return simple_cm.structural_overlap_scalar()


@pytest.fixture(scope="module")
def simple_multiresolution(simple_cm):
    return simple_cm.multiresolution_overlap_scalar()


@pytest.fixture(scope="module")
def simple_distributional(simple_cm):
    return simple_cm.distributional_measures_scalar()


@pytest.fixture(scope="module")
def complex_distributional(complex_cm):
    return complex_cm.distributional_measures_scalar()


# ---------------------------------------------------------------------------
# Feature overlap
# ---------------------------------------------------------------------------

class TestFeatureOverlapRanges:
    def test_f2_in_unit_interval(self, simple_feature):
        assert 0.0 <= simple_feature["F2"] <= 1.0

    def test_f3_in_unit_interval(self, simple_feature):
        assert 0.0 <= simple_feature["F3"] <= 1.0

    def test_f4_in_unit_interval(self, simple_feature):
        assert 0.0 <= simple_feature["F4"] <= 1.0

    def test_f1_non_negative(self, simple_feature):
        assert simple_feature["F1"] >= 0.0


class TestFeatureOverlapOrdering:
    def test_f1_lower_for_separable(self, simple_feature, complex_feature):
        """F1 in PyCol is a complexity measure: lower = less overlap = more separable."""
        assert simple_feature["F1"] < complex_feature["F1"]

    def test_f2_lower_for_separable(self, simple_feature, complex_feature):
        """Smaller overlapping region → lower F2 for separable data."""
        assert simple_feature["F2"] < complex_feature["F2"]


# ---------------------------------------------------------------------------
# Instance overlap
# ---------------------------------------------------------------------------

class TestInstanceOverlapRanges:
    def test_n3_in_unit_interval(self, simple_instance):
        assert 0.0 <= simple_instance["N3"] <= 1.0

    def test_kdn_in_unit_interval(self, simple_instance):
        assert 0.0 <= simple_instance["kDN"] <= 1.0

    def test_borderline_in_unit_interval(self, simple_instance):
        assert 0.0 <= simple_instance["Borderline"] <= 1.0


class TestInstanceOverlapOrdering:
    def test_n3_lower_for_separable(self, simple_instance, complex_instance):
        """1-NN makes fewer errors on well-separated data."""
        assert simple_instance["N3"] < complex_instance["N3"]

    def test_kdn_lower_for_separable(self, simple_instance, complex_instance):
        """Fewer disagreeing neighbours in well-separated data."""
        assert simple_instance["kDN"] < complex_instance["kDN"]


# ---------------------------------------------------------------------------
# Structural overlap
# ---------------------------------------------------------------------------

class TestStructuralOverlapRanges:
    def test_n1_in_unit_interval(self, simple_structural):
        assert 0.0 <= simple_structural["N1"] <= 1.0

    def test_t1_positive_and_at_most_one(self, simple_structural):
        assert 0.0 < simple_structural["T1"] <= 1.0


# ---------------------------------------------------------------------------
# Multiresolution overlap
# ---------------------------------------------------------------------------

class TestMultiresolutionRanges:
    def test_c1_in_unit_interval(self, simple_multiresolution):
        assert 0.0 <= simple_multiresolution["C1"] <= 1.0

    def test_purity_in_unit_interval(self, simple_multiresolution):
        assert 0.0 <= simple_multiresolution["Purity"] <= 1.0


# ---------------------------------------------------------------------------
# Distributional measures
# ---------------------------------------------------------------------------

class TestDistributionalRanges:
    def test_silhouette_in_range(self, simple_distributional):
        assert -1.0 <= simple_distributional["Silhouette"] <= 1.0

    def test_bhattacharyya_non_negative(self, simple_distributional):
        assert simple_distributional["Bhattacharyya"] >= 0.0

    def test_wasserstein_non_negative(self, simple_distributional):
        assert simple_distributional["Wasserstein"] >= 0.0

    def test_svm_svr_in_unit_interval(self, simple_distributional):
        assert 0.0 <= simple_distributional["SVM_SVR"] <= 1.0

    def test_twonn_id_at_least_one(self, simple_distributional):
        assert simple_distributional["TwoNN_ID"] >= 1.0


class TestDistributionalOrdering:
    def test_silhouette_higher_for_separable(self, simple_distributional, complex_distributional):
        """Better-separated clusters → higher silhouette score."""
        assert simple_distributional["Silhouette"] > complex_distributional["Silhouette"]

    def test_svm_svr_lower_for_separable(self, simple_distributional, complex_distributional):
        """Clean boundary needs fewer support vectors."""
        assert simple_distributional["SVM_SVR"] < complex_distributional["SVM_SVR"]

    def test_wasserstein_higher_for_separable(self, simple_distributional, complex_distributional):
        """More separation between class distributions → larger Wasserstein distance."""
        assert simple_distributional["Wasserstein"] > complex_distributional["Wasserstein"]

    def test_bhattacharyya_lower_for_separable(self, simple_distributional, complex_distributional):
        """Less distributional overlap → lower Bhattacharyya coefficient."""
        assert simple_distributional["Bhattacharyya"] < complex_distributional["Bhattacharyya"]
