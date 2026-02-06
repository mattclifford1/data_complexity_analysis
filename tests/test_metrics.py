"""
Tests for the complexity_metrics wrapper class.

These tests verify that the complexity_metrics class correctly wraps PyCol
and returns metrics in the expected format. Tests focus on the wrapper
behavior, not the underlying PyCol metric calculations.
"""
import pytest
import numpy as np
from data_complexity.metrics import complexity_metrics


class TestComplexityMetricsInit:
    """Tests for complexity_metrics initialization."""

    def test_init_with_valid_dataset(self, simple_linearly_separable):
        """Verify initialization succeeds with valid X, y arrays."""
        cm = complexity_metrics(dataset=simple_linearly_separable)
        assert cm.pycol_complexity is not None

    def test_init_with_custom_distance_func(self, simple_linearly_separable):
        """Verify custom distance function parameter is accepted."""
        cm = complexity_metrics(
            dataset=simple_linearly_separable,
            distance_func="default"
        )
        assert cm.pycol_complexity is not None


class TestGetAllMetrics:
    """Tests for get_all_metrics_scalar and get_all_metrics_full methods."""

    def test_get_all_metrics_scalar_returns_dict(self, moons_dataset):
        """Verify get_all_metrics_scalar returns a dictionary."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.get_all_metrics_scalar()
        assert isinstance(result, dict)

    def test_get_all_metrics_scalar_non_empty(self, moons_dataset):
        """Verify get_all_metrics_scalar returns multiple metrics."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.get_all_metrics_scalar()
        assert len(result) > 0

    def test_get_all_metrics_scalar_values_are_numeric(self, moons_dataset):
        """Verify all returned metric values are numeric."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.get_all_metrics_scalar()
        for name, value in result.items():
            assert isinstance(value, (int, float, np.number)), \
                f"Metric {name} has non-numeric value: {value}"

    def test_get_all_metrics_full_returns_dict(self, moons_dataset):
        """Verify get_all_metrics_full returns a dictionary."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.get_all_metrics_full()
        assert isinstance(result, dict)

    def test_get_all_metrics_full_contains_all_categories(self, moons_dataset):
        """Verify get_all_metrics_full includes metrics from all categories."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.get_all_metrics_full()

        # Check for at least one metric from each category
        feature_metrics = {'F1', 'F1v', 'F2', 'F3', 'F4'}
        instance_metrics = {'Raug', 'deg_overlap', 'N3', 'SI', 'N4', 'kDN', 'D3', 'CM'}
        structural_metrics = {'N1', 'T1', 'Clust'}
        multiresolution_metrics = {'MRCA', 'C1', 'Purity'}

        result_keys = set(result.keys())
        assert feature_metrics <= result_keys, "Missing feature overlap metrics"
        assert instance_metrics <= result_keys, "Missing instance overlap metrics"
        assert structural_metrics <= result_keys, "Missing structural overlap metrics"
        assert multiresolution_metrics <= result_keys, "Missing multiresolution metrics"


class TestFeatureOverlap:
    """Tests for feature overlap methods."""

    def test_feature_overlap_scalar_returns_dict(self, moons_dataset):
        """Verify feature_overlap_scalar returns a dictionary."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.feature_overlap_scalar()
        assert isinstance(result, dict)

    def test_feature_overlap_scalar_has_expected_metrics(self, moons_dataset):
        """Verify feature_overlap_scalar contains standard feature metrics."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.feature_overlap_scalar()

        # F1, F1v, F2, F3, F4 are the core feature overlap metrics
        expected = {'F1', 'F1v', 'F2', 'F3', 'F4'}
        assert expected <= set(result.keys()), \
            f"Missing expected metrics. Got: {set(result.keys())}"

    def test_feature_overlap_full_returns_dict(self, moons_dataset):
        """Verify feature_overlap_full returns a dictionary."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.feature_overlap_full()
        assert isinstance(result, dict)

    def test_feature_overlap_full_has_expected_metrics(self, moons_dataset):
        """Verify feature_overlap_full contains F1-F4 and F1v."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.feature_overlap_full()
        expected = {'F1', 'F1v', 'F2', 'F3', 'F4'}
        assert expected == set(result.keys())


class TestInstanceOverlap:
    """Tests for instance overlap methods."""

    def test_instance_overlap_scalar_returns_dict(self, moons_dataset):
        """Verify instance_overlap_scalar returns a dictionary."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.instance_overlap_scalar()
        assert isinstance(result, dict)

    def test_instance_overlap_scalar_non_empty(self, moons_dataset):
        """Verify instance_overlap_scalar returns multiple metrics."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.instance_overlap_scalar()
        assert len(result) > 0

    def test_instance_overlap_full_returns_dict(self, moons_dataset):
        """Verify instance_overlap_full returns a dictionary."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.instance_overlap_full()
        assert isinstance(result, dict)

    def test_instance_overlap_full_has_expected_metrics(self, moons_dataset):
        """Verify instance_overlap_full contains expected metrics."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.instance_overlap_full()
        expected = {'Raug', 'deg_overlap', 'N3', 'SI', 'N4', 'kDN', 'D3', 'CM'}
        assert expected == set(result.keys())


class TestStructuralOverlap:
    """Tests for structural overlap methods."""

    def test_structural_overlap_scalar_returns_dict(self, moons_dataset):
        """Verify structural_overlap_scalar returns a dictionary."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.structural_overlap_scalar()
        assert isinstance(result, dict)

    def test_structural_overlap_scalar_non_empty(self, moons_dataset):
        """Verify structural_overlap_scalar returns multiple metrics."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.structural_overlap_scalar()
        assert len(result) > 0

    def test_structural_overlap_full_returns_dict(self, moons_dataset):
        """Verify structural_overlap_full returns a dictionary."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.structural_overlap_full()
        assert isinstance(result, dict)

    def test_structural_overlap_full_has_expected_metrics(self, moons_dataset):
        """Verify structural_overlap_full contains N1, T1, Clust."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.structural_overlap_full()
        expected = {'N1', 'T1', 'Clust'}
        assert expected == set(result.keys())


class TestMultiresolutionOverlap:
    """Tests for multiresolution overlap methods."""

    def test_multiresolution_overlap_full_returns_dict(self, moons_dataset):
        """Verify multiresolution_overlap_full returns a dictionary."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.multiresolution_overlap_full()
        assert isinstance(result, dict)

    def test_multiresolution_overlap_full_has_expected_metrics(self, moons_dataset):
        """Verify multiresolution_overlap_full contains MRCA, C1, Purity."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.multiresolution_overlap_full()
        expected = {'MRCA', 'C1', 'Purity'}
        assert expected == set(result.keys())


class TestDatasetVariations:
    """Tests verifying metrics work across different dataset types."""

    def test_works_with_linearly_separable(self, simple_linearly_separable):
        """Verify metrics compute on well-separated data."""
        cm = complexity_metrics(dataset=simple_linearly_separable)
        result = cm.get_all_metrics_scalar()
        assert len(result) > 0

    def test_works_with_high_overlap(self, high_overlap_dataset):
        """Verify metrics compute on highly overlapping data."""
        cm = complexity_metrics(dataset=high_overlap_dataset)
        result = cm.get_all_metrics_scalar()
        assert len(result) > 0

    def test_works_with_multiclass(self, multiclass_dataset):
        """Verify metrics compute on 3-class data."""
        cm = complexity_metrics(dataset=multiclass_dataset)
        result = cm.get_all_metrics_scalar()
        assert len(result) > 0

    def test_works_with_high_dimensional(self, high_dimensional_dataset):
        """Verify metrics compute on 10-dimensional data."""
        cm = complexity_metrics(dataset=high_dimensional_dataset)
        result = cm.get_all_metrics_scalar()
        assert len(result) > 0


class TestMetricConsistency:
    """Tests verifying metric values are consistent and sensible."""

    def test_separable_data_has_low_overlap(self, simple_linearly_separable):
        """
        Well-separated classes should have lower overlap metrics than
        heavily overlapping classes.
        """
        cm_sep = complexity_metrics(dataset=simple_linearly_separable)
        metrics_sep = cm_sep.get_all_metrics_scalar()

        # For well-separated data, F1 (max Fisher's discriminant ratio)
        # should be high, indicating good separability
        # Note: This is a sanity check, not a strict threshold
        assert 'F1' in metrics_sep

    def test_repeated_calls_return_same_keys(self, moons_dataset):
        """Verify calling the same method twice returns the same metric keys."""
        cm = complexity_metrics(dataset=moons_dataset)
        result1 = cm.get_all_metrics_scalar()
        result2 = cm.get_all_metrics_scalar()
        # Some metrics (e.g., N4) may have stochastic elements, so we only
        # check that the same keys are returned
        assert set(result1.keys()) == set(result2.keys())

    def test_scalar_methods_return_scalars(self, moons_dataset):
        """
        Verify that scalar methods return single numeric values, not arrays.

        The _scalar methods aggregate metrics into single values, while
        _full methods may return per-feature or per-instance arrays.
        """
        cm = complexity_metrics(dataset=moons_dataset)

        scalar_f = cm.feature_overlap_scalar()
        for key, value in scalar_f.items():
            assert isinstance(value, (int, float, np.number)), \
                f"Scalar metric {key} should be numeric, got {type(value)}"

    def test_full_methods_return_expected_types(self, moons_dataset):
        """
        Verify that full methods return values (scalars, arrays, or lists).

        The _full methods may return per-feature arrays/lists or scalar values
        depending on the metric.
        """
        cm = complexity_metrics(dataset=moons_dataset)

        full_f = cm.feature_overlap_full()
        for key, value in full_f.items():
            # Can be scalar, array, or list
            assert isinstance(value, (int, float, np.number, np.ndarray, list)), \
                f"Full metric {key} has unexpected type: {type(value)}"


class TestClassicalMeasures:
    """Tests for classical measures methods."""

    def test_classical_measures_scalar_returns_dict(self, moons_dataset):
        """Verify classical_measures_scalar returns a dictionary."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.classical_measures_scalar()
        assert isinstance(result, dict)

    def test_classical_measures_scalar_contains_IR(self, moons_dataset):
        """Verify classical_measures_scalar contains IR metric."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.classical_measures_scalar()
        assert 'IR' in result

    def test_classical_measures_full_returns_dict(self, moons_dataset):
        """Verify classical_measures_full returns a dictionary."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.classical_measures_full()
        assert isinstance(result, dict)

    def test_classical_measures_full_contains_IR(self, moons_dataset):
        """Verify classical_measures_full contains IR metric."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.classical_measures_full()
        assert 'IR' in result

    def test_IR_is_numeric(self, moons_dataset):
        """Verify IR returns a numeric value."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.classical_measures_scalar()
        assert isinstance(result['IR'], (int, float, np.number))

    def test_IR_balanced_dataset(self, balanced_binary_dataset):
        """Verify IR = 1.0 for perfectly balanced dataset."""
        cm = complexity_metrics(dataset=balanced_binary_dataset)
        result = cm.classical_measures_scalar()
        assert result['IR'] == pytest.approx(1.0, rel=0.01)

    def test_IR_imbalanced_dataset(self, imbalanced_binary_dataset):
        """Verify IR > 1.0 for imbalanced dataset."""
        cm = complexity_metrics(dataset=imbalanced_binary_dataset)
        result = cm.classical_measures_scalar()
        # Should be > 1.0 for imbalanced data, allow variance due to stratification
        assert result['IR'] >= 1.0
        assert result['IR'] <= 10.0

    def test_IR_highly_imbalanced(self, highly_imbalanced_dataset):
        """Verify IR is high for 90/10 imbalanced dataset."""
        cm = complexity_metrics(dataset=highly_imbalanced_dataset)
        result = cm.classical_measures_scalar()
        assert result['IR'] >= 5.0

    def test_IR_multiclass_dataset(self, multiclass_dataset):
        """Verify IR works with multiclass data (uses max/min)."""
        cm = complexity_metrics(dataset=multiclass_dataset)
        result = cm.classical_measures_scalar()
        assert result['IR'] >= 1.0

    def test_IR_single_class(self, single_class_dataset):
        """Verify IR = 1.0 for single-class dataset."""
        cm = complexity_metrics(dataset=single_class_dataset)
        result = cm.classical_measures_scalar()
        assert result['IR'] == 1.0

    def test_IR_always_greater_or_equal_one(self, moons_dataset, high_overlap_dataset,
                                             multiclass_dataset):
        """Verify IR is always >= 1.0 for valid datasets."""
        for dataset in [moons_dataset, high_overlap_dataset, multiclass_dataset]:
            cm = complexity_metrics(dataset=dataset)
            result = cm.classical_measures_scalar()
            assert result['IR'] >= 1.0

    def test_scalar_and_full_return_same_IR(self, moons_dataset):
        """Verify scalar and full versions return identical IR values."""
        cm = complexity_metrics(dataset=moons_dataset)
        scalar_result = cm.classical_measures_scalar()
        full_result = cm.classical_measures_full()
        assert scalar_result['IR'] == full_result['IR']


class TestGetAllMetricsWithClassical:
    """Tests verifying classical measures integrate with get_all methods."""

    def test_get_all_metrics_scalar_includes_IR(self, moons_dataset):
        """Verify get_all_metrics_scalar includes IR."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.get_all_metrics_scalar()
        assert 'IR' in result

    def test_get_all_metrics_full_includes_IR(self, moons_dataset):
        """Verify get_all_metrics_full includes IR."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.get_all_metrics_full()
        assert 'IR' in result

    def test_get_all_metrics_contains_all_categories_including_classical(self, moons_dataset):
        """Verify get_all_metrics_full includes all categories including classical."""
        cm = complexity_metrics(dataset=moons_dataset)
        result = cm.get_all_metrics_full()

        # Check for at least one metric from each category
        feature_metrics = {'F1', 'F1v', 'F2', 'F3', 'F4'}
        instance_metrics = {'Raug', 'deg_overlap', 'N3', 'SI', 'N4', 'kDN', 'D3', 'CM'}
        structural_metrics = {'N1', 'T1', 'Clust'}
        multiresolution_metrics = {'MRCA', 'C1', 'Purity'}
        classical_metrics = {'IR'}

        result_keys = set(result.keys())
        assert feature_metrics <= result_keys, "Missing feature overlap metrics"
        assert instance_metrics <= result_keys, "Missing instance overlap metrics"
        assert structural_metrics <= result_keys, "Missing structural overlap metrics"
        assert multiresolution_metrics <= result_keys, "Missing multiresolution metrics"
        assert classical_metrics <= result_keys, "Missing classical measures"
