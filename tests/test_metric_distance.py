"""
Tests for all 8 DistanceBetweenMetrics implementations.

Covers interface contracts, property values, and analytically known outputs
for simple inputs where the expected result can be computed by hand.
"""
import numpy as np
import pytest

from data_complexity.experiments.pipeline.metric_distance import (
    CosineSimilarity,
    DistanceBetweenMetrics,
    DistanceCorrelation,
    EuclideanDistance,
    KendallTau,
    ManhattanDistance,
    MutualInformation,
    PearsonCorrelation,
    SpearmanCorrelation,
    get_all_measures,
)


class TestDistanceMeasureInterface:
    """Abstract base class contract and factory function."""

    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            DistanceBetweenMetrics()  # type: ignore[abstract]

    def test_get_all_measures_returns_8_instances(self):
        assert len(get_all_measures()) == 8

    def test_get_all_measures_all_are_subclasses(self):
        for m in get_all_measures():
            assert isinstance(m, DistanceBetweenMetrics)

    def test_all_measures_have_unique_names(self):
        names = [m.name for m in get_all_measures()]
        assert len(names) == len(set(names))

    def test_all_measures_have_nonempty_display_names(self):
        for m in get_all_measures():
            assert isinstance(m.display_name, str)
            assert len(m.display_name) > 0


class TestMeasureProperties:
    """Exact name strings and signed flag for every measure."""

    @pytest.mark.parametrize("cls, expected_name", [
        (PearsonCorrelation,  "pearson_r"),
        (SpearmanCorrelation, "spearman_rho"),
        (KendallTau,          "kendall_tau"),
        (MutualInformation,   "mutual_information"),
        (EuclideanDistance,   "euclidean_distance"),
        (DistanceCorrelation, "distance_correlation"),
        (CosineSimilarity,    "cosine_similarity"),
        (ManhattanDistance,   "manhattan_distance"),
    ])
    def test_exact_name(self, cls, expected_name):
        assert cls().name == expected_name

    @pytest.mark.parametrize("cls", [
        PearsonCorrelation, SpearmanCorrelation, KendallTau, CosineSimilarity
    ])
    def test_signed_true(self, cls):
        assert cls().signed is True

    @pytest.mark.parametrize("cls", [
        MutualInformation, EuclideanDistance, DistanceCorrelation, ManhattanDistance
    ])
    def test_signed_false(self, cls):
        assert cls().signed is False


class TestPearsonCorrelation:
    def test_identical_arrays_give_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r, _ = PearsonCorrelation().compute(x, x)
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_negated_array_gives_minus_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r, _ = PearsonCorrelation().compute(x, -x)
        assert r == pytest.approx(-1.0, abs=1e-9)

    def test_returns_two_floats(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = PearsonCorrelation().compute(x, x)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_r_in_range_on_random_data(self):
        rng = np.random.default_rng(42)
        x, y = rng.random(50), rng.random(50)
        r, _ = PearsonCorrelation().compute(x, y)
        assert -1.0 <= r <= 1.0


class TestSpearmanCorrelation:
    def test_identical_arrays_give_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r, _ = SpearmanCorrelation().compute(x, x)
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_negated_array_gives_minus_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r, _ = SpearmanCorrelation().compute(x, -x)
        assert r == pytest.approx(-1.0, abs=1e-9)

    def test_monotonic_nonlinear_gives_one(self):
        x = np.arange(1.0, 11.0)   # sorted ascending
        r, _ = SpearmanCorrelation().compute(x, x ** 2)
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_returns_two_floats(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = SpearmanCorrelation().compute(x, x)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


class TestKendallTau:
    def test_identical_arrays_give_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tau, _ = KendallTau().compute(x, x)
        assert tau == pytest.approx(1.0, abs=1e-9)

    def test_negated_array_gives_minus_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tau, _ = KendallTau().compute(x, -x)
        assert tau == pytest.approx(-1.0, abs=1e-9)

    def test_returns_two_floats(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = KendallTau().compute(x, x)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)


class TestMutualInformation:
    def test_p_value_is_none(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, p = MutualInformation().compute(x, x)
        assert p is None

    def test_non_negative(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        val, _ = MutualInformation().compute(x, x)
        assert val >= 0.0

    def test_identical_higher_than_random_noise(self):
        rng = np.random.default_rng(42)
        x = rng.random(50)
        noise = rng.random(50)
        mi_self, _ = MutualInformation().compute(x, x)
        mi_noise, _ = MutualInformation().compute(x, noise)
        assert mi_self > mi_noise


class TestEuclideanDistance:
    def test_identical_arrays_give_zero(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        val, _ = EuclideanDistance().compute(x, x)
        assert val == pytest.approx(0.0, abs=1e-9)

    def test_p_value_is_none(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, p = EuclideanDistance().compute(x, x)
        assert p is None

    def test_non_negative_on_random_data(self):
        rng = np.random.default_rng(42)
        x, y = rng.random(50), rng.random(50)
        val, _ = EuclideanDistance().compute(x, y)
        assert val >= 0.0


class TestDistanceCorrelation:
    def test_identical_arrays_give_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        val, _ = DistanceCorrelation().compute(x, x)
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_p_value_is_none(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, p = DistanceCorrelation().compute(x, x)
        assert p is None

    def test_result_in_unit_interval_on_random_data(self):
        rng = np.random.default_rng(42)
        x, y = rng.random(20), rng.random(20)
        val, _ = DistanceCorrelation().compute(x, y)
        assert 0.0 <= val <= 1.0


class TestCosineSimilarity:
    def test_identical_arrays_give_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        val, _ = CosineSimilarity().compute(x, x)
        assert val == pytest.approx(1.0, abs=1e-9)

    def test_negated_array_gives_minus_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        val, _ = CosineSimilarity().compute(x, -x)
        assert val == pytest.approx(-1.0, abs=1e-9)

    def test_orthogonal_vectors_give_zero(self):
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        val, _ = CosineSimilarity().compute(x, y)
        assert val == pytest.approx(0.0, abs=1e-9)

    def test_p_value_is_none(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, p = CosineSimilarity().compute(x, x)
        assert p is None


class TestManhattanDistance:
    def test_identical_arrays_give_zero(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        val, _ = ManhattanDistance().compute(x, x)
        assert val == pytest.approx(0.0, abs=1e-9)

    def test_p_value_is_none(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, p = ManhattanDistance().compute(x, x)
        assert p is None

    def test_non_negative_on_random_data(self):
        rng = np.random.default_rng(42)
        x, y = rng.random(50), rng.random(50)
        val, _ = ManhattanDistance().compute(x, y)
        assert val >= 0.0
