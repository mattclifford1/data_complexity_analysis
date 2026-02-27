"""
Tests for ComplexityCollection and DatasetEntry.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data_complexity.examples.legacy_complexity_over_datasets import (
    ComplexityCollection,
    DatasetEntry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_data():
    """Simple two-class binary dataset."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 2))
    y = (X[:, 0] > 0).astype(int)
    return {"X": X, "y": y}


@pytest.fixture
def mock_metrics():
    """Fixed complexity metrics dict used across tests."""
    return {"F1": 0.5, "F2": 0.3, "N3": 0.7, "T1": 0.2}


def _patch_complexity(metrics_dict):
    """Return a context manager that patches ComplexityMetrics to return metrics_dict."""
    return patch(
        "data_complexity.examples.legacy_complexity_over_datasets.ComplexityMetrics",
        return_value=MagicMock(
            **{"get_all_metrics_scalar.return_value": metrics_dict}
        ),
    )


# ---------------------------------------------------------------------------
# DatasetEntry
# ---------------------------------------------------------------------------


class TestDatasetEntry:
    def test_default_fields(self):
        entry = DatasetEntry(name="test")
        assert entry.name == "test"
        assert entry.data is None
        assert entry.dataset_type is None
        assert entry.dataset_params == {}

    def test_with_preloaded_data(self, simple_data):
        entry = DatasetEntry(name="my_dataset", data=simple_data)
        assert entry.data is simple_data
        assert entry.dataset_type is None

    def test_with_synthetic_params(self):
        entry = DatasetEntry(
            name="gauss",
            dataset_type="Gaussian",
            dataset_params={"cov_scale": 1.0},
        )
        assert entry.data is None
        assert entry.dataset_type == "Gaussian"
        assert entry.dataset_params == {"cov_scale": 1.0}


# ---------------------------------------------------------------------------
# add_* methods
# ---------------------------------------------------------------------------


class TestAddMethods:
    def test_add_dataset_returns_self(self, simple_data):
        coll = ComplexityCollection()
        assert coll.add_dataset("iris", simple_data) is coll

    def test_add_dataset_stores_entry(self, simple_data):
        coll = ComplexityCollection()
        coll.add_dataset("iris", simple_data)
        assert len(coll._entries) == 1
        assert coll._entries[0].name == "iris"
        assert coll._entries[0].data is simple_data

    def test_add_synthetic_returns_self(self):
        coll = ComplexityCollection()
        assert coll.add_synthetic("gauss", "Gaussian") is coll

    def test_add_synthetic_stores_entry(self):
        coll = ComplexityCollection()
        coll.add_synthetic("gauss", "Gaussian", {"cov_scale": 2.0})
        assert len(coll._entries) == 1
        e = coll._entries[0]
        assert e.name == "gauss"
        assert e.dataset_type == "Gaussian"
        assert e.dataset_params == {"cov_scale": 2.0}

    def test_add_synthetic_no_params(self):
        coll = ComplexityCollection()
        coll.add_synthetic("gauss", "Gaussian")
        assert coll._entries[0].dataset_params == {}

    def test_add_synthetic_sweep_correct_count(self):
        coll = ComplexityCollection()
        coll.add_synthetic_sweep("moons", "Moons", {}, "noise", [0.1, 0.2, 0.3])
        assert len(coll._entries) == 3

    def test_add_synthetic_sweep_default_names(self):
        coll = ComplexityCollection()
        coll.add_synthetic_sweep("moons", "Moons", {}, "noise", [0.1, 0.2, 0.3])
        assert coll._entries[0].name == "moons_noise=0.1"
        assert coll._entries[1].name == "moons_noise=0.2"
        assert coll._entries[2].name == "moons_noise=0.3"

    def test_add_synthetic_sweep_merges_fixed_params(self):
        coll = ComplexityCollection()
        coll.add_synthetic_sweep(
            "m", "Moons", {"n_samples": 200}, "noise", [0.1, 0.2]
        )
        assert coll._entries[0].dataset_params == {"n_samples": 200, "noise": 0.1}
        assert coll._entries[1].dataset_params == {"n_samples": 200, "noise": 0.2}

    def test_add_synthetic_sweep_custom_name_format(self):
        coll = ComplexityCollection()
        coll.add_synthetic_sweep(
            "m", "Moons", {}, "noise", [0.1, 0.2], name_format="{base}_{value}"
        )
        assert coll._entries[0].name == "m_0.1"
        assert coll._entries[1].name == "m_0.2"

    def test_fluent_chaining(self, simple_data):
        coll = (
            ComplexityCollection()
            .add_dataset("ds1", simple_data)
            .add_synthetic("syn1", "Gaussian")
            .add_synthetic_sweep("moons", "Moons", {}, "noise", [0.1, 0.2])
        )
        assert len(coll._entries) == 4


# ---------------------------------------------------------------------------
# compute()
# ---------------------------------------------------------------------------


class TestCompute:
    def test_returns_dataframe(self, simple_data, mock_metrics):
        coll = ComplexityCollection(seeds=2)
        coll.add_dataset("ds1", simple_data)
        coll.add_dataset("ds2", simple_data)

        with _patch_complexity(mock_metrics):
            df = coll.compute()

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, len(mock_metrics))

    def test_index_is_dataset_names(self, simple_data, mock_metrics):
        coll = ComplexityCollection(seeds=1)
        coll.add_dataset("iris", simple_data)
        coll.add_dataset("wine", simple_data)

        with _patch_complexity(mock_metrics):
            df = coll.compute()

        assert list(df.index) == ["iris", "wine"]

    def test_columns_are_metric_names(self, simple_data, mock_metrics):
        coll = ComplexityCollection(seeds=1)
        coll.add_dataset("ds1", simple_data)

        with _patch_complexity(mock_metrics):
            df = coll.compute()

        assert set(df.columns) == set(mock_metrics.keys())

    def test_averages_across_seeds(self, simple_data):
        """Metrics must be averaged across seeds, not just taken from the last."""
        coll = ComplexityCollection(seeds=3)
        coll.add_dataset("ds1", simple_data)

        call_count = [0]

        def make_mock(*args, **kwargs):
            mock = MagicMock()
            mock.get_all_metrics_scalar.return_value = {"F1": 0.1 * (call_count[0] + 1)}
            call_count[0] += 1
            return mock

        with patch(
            "data_complexity.examples.legacy_complexity_over_datasets.ComplexityMetrics",
            side_effect=make_mock,
        ):
            df = coll.compute()

        # seeds return F1 = 0.1, 0.2, 0.3 → mean = 0.2
        assert abs(df.loc["ds1", "F1"] - 0.2) < 1e-10

    def test_synthetic_uses_get_dataset(self, mock_metrics):
        """Synthetic entries should call get_dataset and get_train_test_split."""
        coll = ComplexityCollection(seeds=2)
        coll.add_synthetic("gauss", "Gaussian", {"class_separation": 3.0})

        mock_train = {"X": np.zeros((10, 2)), "y": np.zeros(10, dtype=int)}
        mock_dataset = MagicMock()
        mock_dataset.get_train_test_split.return_value = (mock_train, {})

        with _patch_complexity(mock_metrics), patch(
            "data_loaders.get_dataset", return_value=mock_dataset
        ):
            df = coll.compute()

        assert "gauss" in df.index
        assert mock_dataset.get_train_test_split.call_count == 2  # one per seed

    def test_preloaded_does_not_call_get_dataset(self, simple_data, mock_metrics):
        """Pre-loaded entries must NOT call get_dataset."""
        coll = ComplexityCollection(seeds=1)
        coll.add_dataset("ds1", simple_data)

        with _patch_complexity(mock_metrics), patch(
            "data_loaders.get_dataset"
        ) as mock_get:
            coll.compute()

        mock_get.assert_not_called()

    def test_synthetic_sweep_rows(self, mock_metrics):
        coll = ComplexityCollection(seeds=1)
        coll.add_synthetic_sweep("m", "Moons", {}, "noise", [0.1, 0.2, 0.3])

        mock_train = {"X": np.zeros((10, 2)), "y": np.zeros(10, dtype=int)}
        mock_dataset = MagicMock()
        mock_dataset.get_train_test_split.return_value = (mock_train, {})

        with _patch_complexity(mock_metrics), patch(
            "data_loaders.get_dataset", return_value=mock_dataset
        ):
            df = coll.compute()

        assert df.shape[0] == 3


# ---------------------------------------------------------------------------
# compute_correlations()
# ---------------------------------------------------------------------------


class TestComputeCorrelations:
    def _make_varying_metrics(self, n_datasets: int) -> list[dict]:
        """Return a list of metrics dicts that vary across datasets."""
        return [
            {"F1": 0.1 * i, "F2": 0.2 * i, "N3": 0.3 * (n_datasets - i)}
            for i in range(1, n_datasets + 1)
        ]

    def test_returns_square_matrix(self, simple_data):
        coll = ComplexityCollection(seeds=1)
        for i in range(4):
            coll.add_dataset(f"ds{i}", simple_data)

        metric_seq = self._make_varying_metrics(4)
        call_count = [0]

        def make_mock(*args, **kwargs):
            m = MagicMock()
            m.get_all_metrics_scalar.return_value = metric_seq[call_count[0] % 4]
            call_count[0] += 1
            return m

        with patch(
            "data_complexity.examples.legacy_complexity_over_datasets.ComplexityMetrics",
            side_effect=make_mock,
        ):
            corr = coll.compute_correlations()

        n = len(metric_seq[0])
        assert corr.shape == (n, n)

    def test_symmetric(self, simple_data):
        coll = ComplexityCollection(seeds=1)
        for i in range(5):
            coll.add_dataset(f"ds{i}", simple_data)

        metric_seq = self._make_varying_metrics(5)
        call_count = [0]

        def make_mock(*args, **kwargs):
            m = MagicMock()
            m.get_all_metrics_scalar.return_value = metric_seq[call_count[0] % 5]
            call_count[0] += 1
            return m

        with patch(
            "data_complexity.examples.legacy_complexity_over_datasets.ComplexityMetrics",
            side_effect=make_mock,
        ):
            corr = coll.compute_correlations()

        pd.testing.assert_frame_equal(corr, corr.T)

    def test_drops_constant_columns(self, simple_data):
        """Columns with zero variance must be excluded from the correlation matrix."""
        coll = ComplexityCollection(seeds=1)
        for _ in range(3):
            coll.add_dataset("ds", simple_data)

        with _patch_complexity({"F1": 0.5, "const": 1.0}):
            coll.compute()

        # Inject a known-constant column directly
        coll._metrics_df = pd.DataFrame(
            {"F1": [0.1, 0.2, 0.3], "const": [1.0, 1.0, 1.0]},
            index=["a", "b", "c"],
        )
        coll._correlations_df = None

        corr = coll.compute_correlations()
        assert "const" not in corr.columns

    def test_triggers_compute_if_needed(self, simple_data, mock_metrics):
        """compute_correlations() must call compute() when _metrics_df is None."""
        coll = ComplexityCollection(seeds=1)
        coll.add_dataset("ds1", simple_data)
        coll.add_dataset("ds2", simple_data)

        with _patch_complexity(mock_metrics):
            corr = coll.compute_correlations()

        assert coll._metrics_df is not None
        assert corr is not None


# ---------------------------------------------------------------------------
# plot_heatmap()
# ---------------------------------------------------------------------------


class TestPlotHeatmap:
    def test_returns_figure(self, simple_data, mock_metrics):
        coll = ComplexityCollection(seeds=1)
        for i in range(3):
            coll.add_dataset(f"ds{i}", simple_data)

        with _patch_complexity(mock_metrics):
            fig = coll.plot_heatmap()

        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_accepts_custom_title(self, simple_data, mock_metrics):
        coll = ComplexityCollection(seeds=1)
        for i in range(3):
            coll.add_dataset(f"ds{i}", simple_data)

        with _patch_complexity(mock_metrics), patch(
            "data_complexity.examples.legacy_complexity_over_datasets"
            ".plot_pairwise_heatmap"
        ) as mock_plot:
            mock_plot.return_value = MagicMock()
            coll.plot_heatmap(title="My Custom Title")

        mock_plot.assert_called_once()
        _, kwargs = mock_plot.call_args
        assert kwargs.get("title") == "My Custom Title" or mock_plot.call_args[0][1] == "My Custom Title"


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------


class TestSave:
    def test_writes_expected_files(self, tmp_path, simple_data, mock_metrics):
        coll = ComplexityCollection(seeds=1)
        for i in range(3):
            coll.add_dataset(f"ds{i}", simple_data)

        with _patch_complexity(mock_metrics):
            coll.save(tmp_path)

        assert (tmp_path / "data" / "complexity_metrics.csv").exists()
        assert (tmp_path / "data" / "complexity_correlations.csv").exists()
        assert (tmp_path / "plots" / "complexity_correlations_heatmap.png").exists()

    def test_creates_directory(self, tmp_path, simple_data, mock_metrics):
        save_dir = tmp_path / "nested" / "dir"
        coll = ComplexityCollection(seeds=1)
        for i in range(3):
            coll.add_dataset(f"ds{i}", simple_data)

        with _patch_complexity(mock_metrics):
            coll.save(save_dir)

        assert save_dir.is_dir()

    def test_metrics_csv_correct_shape(self, tmp_path, simple_data, mock_metrics):
        coll = ComplexityCollection(seeds=1)
        coll.add_dataset("ds1", simple_data)
        coll.add_dataset("ds2", simple_data)

        with _patch_complexity(mock_metrics):
            coll.save(tmp_path)

        df = pd.read_csv(tmp_path / "data" / "complexity_metrics.csv", index_col=0)
        assert df.shape == (2, len(mock_metrics))
        assert list(df.index) == ["ds1", "ds2"]

    def test_correlations_csv_is_square(self, tmp_path, simple_data, mock_metrics):
        coll = ComplexityCollection(seeds=1)
        for i in range(3):
            coll.add_dataset(f"ds{i}", simple_data)

        with _patch_complexity(mock_metrics):
            coll.save(tmp_path)

        corr = pd.read_csv(tmp_path / "data" / "complexity_correlations.csv", index_col=0)
        assert corr.shape[0] == corr.shape[1]
