"""Tests for GroupedExperiment and related helpers."""
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data_complexity.experiments.pipeline import (
    DatasetSpec,
    ExperimentConfig,
    ParameterSpec,
    PearsonCorrelation,
    PlotType,
    RunMode,
    datasets_from_sweep,
)
from data_complexity.experiments.pipeline.grouped_experiment import (
    GroupedExperiment,
    GroupedExperimentConfig,
    _clone_config,
    mean_matrices,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_matrix(values: list[list[float]], labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(values, index=labels, columns=labels)


def _make_base_config(**kwargs) -> ExperimentConfig:
    defaults = dict(
        datasets=[],
        x_label="noise",
        cv_folds=2,
        run_mode=RunMode.COMPLEXITY_ONLY,
        plots=[],
        pairwise_distance_measures=[PearsonCorrelation()],
        ml_metrics=["accuracy"],
        name="base",
    )
    defaults.update(kwargs)
    return ExperimentConfig(**defaults)


def _two_spec_sweep(dataset_type: str, param_name: str, values: list) -> list[DatasetSpec]:
    return datasets_from_sweep(
        DatasetSpec(dataset_type, {}),
        ParameterSpec(param_name, values, label_format="{value}"),
    )


# ---------------------------------------------------------------------------
# mean_matrices
# ---------------------------------------------------------------------------

class TestMeanMatrices:

    def test_basic_mean(self):
        m1 = _make_matrix([[1.0, 0.8], [0.8, 1.0]], ["A", "B"])
        m2 = _make_matrix([[1.0, 0.4], [0.4, 1.0]], ["A", "B"])
        result = mean_matrices([m1, m2])
        assert np.isclose(result.loc["A", "B"], 0.6)
        assert np.isclose(result.loc["B", "A"], 0.6)
        assert np.isclose(result.loc["A", "A"], 1.0)

    def test_single_matrix_unchanged(self):
        m = _make_matrix([[1.0, 0.5], [0.5, 1.0]], ["A", "B"])
        result = mean_matrices([m])
        pd.testing.assert_frame_equal(result, m)

    def test_three_matrices(self):
        matrices = [
            _make_matrix([[1.0, v], [v, 1.0]], ["X", "Y"])
            for v in [0.1, 0.4, 0.7]
        ]
        result = mean_matrices(matrices)
        assert np.isclose(result.loc["X", "Y"], 0.4)

    def test_shared_index_only(self):
        m1 = _make_matrix(
            [[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]],
            ["A", "B", "C"],
        )
        m2 = _make_matrix([[1.0, 0.6], [0.6, 1.0]], ["A", "B"])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = mean_matrices([m1, m2])

        # "C" missing from m2 → warning issued
        assert any("C" in str(w.message) for w in caught)
        # Only A and B in result
        assert list(result.columns) == ["A", "B"]
        assert list(result.index) == ["A", "B"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            mean_matrices([])

    def test_order_independent(self):
        m1 = _make_matrix([[1.0, 0.2], [0.2, 1.0]], ["A", "B"])
        m2 = _make_matrix([[1.0, 0.8], [0.8, 1.0]], ["A", "B"])
        r1 = mean_matrices([m1, m2])
        r2 = mean_matrices([m2, m1])
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# _clone_config
# ---------------------------------------------------------------------------

class TestCloneConfig:

    def test_overrides_datasets_name_save_dir(self, tmp_path):
        base = _make_base_config()
        specs = _two_spec_sweep("Moons", "moons_noise", [0.1, 0.2])
        cloned = _clone_config(base, specs, name="cloned", save_dir=tmp_path / "cloned")

        assert cloned.datasets == specs
        assert cloned.name == "cloned"
        assert cloned.save_dir == tmp_path / "cloned"

    def test_preserves_other_fields(self, tmp_path):
        base = _make_base_config(cv_folds=7)
        cloned = _clone_config(base, [], name="x", save_dir=tmp_path)
        assert cloned.cv_folds == 7
        assert cloned.run_mode == RunMode.COMPLEXITY_ONLY

    def test_does_not_mutate_base(self, tmp_path):
        base = _make_base_config()
        original_name = base.name
        _clone_config(base, [], name="new_name", save_dir=tmp_path)
        assert base.name == original_name


# ---------------------------------------------------------------------------
# GroupedExperimentConfig
# ---------------------------------------------------------------------------

class TestGroupedExperimentConfig:

    def test_defaults(self):
        base = _make_base_config()
        cfg = GroupedExperimentConfig(
            dataset_groups={"g1": []},
            base_config=base,
        )
        assert cfg.name == "grouped_experiment"
        assert cfg.save_dir is not None
        assert cfg.aggregation_fn is mean_matrices

    def test_explicit_name_and_save_dir(self, tmp_path):
        base = _make_base_config()
        cfg = GroupedExperimentConfig(
            dataset_groups={"g1": []},
            base_config=base,
            name="my_grouped",
            save_dir=tmp_path,
        )
        assert cfg.name == "my_grouped"
        assert cfg.save_dir == tmp_path

    def test_save_dir_auto_contains_name(self):
        base = _make_base_config()
        cfg = GroupedExperimentConfig(
            dataset_groups={},
            base_config=base,
            name="grouped_test",
        )
        assert "grouped_test" in str(cfg.save_dir)

    def test_custom_aggregation_fn(self):
        custom_fn = MagicMock(return_value=pd.DataFrame())
        base = _make_base_config()
        cfg = GroupedExperimentConfig(
            dataset_groups={},
            base_config=base,
            aggregation_fn=custom_fn,
        )
        assert cfg.aggregation_fn is custom_fn


# ---------------------------------------------------------------------------
# GroupedExperiment.run() — with mocked child experiments
# ---------------------------------------------------------------------------

def _mock_experiment_for_group(group_matrices: dict[str, pd.DataFrame]) -> MagicMock:
    """Return a mock Experiment that returns pre-set pairwise distance matrices."""
    exp = MagicMock()
    exp.compute_complexity_pairwise_distances.return_value = group_matrices
    exp.results = MagicMock()
    return exp


class TestGroupedExperimentRun:

    @pytest.fixture
    def base_config(self):
        return _make_base_config()

    def test_run_creates_one_experiment_per_group(self, base_config, tmp_path):
        groups = {
            "g1": _two_spec_sweep("Moons", "moons_noise", [0.1, 0.2]),
            "g2": _two_spec_sweep("Circles", "circles_noise", [0.1, 0.2]),
        }
        cfg = GroupedExperimentConfig(
            dataset_groups=groups,
            base_config=base_config,
            name="test",
            save_dir=tmp_path,
        )
        grouped = GroupedExperiment(cfg)

        with patch(
            "data_complexity.experiments.pipeline.grouped_experiment.Experiment"
        ) as MockExp:
            mock_instance = MagicMock()
            mock_instance.run.return_value = MagicMock()
            MockExp.return_value = mock_instance

            grouped.run(verbose=False)

        assert MockExp.call_count == 2
        assert set(grouped.experiments.keys()) == {"g1", "g2"}

    def test_run_uses_group_scoped_save_dir(self, base_config, tmp_path):
        """Each child experiment should get save_dir = save_dir/groups/{group_name}."""
        groups = {"alpha": _two_spec_sweep("Moons", "moons_noise", [0.1, 0.2])}
        cfg = GroupedExperimentConfig(
            dataset_groups=groups,
            base_config=base_config,
            name="test",
            save_dir=tmp_path,
        )
        grouped = GroupedExperiment(cfg)

        captured_configs = []

        with patch(
            "data_complexity.experiments.pipeline.grouped_experiment.Experiment"
        ) as MockExp:
            def capture(config):
                captured_configs.append(config)
                m = MagicMock()
                m.run.return_value = MagicMock()
                return m

            MockExp.side_effect = capture
            grouped.run(verbose=False)

        assert len(captured_configs) == 1
        assert captured_configs[0].save_dir == tmp_path / "groups" / "alpha"

    def test_run_uses_group_scoped_name(self, base_config, tmp_path):
        groups = {"beta": _two_spec_sweep("Moons", "moons_noise", [0.1])}
        cfg = GroupedExperimentConfig(
            dataset_groups=groups,
            base_config=base_config,
            name="mygroup",
            save_dir=tmp_path,
        )
        grouped = GroupedExperiment(cfg)

        captured = []
        with patch(
            "data_complexity.experiments.pipeline.grouped_experiment.Experiment"
        ) as MockExp:
            def cap(config):
                captured.append(config)
                m = MagicMock()
                m.run.return_value = MagicMock()
                return m
            MockExp.side_effect = cap
            grouped.run(verbose=False)

        assert captured[0].name == "mygroup__beta"


# ---------------------------------------------------------------------------
# GroupedExperiment.compute_averaged_pairwise_distances()
# ---------------------------------------------------------------------------

class TestComputeAveragedPairwiseDistances:

    def _make_grouped(
        self,
        group_matrices: dict[str, dict[str, pd.DataFrame]],
        tmp_path: Path,
    ) -> GroupedExperiment:
        base = _make_base_config()
        cfg = GroupedExperimentConfig(
            dataset_groups={g: [] for g in group_matrices},
            base_config=base,
            name="avg_test",
            save_dir=tmp_path,
        )
        grouped = GroupedExperiment(cfg)
        # Inject pre-built mock experiments
        for group_name, matrices in group_matrices.items():
            grouped.experiments[group_name] = _mock_experiment_for_group(matrices)
        return grouped

    def test_basic_average(self, tmp_path):
        m1 = _make_matrix([[1.0, 0.2], [0.2, 1.0]], ["F1", "N3"])
        m2 = _make_matrix([[1.0, 0.8], [0.8, 1.0]], ["F1", "N3"])
        grouped = self._make_grouped(
            {"g1": {"pearson_r": m1}, "g2": {"pearson_r": m2}},
            tmp_path,
        )

        result = grouped.compute_averaged_pairwise_distances()

        assert "pearson_r" in result
        assert np.isclose(result["pearson_r"].loc["F1", "N3"], 0.5)

    def test_result_stored_on_instance(self, tmp_path):
        m = _make_matrix([[1.0, 0.5], [0.5, 1.0]], ["A", "B"])
        grouped = self._make_grouped({"g1": {"pearson_r": m}}, tmp_path)
        grouped.compute_averaged_pairwise_distances()
        assert grouped.averaged_pairwise_distances == grouped.averaged_pairwise_distances

    def test_per_group_stored(self, tmp_path):
        m1 = _make_matrix([[1.0, 0.3], [0.3, 1.0]], ["A", "B"])
        m2 = _make_matrix([[1.0, 0.7], [0.7, 1.0]], ["A", "B"])
        grouped = self._make_grouped(
            {"g1": {"pearson_r": m1}, "g2": {"pearson_r": m2}},
            tmp_path,
        )
        grouped.compute_averaged_pairwise_distances()

        assert "g1" in grouped.per_group_pairwise_distances
        assert "g2" in grouped.per_group_pairwise_distances

    def test_multiple_measures(self, tmp_path):
        from data_complexity.experiments.pipeline import SpearmanCorrelation

        m_pearson = _make_matrix([[1.0, 0.5], [0.5, 1.0]], ["A", "B"])
        m_spearman = _make_matrix([[1.0, 0.3], [0.3, 1.0]], ["A", "B"])
        grouped = self._make_grouped(
            {
                "g1": {"pearson_r": m_pearson, "spearman_rho": m_spearman},
                "g2": {"pearson_r": m_pearson, "spearman_rho": m_spearman},
            },
            tmp_path,
        )
        result = grouped.compute_averaged_pairwise_distances()

        assert "pearson_r" in result
        assert "spearman_rho" in result

    def test_raises_if_not_run(self, tmp_path):
        base = _make_base_config()
        cfg = GroupedExperimentConfig(
            dataset_groups={"g": []},
            base_config=base,
            save_dir=tmp_path,
        )
        grouped = GroupedExperiment(cfg)
        with pytest.raises(RuntimeError, match="run()"):
            grouped.compute_averaged_pairwise_distances()

    def test_warns_on_missing_measure_in_group(self, tmp_path):
        m = _make_matrix([[1.0, 0.5], [0.5, 1.0]], ["A", "B"])
        # g1 has pearson_r, g2 does not
        grouped = self._make_grouped(
            {"g1": {"pearson_r": m}, "g2": {}},
            tmp_path,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            grouped.compute_averaged_pairwise_distances()

        assert any("pearson_r" in str(w.message) for w in caught)

    def test_custom_aggregation_fn(self, tmp_path):
        """Custom aggregation_fn receives per-group matrices and its result is stored."""
        m1 = _make_matrix([[1.0, 0.2], [0.2, 1.0]], ["A", "B"])
        m2 = _make_matrix([[1.0, 0.6], [0.6, 1.0]], ["A", "B"])
        expected = _make_matrix([[1.0, 0.99], [0.99, 1.0]], ["A", "B"])

        custom_fn = MagicMock(return_value=expected)
        base = _make_base_config()
        cfg = GroupedExperimentConfig(
            dataset_groups={"g1": [], "g2": []},
            base_config=base,
            aggregation_fn=custom_fn,
            name="custom",
            save_dir=tmp_path,
        )
        grouped = GroupedExperiment(cfg)
        grouped.experiments["g1"] = _mock_experiment_for_group({"pearson_r": m1})
        grouped.experiments["g2"] = _mock_experiment_for_group({"pearson_r": m2})

        result = grouped.compute_averaged_pairwise_distances()

        custom_fn.assert_called_once()
        pd.testing.assert_frame_equal(result["pearson_r"], expected)


# ---------------------------------------------------------------------------
# GroupedExperiment.save()
# ---------------------------------------------------------------------------

class TestGroupedExperimentSave:

    def _make_grouped_with_results(
        self, tmp_path: Path, measure_name: str = "pearson_r"
    ) -> GroupedExperiment:
        base = _make_base_config()
        cfg = GroupedExperimentConfig(
            dataset_groups={"g1": [], "g2": []},
            base_config=base,
            name="save_test",
            save_dir=tmp_path,
        )
        grouped = GroupedExperiment(cfg)
        m = _make_matrix([[1.0, 0.5], [0.5, 1.0]], ["F1", "N3"])
        grouped.averaged_pairwise_distances = {measure_name: m}

        for name in ["g1", "g2"]:
            mock_exp = MagicMock()
            mock_exp.save = MagicMock()
            grouped.experiments[name] = mock_exp

        return grouped

    def test_saves_averaged_csv(self, tmp_path):
        grouped = self._make_grouped_with_results(tmp_path)
        grouped.save(save_dir=tmp_path)

        csv = tmp_path / "data" / "averaged_pairwise_distances_pearson_r.csv"
        assert csv.exists()
        loaded = pd.read_csv(csv, index_col=0)
        assert "F1" in loaded.columns
        assert "N3" in loaded.columns

    def test_saves_heatmap_png(self, tmp_path):
        grouped = self._make_grouped_with_results(tmp_path)
        grouped.save(save_dir=tmp_path)

        png = tmp_path / "plots" / "averaged_pairwise_distances_pearson_r.png"
        assert png.exists()

    def test_delegates_to_child_experiments(self, tmp_path):
        grouped = self._make_grouped_with_results(tmp_path)
        grouped.save(save_dir=tmp_path)

        for name, exp in grouped.experiments.items():
            exp.save.assert_called_once_with(save_dir=tmp_path / "groups" / name)

    def test_creates_required_directories(self, tmp_path):
        grouped = self._make_grouped_with_results(tmp_path)
        grouped.save(save_dir=tmp_path)

        assert (tmp_path / "data").is_dir()
        assert (tmp_path / "plots").is_dir()


# ---------------------------------------------------------------------------
# GroupedExperiment.plot()
# ---------------------------------------------------------------------------

class TestGroupedExperimentPlot:

    def test_plot_returns_figures(self, tmp_path):
        import matplotlib.pyplot as plt

        base = _make_base_config()
        cfg = GroupedExperimentConfig(
            dataset_groups={"g": []},
            base_config=base,
            name="plot_test",
            save_dir=tmp_path,
        )
        grouped = GroupedExperiment(cfg)
        m = _make_matrix([[1.0, 0.5], [0.5, 1.0]], ["F1", "N3"])
        grouped.averaged_pairwise_distances = {"pearson_r": m}

        figures = grouped.plot()

        assert "pearson_r" in figures
        assert isinstance(figures["pearson_r"], plt.Figure)
        for fig in figures.values():
            plt.close(fig)

    def test_plot_raises_without_averaged_distances(self, tmp_path):
        base = _make_base_config()
        cfg = GroupedExperimentConfig(
            dataset_groups={},
            base_config=base,
            save_dir=tmp_path,
        )
        grouped = GroupedExperiment(cfg)
        with pytest.raises(RuntimeError, match="compute_averaged"):
            grouped.plot()


# ---------------------------------------------------------------------------
# End-to-end integration (real datasets, fast)
# ---------------------------------------------------------------------------

class TestGroupedExperimentIntegration:
    """Small integration test using real data — no mocks."""

    @pytest.fixture
    def fast_base_config(self):
        return ExperimentConfig(
            datasets=[],
            x_label="noise",
            cv_folds=1,
            run_mode=RunMode.COMPLEXITY_ONLY,
            plots=[],
            pairwise_distance_measures=[PearsonCorrelation()],
            ml_metrics=["accuracy"],
        )

    def test_run_and_compute(self, fast_base_config, tmp_path):
        grouped = GroupedExperiment(
            GroupedExperimentConfig(
                dataset_groups={
                    "moons": datasets_from_sweep(
                        DatasetSpec("Moons", {"num_samples": 80}),
                        ParameterSpec("moons_noise", [0.05, 0.3], label_format="{value}"),
                    ),
                    "circles": datasets_from_sweep(
                        DatasetSpec("Circles", {"num_samples": 80}),
                        ParameterSpec("circles_noise", [0.05, 0.3], label_format="{value}"),
                    ),
                },
                base_config=fast_base_config,
                name="integration_test",
                save_dir=tmp_path,
            )
        )

        grouped.run(verbose=False)
        result = grouped.compute_averaged_pairwise_distances()

        assert "pearson_r" in result
        mat = result["pearson_r"]
        # Square matrix; diagonal should be 1
        assert mat.shape[0] == mat.shape[1]
        assert np.allclose(np.diag(mat.values), 1.0, atol=1e-6)

    def test_save_produces_files(self, fast_base_config, tmp_path):
        grouped = GroupedExperiment(
            GroupedExperimentConfig(
                dataset_groups={
                    "g1": datasets_from_sweep(
                        DatasetSpec("Moons", {"num_samples": 60}),
                        ParameterSpec("moons_noise", [0.1, 0.3], label_format="{value}"),
                    ),
                },
                base_config=fast_base_config,
                name="save_integration",
                save_dir=tmp_path,
            )
        )

        grouped.run(verbose=False)
        grouped.compute_averaged_pairwise_distances()
        grouped.save()

        assert (tmp_path / "data" / "averaged_pairwise_distances_pearson_r.csv").exists()
        assert (tmp_path / "plots" / "averaged_pairwise_distances_pearson_r.png").exists()
        assert (tmp_path / "groups" / "g1").is_dir()
