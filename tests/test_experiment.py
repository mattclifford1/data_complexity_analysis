"""Tests for the experiment framework."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from data_complexity.experiments.experiment import (
    ParameterSpec,
    DatasetSpec,
    ExperimentConfig,
    ExperimentResults,
    Experiment,
    PlotType,
)
from data_complexity.experiments.experiment_configs import (
    gaussian_variance_config,
    gaussian_separation_config,
    moons_noise_config,
    circles_noise_config,
    get_config,
    list_configs,
    EXPERIMENT_CONFIGS,
)


class TestParameterSpec:
    """Tests for ParameterSpec dataclass."""

    def test_default_label_format(self):
        spec = ParameterSpec(name="scale", values=[1.0, 2.0])
        label = spec.format_label(1.5)
        assert label == "scale=1.5"

    def test_custom_label_format(self):
        spec = ParameterSpec(
            name="noise", values=[0.1, 0.2], label_format="n={value}"
        )
        label = spec.format_label(0.15)
        assert label == "n=0.15"

    def test_values_stored(self):
        spec = ParameterSpec(name="x", values=[1, 2, 3])
        assert spec.values == [1, 2, 3]


class TestDatasetSpec:
    """Tests for DatasetSpec dataclass."""

    def test_defaults(self):
        spec = DatasetSpec(dataset_type="Gaussian")
        assert spec.dataset_type == "Gaussian"
        assert spec.fixed_params == {}
        assert spec.num_samples == 400

    def test_with_params(self):
        spec = DatasetSpec(
            dataset_type="Moons",
            fixed_params={"random_state": 42},
            num_samples=200,
        )
        assert spec.fixed_params == {"random_state": 42}
        assert spec.num_samples == 200


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_auto_name_generation(self):
        config = ExperimentConfig(
            dataset=DatasetSpec(dataset_type="Gaussian"),
            vary_parameter=ParameterSpec(name="cov_scale", values=[1.0]),
        )
        assert config.name == "gaussian_cov_scale"

    def test_explicit_name(self):
        config = ExperimentConfig(
            dataset=DatasetSpec(dataset_type="Gaussian"),
            vary_parameter=ParameterSpec(name="cov_scale", values=[1.0]),
            name="my_experiment",
        )
        assert config.name == "my_experiment"

    def test_auto_save_dir(self):
        config = ExperimentConfig(
            dataset=DatasetSpec(dataset_type="Gaussian"),
            vary_parameter=ParameterSpec(name="cov_scale", values=[1.0]),
            name="test_exp",
        )
        assert "test_exp" in str(config.save_dir)
        assert config.save_dir.name == "test_exp"

    def test_explicit_save_dir(self):
        custom_dir = Path("/tmp/custom_dir")
        config = ExperimentConfig(
            dataset=DatasetSpec(dataset_type="Gaussian"),
            vary_parameter=ParameterSpec(name="cov_scale", values=[1.0]),
            save_dir=custom_dir,
        )
        assert config.save_dir == custom_dir

    def test_default_values(self):
        config = ExperimentConfig(
            dataset=DatasetSpec(dataset_type="Gaussian"),
            vary_parameter=ParameterSpec(name="x", values=[1]),
        )
        assert config.cv_folds == 5
        assert config.ml_metrics == ["accuracy", "f1"]
        assert config.correlation_target == "best_accuracy"
        assert PlotType.CORRELATIONS in config.plots
        assert PlotType.SUMMARY in config.plots


class TestExperimentResults:
    """Tests for ExperimentResults class."""

    @pytest.fixture
    def mock_config(self):
        return ExperimentConfig(
            dataset=DatasetSpec(dataset_type="Gaussian"),
            vary_parameter=ParameterSpec(name="scale", values=[1.0, 2.0]),
            ml_metrics=["accuracy"],
        )

    def test_add_result(self, mock_config):
        results = ExperimentResults(mock_config)
        results.add_result(
            param_value=1.0,
            complexity_metrics_dict={"F1": 0.5, "N3": 0.3},
            ml_results={
                "LogisticRegression": {"accuracy": {"mean": 0.9, "std": 0.05}},
                "KNN-5": {"accuracy": {"mean": 0.85, "std": 0.03}},
            },
        )
        results.finalize()

        assert len(results.complexity_df) == 1
        assert results.complexity_df.iloc[0]["F1"] == 0.5
        assert results.complexity_df.iloc[0]["param_value"] == 1.0

    def test_multiple_results(self, mock_config):
        results = ExperimentResults(mock_config)
        for val in [1.0, 2.0]:
            results.add_result(
                param_value=val,
                complexity_metrics_dict={"F1": val * 0.5},
                ml_results={
                    "Model": {"accuracy": {"mean": 1.0 / val, "std": 0.01}},
                },
            )
        results.finalize()

        assert len(results.complexity_df) == 2
        assert len(results.ml_df) == 2

    def test_get_param_values(self, mock_config):
        results = ExperimentResults(mock_config)
        results.add_result(1.0, {"F1": 0.5}, {"M": {"accuracy": {"mean": 0.9, "std": 0.1}}})
        results.add_result(2.0, {"F1": 0.3}, {"M": {"accuracy": {"mean": 0.8, "std": 0.1}}})
        results.finalize()

        assert results.get_param_values() == [1.0, 2.0]


class TestExperimentConfigPresets:
    """Tests for pre-defined experiment configurations."""

    def test_gaussian_variance_config(self):
        config = gaussian_variance_config()
        assert config.name == "gaussian_variance"
        assert config.dataset.dataset_type == "Gaussian"
        assert config.vary_parameter.name == "cov_scale"
        assert len(config.vary_parameter.values) > 3

    def test_gaussian_separation_config(self):
        config = gaussian_separation_config()
        assert config.name == "gaussian_separation"
        assert config.vary_parameter.name == "class_separation"

    def test_moons_noise_config(self):
        config = moons_noise_config()
        assert config.name == "moons_noise"
        assert config.dataset.dataset_type == "Moons"
        assert config.vary_parameter.name == "moons_noise"

    def test_circles_noise_config(self):
        config = circles_noise_config()
        assert config.name == "circles_noise"
        assert config.dataset.dataset_type == "Circles"

    def test_get_config(self):
        config = get_config("gaussian_variance")
        assert config.name == "gaussian_variance"

    def test_get_config_unknown(self):
        with pytest.raises(ValueError, match="Unknown config"):
            get_config("nonexistent_config")

    def test_list_configs(self):
        configs = list_configs()
        assert "gaussian_variance" in configs
        assert "moons_noise" in configs
        assert len(configs) == len(EXPERIMENT_CONFIGS)


class TestExperimentRun:
    """Tests for Experiment.run() with mocked dependencies."""

    @pytest.fixture
    def simple_config(self):
        return ExperimentConfig(
            dataset=DatasetSpec(dataset_type="Gaussian", num_samples=50),
            vary_parameter=ParameterSpec(name="cov_scale", values=[1.0, 2.0]),
            cv_folds=2,
            ml_metrics=["accuracy"],
        )

    def test_experiment_init(self, simple_config):
        exp = Experiment(simple_config)
        assert exp.config == simple_config
        assert exp.results is None

    @patch("data_complexity.experiments.experiment.complexity_metrics")
    @patch("data_complexity.experiments.experiment.evaluate_models")
    def test_run_with_mocked_data_loaders(
        self, mock_evaluate, mock_complexity, simple_config
    ):
        """Test experiment run with mocked data loaders."""
        mock_dataset = MagicMock()
        mock_dataset.get_data_dict.return_value = {
            "X": np.random.rand(50, 2),
            "y": np.array([0] * 25 + [1] * 25),
        }

        mock_get_dataset = MagicMock(return_value=mock_dataset)

        mock_complexity_instance = MagicMock()
        mock_complexity_instance.get_all_metrics_scalar.return_value = {
            "F1": 0.5,
            "N3": 0.3,
        }
        mock_complexity.return_value = mock_complexity_instance

        mock_evaluate.return_value = {
            "LogisticRegression": {"accuracy": {"mean": 0.9, "std": 0.05}},
        }

        exp = Experiment(simple_config)
        exp._get_dataset = mock_get_dataset

        results = exp.run(verbose=False)

        assert results is not None
        assert len(results.complexity_df) == 2
        assert len(results.ml_df) == 2
        assert mock_get_dataset.call_count == 2

    @patch("data_complexity.experiments.experiment.complexity_metrics")
    @patch("data_complexity.experiments.experiment.evaluate_models")
    def test_compute_correlations(self, mock_evaluate, mock_complexity, simple_config):
        """Test correlation computation."""
        mock_dataset = MagicMock()
        mock_dataset.get_data_dict.return_value = {
            "X": np.random.rand(50, 2),
            "y": np.array([0] * 25 + [1] * 25),
        }
        mock_get_dataset = MagicMock(return_value=mock_dataset)

        mock_complexity_instance = MagicMock()
        mock_complexity.return_value = mock_complexity_instance

        mock_complexity_instance.get_all_metrics_scalar.side_effect = [
            {"F1": 0.3, "N3": 0.2},
            {"F1": 0.6, "N3": 0.4},
        ]
        mock_evaluate.side_effect = [
            {"Model": {"accuracy": {"mean": 0.9, "std": 0.05}}},
            {"Model": {"accuracy": {"mean": 0.7, "std": 0.05}}},
        ]

        exp = Experiment(simple_config)
        exp._get_dataset = mock_get_dataset
        exp.run(verbose=False)

        corr_df = exp.compute_correlations()

        assert "correlation" in corr_df.columns
        assert "p_value" in corr_df.columns
        assert "complexity_metric" in corr_df.columns
        assert len(corr_df) > 0


class TestSaveLoad:
    """Tests for save/load round-trip."""

    @pytest.fixture
    def results_with_data(self):
        config = ExperimentConfig(
            dataset=DatasetSpec(dataset_type="Gaussian"),
            vary_parameter=ParameterSpec(name="scale", values=[1.0, 2.0]),
            ml_metrics=["accuracy"],
        )
        results = ExperimentResults(config)

        results.add_result(
            1.0,
            {"F1": 0.5, "N3": 0.3},
            {"Model": {"accuracy": {"mean": 0.9, "std": 0.05}}},
        )
        results.add_result(
            2.0,
            {"F1": 0.4, "N3": 0.5},
            {"Model": {"accuracy": {"mean": 0.8, "std": 0.06}}},
        )
        results.finalize()

        results.correlations_df = pd.DataFrame(
            {
                "complexity_metric": ["F1", "N3"],
                "ml_metric": ["best_accuracy", "best_accuracy"],
                "correlation": [-0.8, 0.6],
                "p_value": [0.01, 0.05],
                "abs_correlation": [0.8, 0.6],
            }
        )

        return config, results

    def test_save_load_roundtrip(self, results_with_data, tmp_path):
        config, results = results_with_data
        save_dir = tmp_path / "test_results"

        exp = Experiment(config)
        exp.results = results
        exp.save(save_dir)

        # Check for new subfolder structure
        assert (save_dir / "data" / "complexity_metrics.csv").exists()
        assert (save_dir / "data" / "ml_performance.csv").exists()
        assert (save_dir / "data" / "correlations.csv").exists()

        exp2 = Experiment(config)
        loaded = exp2.load_results(save_dir)

        pd.testing.assert_frame_equal(
            loaded.complexity_df.reset_index(drop=True),
            results.complexity_df.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            loaded.ml_df.reset_index(drop=True),
            results.ml_df.reset_index(drop=True),
        )

    def test_load_legacy_flat_structure(self, tmp_path):
        """Test that load_results can read old flat structure."""
        # Create old-style flat structure
        save_dir = tmp_path / "legacy"
        save_dir.mkdir()

        # Create minimal CSVs in flat structure
        complexity_df = pd.DataFrame({"F1": [0.5], "param_value": [1.0]})
        ml_df = pd.DataFrame({"best_accuracy": [0.9], "param_value": [1.0]})

        complexity_df.to_csv(save_dir / "complexity_metrics.csv", index=False)
        ml_df.to_csv(save_dir / "ml_performance.csv", index=False)

        # Load using new code
        config = ExperimentConfig(
            dataset=DatasetSpec(dataset_type="Gaussian"),
            vary_parameter=ParameterSpec(name="scale", values=[1.0]),
            ml_metrics=["accuracy"],
        )
        exp = Experiment(config)
        results = exp.load_results(save_dir)

        # Verify it loaded correctly
        assert results.complexity_df is not None
        assert results.ml_df is not None
        assert len(results.complexity_df) == 1
        assert len(results.ml_df) == 1

    def test_save_creates_subfolders(self, results_with_data, tmp_path):
        """Test that save creates proper subfolder structure."""
        config, results = results_with_data
        save_dir = tmp_path / "test_subfolders"

        exp = Experiment(config)
        exp.results = results
        exp.save(save_dir)

        # Verify subfolder structure
        assert (save_dir / "data").is_dir()
        assert (save_dir / "plots").is_dir()
        assert (save_dir / "datasets").is_dir()

        # Verify files in correct locations
        assert (save_dir / "data" / "complexity_metrics.csv").exists()
        assert (save_dir / "data" / "ml_performance.csv").exists()
        assert (save_dir / "plots" / "correlations.png").exists()


class TestPlotting:
    """Tests for plotting functionality."""

    @pytest.fixture
    def experiment_with_results(self):
        config = ExperimentConfig(
            dataset=DatasetSpec(dataset_type="Gaussian"),
            vary_parameter=ParameterSpec(name="scale", values=[1.0, 2.0, 3.0]),
            ml_metrics=["accuracy"],
        )
        results = ExperimentResults(config)

        for val in [1.0, 2.0, 3.0]:
            results.add_result(
                val,
                {"F1": val * 0.2, "N3": 1.0 - val * 0.2},
                {"Model": {"accuracy": {"mean": 1.0 - val * 0.1, "std": 0.05}}},
            )
        results.finalize()

        results.correlations_df = pd.DataFrame(
            {
                "complexity_metric": ["F1", "N3"],
                "ml_metric": ["best_accuracy", "best_accuracy"],
                "correlation": [0.9, -0.9],
                "p_value": [0.001, 0.001],
                "abs_correlation": [0.9, 0.9],
            }
        )

        exp = Experiment(config)
        exp.results = results
        return exp

    def test_plot_returns_figures(self, experiment_with_results):
        import matplotlib.pyplot as plt

        figures = experiment_with_results.plot()

        assert PlotType.CORRELATIONS in figures
        assert PlotType.SUMMARY in figures

        for fig in figures.values():
            assert isinstance(fig, plt.Figure)
            plt.close(fig)


class TestPrintSummary:
    """Tests for print_summary method."""

    @pytest.fixture
    def experiment_with_correlations(self):
        config = ExperimentConfig(
            dataset=DatasetSpec(dataset_type="Gaussian"),
            vary_parameter=ParameterSpec(name="scale", values=[1.0]),
        )
        results = ExperimentResults(config)
        results.add_result(
            1.0,
            {"F1": 0.5},
            {"Model": {"accuracy": {"mean": 0.9, "std": 0.05}}},
        )
        results.finalize()
        results.correlations_df = pd.DataFrame(
            {
                "complexity_metric": ["F1"],
                "ml_metric": ["best_accuracy"],
                "correlation": [-0.9],
                "p_value": [0.01],
                "abs_correlation": [0.9],
            }
        )

        exp = Experiment(config)
        exp.results = results
        return exp

    def test_print_summary_runs(self, experiment_with_correlations, capsys):
        experiment_with_correlations.print_summary(top_n=5)
        captured = capsys.readouterr()
        assert "F1" in captured.out
        assert "correlation" in captured.out.lower() or "-0.9" in captured.out


class TestErrorHandling:
    """Tests for error handling."""

    def test_compute_correlations_before_run(self):
        config = ExperimentConfig(
            dataset=DatasetSpec(dataset_type="Gaussian"),
            vary_parameter=ParameterSpec(name="scale", values=[1.0]),
        )
        exp = Experiment(config)

        with pytest.raises(RuntimeError, match="Must run experiment"):
            exp.compute_correlations()

    def test_plot_before_run(self):
        config = ExperimentConfig(
            dataset=DatasetSpec(dataset_type="Gaussian"),
            vary_parameter=ParameterSpec(name="scale", values=[1.0]),
        )
        exp = Experiment(config)

        with pytest.raises(RuntimeError, match="Must run experiment"):
            exp.plot()

    def test_save_before_run(self):
        config = ExperimentConfig(
            dataset=DatasetSpec(dataset_type="Gaussian"),
            vary_parameter=ParameterSpec(name="scale", values=[1.0]),
        )
        exp = Experiment(config)

        with pytest.raises(RuntimeError, match="Must run experiment"):
            exp.save()
