"""Tests for the experiment framework."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from data_complexity.model_experiments.experiment import (
    ParameterSpec,
    DatasetSpec,
    ExperimentConfig,
    ExperimentResultsContainer,
    Experiment,
    PlotType,
    _average_dicts,
    _average_ml_results,
    _std_dicts,
)
from data_complexity.model_experiments.experiment_configs import (
    gaussian_variance_config,
    gaussian_separation_config,
    gaussian_imbalance_config,
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

    def test_with_params(self):
        spec = DatasetSpec(
            dataset_type="Moons",
            fixed_params={"random_state": 42, "num_samples": 200, "train_size": 0.7},
        )
        assert spec.fixed_params == {"random_state": 42, "num_samples": 200, "train_size": 0.7}
        assert spec.fixed_params["num_samples"] == 200
        assert spec.fixed_params["train_size"] == 0.7


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


class TestExperimentResultsContainer:
    """Tests for ExperimentResultsContainer class."""

    @pytest.fixture
    def mock_config(self):
        return ExperimentConfig(
            dataset=DatasetSpec(dataset_type="Gaussian"),
            vary_parameter=ParameterSpec(name="scale", values=[1.0, 2.0]),
            ml_metrics=["accuracy"],
        )

    def test_add_result(self, mock_config):
        results = ExperimentResultsContainer(mock_config)
        results.add_result(
            param_value=1.0,
            complexity_metrics_dict={"F1": 0.5, "N3": 0.3},
            ml_results={
                "LogisticRegression": {"accuracy": {"mean": 0.9, "std": 0.05}},
                "KNN-5": {"accuracy": {"mean": 0.85, "std": 0.03}},
            },
        )
        results.covert_to_df()

        assert len(results.complexity_df) == 1
        assert results.complexity_df.iloc[0]["F1"] == 0.5
        assert results.complexity_df.iloc[0]["param_value"] == 1.0

    def test_multiple_results(self, mock_config):
        results = ExperimentResultsContainer(mock_config)
        for val in [1.0, 2.0]:
            results.add_result(
                param_value=val,
                complexity_metrics_dict={"F1": val * 0.5},
                ml_results={
                    "Model": {"accuracy": {"mean": 1.0 / val, "std": 0.01}},
                },
            )
        results.covert_to_df()

        assert len(results.complexity_df) == 2
        assert len(results.ml_df) == 2

    def test_get_param_values(self, mock_config):
        results = ExperimentResultsContainer(mock_config)
        results.add_result(1.0, {"F1": 0.5}, {"M": {"accuracy": {"mean": 0.9, "std": 0.1}}})
        results.add_result(2.0, {"F1": 0.3}, {"M": {"accuracy": {"mean": 0.8, "std": 0.1}}})
        results.covert_to_df()

        assert results.get_param_values() == [1.0, 2.0]

    def test_add_split_result(self, mock_config):
        """Test adding train/test split results."""
        results = ExperimentResultsContainer(mock_config)
        results.add_split_result(
            param_value=1.0,
            train_complexity_dict={"F1": 0.5, "N3": 0.3},
            test_complexity_dict={"F1": 0.6, "N3": 0.4},
            train_ml_results={
                "Model": {"accuracy": {"mean": 0.95, "std": 0.0}},
            },
            test_ml_results={
                "Model": {"accuracy": {"mean": 0.85, "std": 0.0}},
            },
            train_complexity_std_dict={"F1": 0.01, "N3": 0.02},
            test_complexity_std_dict={"F1": 0.03, "N3": 0.04},
        )
        results.covert_to_df()

        # complexity_df returns train complexity
        assert len(results.complexity_df) == 1
        assert results.complexity_df.iloc[0]["F1"] == 0.5

        # ml_df returns test ML
        assert len(results.ml_df) == 1
        assert results.ml_df.iloc[0]["best_accuracy"] == 0.85

        # Explicit train/test access
        assert results.train_complexity_df.iloc[0]["F1"] == 0.5
        assert results.test_complexity_df.iloc[0]["F1"] == 0.6
        assert results.train_ml_df.iloc[0]["best_accuracy"] == 0.95
        assert results.test_ml_df.iloc[0]["best_accuracy"] == 0.85

        # Std columns present in complexity DFs
        assert results.train_complexity_df.iloc[0]["F1_std"] == 0.01
        assert results.test_complexity_df.iloc[0]["F1_std"] == 0.03

        # Std columns present in ML DFs
        assert results.train_ml_df.iloc[0]["Model_accuracy_std"] == 0.0
        assert results.test_ml_df.iloc[0]["Model_accuracy_std"] == 0.0

    def test_split_result_backward_compat(self, mock_config):
        """Test that complexity_df and ml_df return train/test respectively."""
        results = ExperimentResultsContainer(mock_config)
        results.add_split_result(
            param_value=1.0,
            train_complexity_dict={"F1": 0.3},
            test_complexity_dict={"F1": 0.7},
            train_ml_results={"M": {"accuracy": {"mean": 0.99, "std": 0.0}}},
            test_ml_results={"M": {"accuracy": {"mean": 0.80, "std": 0.0}}},
        )
        results.covert_to_df()

        # complexity_df -> train complexity
        assert results.complexity_df.iloc[0]["F1"] == 0.3
        # ml_df -> test ML
        assert results.ml_df.iloc[0]["best_accuracy"] == 0.80


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

    def test_gaussian_imbalance_config(self):
        config = gaussian_imbalance_config()
        assert config.name == "gaussian_imbalance"
        assert config.vary_parameter.name == "minority_reduce_scaler"

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
            dataset=DatasetSpec(dataset_type="Gaussian", fixed_params={"num_samples": 50}),
            vary_parameter=ParameterSpec(name="cov_scale", values=[1.0, 2.0]),
            cv_folds=2,
            ml_metrics=["accuracy"],
        )

    def test_experiment_init(self, simple_config):
        exp = Experiment(simple_config)
        assert exp.config == simple_config
        assert exp.results is None

    @patch("data_complexity.model_experiments.experiment.run.ComplexityMetrics")
    @patch("data_complexity.model_experiments.experiment.run.evaluate_models_train_test")
    def test_run_with_mocked_data_loaders(
        self, mock_evaluate, mock_complexity, simple_config
    ):
        """Test experiment run with mocked data loaders."""
        train_data = {"X": np.random.rand(25, 2), "y": np.array([0] * 12 + [1] * 13)}
        test_data = {"X": np.random.rand(25, 2), "y": np.array([0] * 13 + [1] * 12)}

        mock_dataset = MagicMock()
        mock_dataset.get_data_dict.return_value = train_data
        mock_dataset.get_train_test_split.return_value = (train_data, test_data)

        mock_get_dataset = MagicMock(return_value=mock_dataset)

        mock_complexity_instance = MagicMock()
        mock_complexity_instance.get_all_metrics_scalar.return_value = {
            "F1": 0.5,
            "N3": 0.3,
        }
        mock_complexity.return_value = mock_complexity_instance

        # evaluate_models_train_test returns a tuple (train_results, test_results)
        mock_evaluate.return_value = (
            {"LogisticRegression": {"accuracy": {"mean": 0.95, "std": 0.0}}},
            {"LogisticRegression": {"accuracy": {"mean": 0.9, "std": 0.0}}},
        )

        exp = Experiment(simple_config)
        exp._get_dataset = mock_get_dataset

        results = exp.run(verbose=False)

        assert results is not None
        assert len(results.complexity_df) == 2
        assert len(results.ml_df) == 2
        assert mock_get_dataset.call_count == 2
        # Each parameter value x cv_folds seeds
        assert mock_evaluate.call_count == 2 * simple_config.cv_folds

        # Std columns present in complexity and ML DFs
        assert "F1_std" in results.train_complexity_df.columns
        assert "N3_std" in results.train_complexity_df.columns
        assert "F1_std" in results.test_complexity_df.columns
        assert "LogisticRegression_accuracy_std" in results.train_ml_df.columns
        assert "LogisticRegression_accuracy_std" in results.test_ml_df.columns

    @patch("data_complexity.model_experiments.experiment.run.ComplexityMetrics")
    @patch("data_complexity.model_experiments.experiment.run.evaluate_models_train_test")
    def test_compute_correlations(self, mock_evaluate, mock_complexity, simple_config):
        """Test correlation computation."""
        train_data = {"X": np.random.rand(25, 2), "y": np.array([0] * 12 + [1] * 13)}
        test_data = {"X": np.random.rand(25, 2), "y": np.array([0] * 13 + [1] * 12)}

        mock_dataset = MagicMock()
        mock_dataset.get_train_test_split.return_value = (train_data, test_data)
        mock_get_dataset = MagicMock(return_value=mock_dataset)

        mock_complexity_instance = MagicMock()
        mock_complexity.return_value = mock_complexity_instance

        # Called 2x per seed (train + test), cv_folds=2, 2 param values = 8 calls
        # For param 1.0: train F1=0.3, test F1=0.3 (x2 seeds)
        # For param 2.0: train F1=0.6, test F1=0.6 (x2 seeds)
        complexity_values = [
            {"F1": 0.3, "N3": 0.2}, {"F1": 0.3, "N3": 0.2},  # param=1.0, seed=0
            {"F1": 0.3, "N3": 0.2}, {"F1": 0.3, "N3": 0.2},  # param=1.0, seed=1
            {"F1": 0.6, "N3": 0.4}, {"F1": 0.6, "N3": 0.4},  # param=2.0, seed=0
            {"F1": 0.6, "N3": 0.4}, {"F1": 0.6, "N3": 0.4},  # param=2.0, seed=1
        ]
        mock_complexity_instance.get_all_metrics_scalar.side_effect = complexity_values

        # Return different accuracy values for param=1.0 vs param=2.0
        mock_evaluate.side_effect = [
            ({"Model": {"accuracy": {"mean": 0.95, "std": 0.0}}}, {"Model": {"accuracy": {"mean": 0.7, "std": 0.0}}}),
            ({"Model": {"accuracy": {"mean": 0.95, "std": 0.0}}}, {"Model": {"accuracy": {"mean": 0.7, "std": 0.0}}}),
            ({"Model": {"accuracy": {"mean": 0.95, "std": 0.0}}}, {"Model": {"accuracy": {"mean": 0.9, "std": 0.0}}}),
            ({"Model": {"accuracy": {"mean": 0.95, "std": 0.0}}}, {"Model": {"accuracy": {"mean": 0.9, "std": 0.0}}}),
        ]

        exp = Experiment(simple_config)
        exp._get_dataset = mock_get_dataset
        exp.run(verbose=False)

        corr_df = exp.compute_correlations()

        assert "correlation" in corr_df.columns
        assert "p_value" in corr_df.columns
        assert "complexity_metric" in corr_df.columns
        assert len(corr_df) > 0

    @patch("data_complexity.model_experiments.experiment.run.ComplexityMetrics")
    @patch("data_complexity.model_experiments.experiment.run.evaluate_models_train_test")
    def test_run_stores_train_test_separately(
        self, mock_evaluate, mock_complexity, simple_config
    ):
        """Test that run stores train and test results separately."""
        train_data = {"X": np.random.rand(25, 2), "y": np.array([0] * 12 + [1] * 13)}
        test_data = {"X": np.random.rand(25, 2), "y": np.array([0] * 13 + [1] * 12)}

        mock_dataset = MagicMock()
        mock_dataset.get_train_test_split.return_value = (train_data, test_data)
        mock_get_dataset = MagicMock(return_value=mock_dataset)

        mock_complexity_instance = MagicMock()
        mock_complexity_instance.get_all_metrics_scalar.return_value = {
            "F1": 0.5,
        }
        mock_complexity.return_value = mock_complexity_instance

        mock_evaluate.return_value = (
            {"Model": {"accuracy": {"mean": 0.99, "std": 0.0}}},
            {"Model": {"accuracy": {"mean": 0.85, "std": 0.0}}},
        )

        exp = Experiment(simple_config)
        exp._get_dataset = mock_get_dataset
        results = exp.run(verbose=False)

        # Train and test DFs should be populated
        assert results.train_complexity_df is not None
        assert results.test_complexity_df is not None
        assert results.train_ml_df is not None
        assert results.test_ml_df is not None

        # Sizes correct
        assert len(results.train_complexity_df) == 2
        assert len(results.test_complexity_df) == 2
        assert len(results.train_ml_df) == 2
        assert len(results.test_ml_df) == 2

    @patch("data_complexity.model_experiments.experiment.run.ComplexityMetrics")
    @patch("data_complexity.model_experiments.experiment.run.evaluate_models_train_test")
    def test_correlation_sources(
        self, mock_evaluate, mock_complexity, simple_config
    ):
        """Test that correlations can use different sources."""
        train_data = {"X": np.random.rand(25, 2), "y": np.array([0] * 12 + [1] * 13)}
        test_data = {"X": np.random.rand(25, 2), "y": np.array([0] * 13 + [1] * 12)}

        mock_dataset = MagicMock()
        mock_dataset.get_train_test_split.return_value = (train_data, test_data)
        mock_get_dataset = MagicMock(return_value=mock_dataset)

        # Different complexity for train vs test
        mock_complexity_instance = MagicMock()
        call_count = [0]

        def side_effect_complexity():
            call_count[0] += 1
            # Even calls are train, odd are test
            if call_count[0] % 2 == 1:
                return {"F1": 0.3 * (1 + (call_count[0] // 4))}
            else:
                return {"F1": 0.7 * (1 + (call_count[0] // 4))}

        mock_complexity_instance.get_all_metrics_scalar.side_effect = side_effect_complexity
        mock_complexity.return_value = mock_complexity_instance

        # Return different accuracy values for param=1.0 vs param=2.0 so ml_values isn't constant
        mock_evaluate.side_effect = [
            ({"Model": {"accuracy": {"mean": 0.95, "std": 0.0}}}, {"Model": {"accuracy": {"mean": 0.7, "std": 0.0}}}),
            ({"Model": {"accuracy": {"mean": 0.95, "std": 0.0}}}, {"Model": {"accuracy": {"mean": 0.7, "std": 0.0}}}),
            ({"Model": {"accuracy": {"mean": 0.95, "std": 0.0}}}, {"Model": {"accuracy": {"mean": 0.9, "std": 0.0}}}),
            ({"Model": {"accuracy": {"mean": 0.95, "std": 0.0}}}, {"Model": {"accuracy": {"mean": 0.9, "std": 0.0}}}),
        ]

        exp = Experiment(simple_config)
        exp._get_dataset = mock_get_dataset
        exp.run(verbose=False)

        # Both source options should work
        corr_train = exp.compute_correlations(complexity_source="train", ml_source="test")
        assert len(corr_train) > 0

        corr_test = exp.compute_correlations(complexity_source="test", ml_source="test")
        assert len(corr_test) > 0


class TestAveragingHelpers:
    """Tests for _average_dicts and _average_ml_results."""

    def test_average_dicts(self):
        dicts = [{"F1": 0.3, "N3": 0.2}, {"F1": 0.5, "N3": 0.4}]
        result = _average_dicts(dicts)
        assert abs(result["F1"] - 0.4) < 1e-10
        assert abs(result["N3"] - 0.3) < 1e-10

    def test_average_dicts_empty(self):
        assert _average_dicts([]) == {}

    def test_average_ml_results(self):
        results = [
            {"Model": {"accuracy": {"mean": 0.9, "std": 0.0}}},
            {"Model": {"accuracy": {"mean": 0.8, "std": 0.0}}},
        ]
        averaged = _average_ml_results(results)
        assert abs(averaged["Model"]["accuracy"]["mean"] - 0.85) < 1e-10
        assert averaged["Model"]["accuracy"]["std"] > 0  # Should have nonzero std

    def test_average_ml_results_empty(self):
        assert _average_ml_results([]) == {}

    def test_std_dicts(self):
        dicts = [{"F1": 0.3, "N3": 0.2}, {"F1": 0.5, "N3": 0.4}]
        result = _std_dicts(dicts)
        assert abs(result["F1"] - np.std([0.3, 0.5])) < 1e-10
        assert abs(result["N3"] - np.std([0.2, 0.4])) < 1e-10

    def test_std_dicts_empty(self):
        assert _std_dicts([]) == {}

    def test_std_dicts_single(self):
        dicts = [{"F1": 0.5}]
        result = _std_dicts(dicts)
        assert result["F1"] == 0.0


class TestSaveLoad:
    """Tests for save/load round-trip."""

    @pytest.fixture
    def results_with_data(self):
        config = ExperimentConfig(
            dataset=DatasetSpec(dataset_type="Gaussian"),
            vary_parameter=ParameterSpec(name="scale", values=[1.0, 2.0]),
            ml_metrics=["accuracy"],
        )
        results = ExperimentResultsContainer(config)

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
        results.covert_to_df()

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

    @pytest.fixture
    def split_results_with_data(self):
        config = ExperimentConfig(
            dataset=DatasetSpec(dataset_type="Gaussian"),
            vary_parameter=ParameterSpec(name="scale", values=[1.0, 2.0]),
            ml_metrics=["accuracy"],
        )
        results = ExperimentResultsContainer(config)

        results.add_split_result(
            1.0,
            {"F1": 0.5, "N3": 0.3},
            {"F1": 0.6, "N3": 0.4},
            {"Model": {"accuracy": {"mean": 0.95, "std": 0.0}}},
            {"Model": {"accuracy": {"mean": 0.85, "std": 0.0}}},
        )
        results.add_split_result(
            2.0,
            {"F1": 0.4, "N3": 0.5},
            {"F1": 0.5, "N3": 0.6},
            {"Model": {"accuracy": {"mean": 0.90, "std": 0.0}}},
            {"Model": {"accuracy": {"mean": 0.80, "std": 0.0}}},
        )
        results.covert_to_df()

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

    def test_save_load_split_roundtrip(self, split_results_with_data, tmp_path):
        """Test save/load round-trip with train/test split results."""
        config, results = split_results_with_data
        save_dir = tmp_path / "test_split_results"

        exp = Experiment(config)
        exp.results = results
        exp.save(save_dir)

        # Check train/test CSVs exist
        assert (save_dir / "data" / "train_complexity_metrics.csv").exists()
        assert (save_dir / "data" / "test_complexity_metrics.csv").exists()
        assert (save_dir / "data" / "train_ml_performance.csv").exists()
        assert (save_dir / "data" / "test_ml_performance.csv").exists()

        # Load and verify
        exp2 = Experiment(config)
        loaded = exp2.load_results(save_dir)

        pd.testing.assert_frame_equal(
            loaded.train_complexity_df.reset_index(drop=True),
            results.train_complexity_df.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            loaded.test_complexity_df.reset_index(drop=True),
            results.test_complexity_df.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            loaded.train_ml_df.reset_index(drop=True),
            results.train_ml_df.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            loaded.test_ml_df.reset_index(drop=True),
            results.test_ml_df.reset_index(drop=True),
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

    def test_save_creates_metadata_file(self, results_with_data, tmp_path):
        """Test that save creates experiment metadata JSON file."""
        import json

        config, results = results_with_data
        save_dir = tmp_path / "test_metadata"

        exp = Experiment(config)
        exp.results = results
        exp.save(save_dir)

        # Check metadata file exists
        metadata_path = save_dir / "experiment_metadata.json"
        assert metadata_path.exists()

        # Load and verify metadata content
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["experiment_name"] == config.name
        # assert "timestamp" in metadata
        assert metadata["dataset"]["type"] == "Gaussian"
        assert metadata["vary_parameter"]["name"] == "scale"
        assert metadata["ml_metrics"] == ["accuracy"]
        assert metadata["cv_folds"] == 5
        assert "ml_models" in metadata
        assert len(metadata["ml_models"]) > 0


class TestPlotting:
    """Tests for plotting functionality."""

    @pytest.fixture
    def experiment_with_results(self):
        config = ExperimentConfig(
            dataset=DatasetSpec(dataset_type="Gaussian"),
            vary_parameter=ParameterSpec(name="scale", values=[1.0, 2.0, 3.0]),
            ml_metrics=["accuracy"],
        )
        results = ExperimentResultsContainer(config)

        for val in [1.0, 2.0, 3.0]:
            results.add_result(
                val,
                {"F1": val * 0.2, "N3": 1.0 - val * 0.2},
                {"Model": {"accuracy": {"mean": 1.0 - val * 0.1, "std": 0.05}}},
            )
        results.covert_to_df()

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
        results = ExperimentResultsContainer(config)
        results.add_result(
            1.0,
            {"F1": 0.5},
            {"Model": {"accuracy": {"mean": 0.9, "std": 0.05}}},
        )
        results.covert_to_df()
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


class TestParallelRun:
    """Tests for Experiment.run() with n_jobs != 1."""

    @pytest.fixture
    def tiny_config(self):
        """Minimal config: 2 param values, 1 seed, small dataset — runs fast."""
        from data_complexity.model_experiments.classification import LogisticRegressionModel, AccuracyMetric
        return ExperimentConfig(
            dataset=DatasetSpec(
                dataset_type="Gaussian",
                fixed_params={"num_samples": 60, "cov_type": "spherical", "train_size": 0.5},
            ),
            vary_parameter=ParameterSpec(name="cov_scale", values=[0.5, 2.0]),
            cv_folds=1,
            ml_metrics=["accuracy"],
            models=[LogisticRegressionModel()],
            plots=[],
        )

    def test_n_jobs_minus1_returns_results(self, tiny_config):
        """Parallel run with n_jobs=-1 should return a populated results container."""
        exp = Experiment(tiny_config)
        results = exp.run(verbose=False, n_jobs=-1)

        assert results is not None
        assert results.train_complexity_df is not None
        assert results.test_complexity_df is not None
        assert results.train_ml_df is not None
        assert results.test_ml_df is not None
        assert len(results.train_complexity_df) == 2
        assert len(results.test_complexity_df) == 2

    def test_parallel_matches_sequential(self, tiny_config):
        """Results from n_jobs=-1 should match n_jobs=1 (same seeds, same data)."""
        exp_seq = Experiment(tiny_config)
        results_seq = exp_seq.run(verbose=False, n_jobs=1)

        exp_par = Experiment(tiny_config)
        results_par = exp_par.run(verbose=False, n_jobs=-1)

        # Sort both by param_value to ensure consistent ordering before comparison
        seq_cmplx = results_seq.train_complexity_df.sort_values("param_value").reset_index(drop=True)
        par_cmplx = results_par.train_complexity_df.sort_values("param_value").reset_index(drop=True)
        pd.testing.assert_frame_equal(seq_cmplx, par_cmplx, atol=1e-10)

        seq_ml = results_seq.test_ml_df.sort_values("param_value").reset_index(drop=True)
        par_ml = results_par.test_ml_df.sort_values("param_value").reset_index(drop=True)
        pd.testing.assert_frame_equal(seq_ml, par_ml, atol=1e-10)

    def test_parallel_datasets_not_populated(self, tiny_config):
        """self.datasets should be empty after a parallel run."""
        exp = Experiment(tiny_config)
        exp.run(verbose=False, n_jobs=-1)
        assert exp.datasets == {}

    def test_sequential_datasets_populated(self, tiny_config):
        """self.datasets should be populated after a sequential run."""
        exp = Experiment(tiny_config)
        exp.run(verbose=False, n_jobs=1)
        assert len(exp.datasets) == len(tiny_config.vary_parameter.values)

    def test_n_jobs_2_returns_results(self, tiny_config):
        """n_jobs=2 (explicit worker count) should also work correctly."""
        exp = Experiment(tiny_config)
        results = exp.run(verbose=False, n_jobs=2)

        assert results is not None
        assert len(results.train_complexity_df) == 2


class TestEvaluateModelsTrainTest:
    """Tests for evaluate_models_train_test function."""

    def test_basic_evaluation(self):
        from data_complexity.model_experiments.classification import (
            evaluate_models_train_test,
            LogisticRegressionModel,
            AccuracyMetric,
        )

        np.random.seed(42)
        X_train = np.vstack([
            np.random.randn(50, 2) + [2, 2],
            np.random.randn(50, 2) + [-2, -2],
        ])
        y_train = np.array([0] * 50 + [1] * 50)
        X_test = np.vstack([
            np.random.randn(20, 2) + [2, 2],
            np.random.randn(20, 2) + [-2, -2],
        ])
        y_test = np.array([0] * 20 + [1] * 20)

        train_data = {"X": X_train, "y": y_train}
        test_data = {"X": X_test, "y": y_test}

        train_results, test_results = evaluate_models_train_test(
            train_data, test_data,
            models=[LogisticRegressionModel()],
            metrics=[AccuracyMetric()],
        )

        assert "LogisticRegression" in train_results
        assert "LogisticRegression" in test_results
        assert "accuracy" in train_results["LogisticRegression"]
        assert "accuracy" in test_results["LogisticRegression"]
        # Train accuracy should be high for well-separated data
        assert train_results["LogisticRegression"]["accuracy"]["mean"] > 0.8
        assert test_results["LogisticRegression"]["accuracy"]["mean"] > 0.8
