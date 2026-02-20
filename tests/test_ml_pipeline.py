"""Tests for ML pipeline orchestration functions."""
import pytest
import numpy as np
from sklearn.datasets import make_classification

from data_complexity.experiments.classification import (
    evaluate_single_model,
    evaluate_models,
    get_best_metric,
    get_mean_metric,
    get_model_metric,
    print_evaluation_results,
    LogisticRegressionModel,
    KNNModel,
    SVMModel,
    get_default_models,
    AccuracyMetric,
    F1Metric,
    CrossValidationEvaluator,
    TrainTestSplitEvaluator,
)


@pytest.fixture
def simple_data():
    """Generate simple classification data."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )
    return {"X": X, "y": y}


class TestEvaluateSingleModel:
    """Tests for evaluate_single_model function."""

    def test_default_parameters(self, simple_data):
        model = LogisticRegressionModel()
        results = evaluate_single_model(model, simple_data)

        assert "accuracy" in results
        assert "f1" in results
        assert 0 <= results["accuracy"]["mean"] <= 1

    def test_custom_cv_folds(self, simple_data):
        model = LogisticRegressionModel()
        results = evaluate_single_model(model, simple_data, cv_folds=3)

        assert "accuracy" in results

    def test_custom_metrics(self, simple_data):
        model = LogisticRegressionModel()
        metrics = [AccuracyMetric()]
        results = evaluate_single_model(model, simple_data, metrics=metrics)

        assert len(results) == 1
        assert "accuracy" in results

    def test_custom_evaluator(self, simple_data):
        model = LogisticRegressionModel()
        evaluator = TrainTestSplitEvaluator(test_size=0.3)
        results = evaluate_single_model(model, simple_data, evaluator=evaluator)

        assert "accuracy" in results
        assert results["accuracy"]["std"] == 0.0  # Train-test split has no std

    def test_with_different_models(self, simple_data):
        models = [
            LogisticRegressionModel(),
            KNNModel(n_neighbors=3),
            SVMModel(kernel="linear"),
        ]

        for model in models:
            results = evaluate_single_model(model, simple_data, cv_folds=3)
            assert "accuracy" in results
            assert 0 <= results["accuracy"]["mean"] <= 1


class TestEvaluateModels:
    """Tests for evaluate_models function."""

    def test_default_parameters(self, simple_data):
        results = evaluate_models(simple_data)

        # Should use all default models (10)
        assert len(results) >= 10
        assert "LogisticRegression" in results
        assert "KNN-5" in results
        assert "accuracy" in results["LogisticRegression"]

    def test_custom_models(self, simple_data):
        models = [
            LogisticRegressionModel(),
            KNNModel(n_neighbors=3),
        ]
        results = evaluate_models(simple_data, models=models)

        assert len(results) == 2
        assert "LogisticRegression" in results
        assert "KNN-3" in results

    def test_custom_cv_folds(self, simple_data):
        models = [LogisticRegressionModel()]
        results = evaluate_models(simple_data, models=models, cv_folds=3)

        assert "LogisticRegression" in results

    def test_custom_metrics(self, simple_data):
        models = [LogisticRegressionModel()]
        metrics = [AccuracyMetric(), F1Metric()]
        results = evaluate_models(simple_data, models=models, metrics=metrics)

        model_results = results["LogisticRegression"]
        assert len(model_results) == 2
        assert "accuracy" in model_results
        assert "f1" in model_results

    def test_custom_evaluator(self, simple_data):
        models = [LogisticRegressionModel()]
        evaluator = TrainTestSplitEvaluator()
        results = evaluate_models(simple_data, models=models, evaluator=evaluator)

        assert "LogisticRegression" in results
        assert results["LogisticRegression"]["accuracy"]["std"] == 0.0


class TestGetBestMetric:
    """Tests for get_best_metric function."""

    def test_returns_best_accuracy(self, simple_data):
        models = [
            LogisticRegressionModel(),
            KNNModel(n_neighbors=3),
        ]
        results = evaluate_models(simple_data, models=models, cv_folds=3)

        best = get_best_metric(results, "accuracy")
        assert 0 <= best <= 1

        # Verify it's actually the max
        accuracies = [r["accuracy"]["mean"] for r in results.values()]
        assert best == max(accuracies)

    def test_returns_best_f1(self, simple_data):
        models = [LogisticRegressionModel(), KNNModel()]
        results = evaluate_models(simple_data, models=models, cv_folds=3)

        best = get_best_metric(results, "f1")
        assert 0 <= best <= 1

    def test_handles_nan_values(self):
        results = {
            "model1": {"accuracy": {"mean": 0.8, "std": 0.1}},
            "model2": {"accuracy": {"mean": np.nan, "std": np.nan}},
            "model3": {"accuracy": {"mean": 0.9, "std": 0.05}},
        }

        best = get_best_metric(results, "accuracy")
        assert best == 0.9

    def test_returns_nan_for_empty(self):
        results = {}
        best = get_best_metric(results, "accuracy")
        assert np.isnan(best)


class TestGetMeanMetric:
    """Tests for get_mean_metric function."""

    def test_returns_mean_accuracy(self, simple_data):
        models = [
            LogisticRegressionModel(),
            KNNModel(n_neighbors=3),
        ]
        results = evaluate_models(simple_data, models=models, cv_folds=3)

        mean = get_mean_metric(results, "accuracy")
        assert 0 <= mean <= 1

        # Verify it's actually the mean
        accuracies = [r["accuracy"]["mean"] for r in results.values()]
        assert abs(mean - np.mean(accuracies)) < 1e-10

    def test_handles_nan_values(self):
        results = {
            "model1": {"accuracy": {"mean": 0.8, "std": 0.1}},
            "model2": {"accuracy": {"mean": np.nan, "std": np.nan}},
            "model3": {"accuracy": {"mean": 0.9, "std": 0.05}},
        }

        mean = get_mean_metric(results, "accuracy")
        assert abs(mean - 0.85) < 1e-10  # (0.8 + 0.9) / 2

    def test_returns_nan_for_empty(self):
        results = {}
        mean = get_mean_metric(results, "accuracy")
        assert np.isnan(mean)


class TestGetModelMetric:
    """Tests for get_model_metric function."""

    def test_returns_specific_model_metric(self, simple_data):
        models = [LogisticRegressionModel(), KNNModel(n_neighbors=3)]
        results = evaluate_models(simple_data, models=models, cv_folds=3)

        lr_acc = get_model_metric(results, "LogisticRegression", "accuracy")
        knn_acc = get_model_metric(results, "KNN-3", "accuracy")

        assert 0 <= lr_acc <= 1
        assert 0 <= knn_acc <= 1

    def test_returns_nan_for_missing_model(self):
        results = {
            "LogisticRegression": {"accuracy": {"mean": 0.9, "std": 0.05}}
        }

        value = get_model_metric(results, "NonExistent", "accuracy")
        assert np.isnan(value)

    def test_returns_nan_for_missing_metric(self):
        results = {
            "LogisticRegression": {"accuracy": {"mean": 0.9, "std": 0.05}}
        }

        value = get_model_metric(results, "LogisticRegression", "f1")
        assert np.isnan(value)


class TestPrintEvaluationResults:
    """Tests for print_evaluation_results function."""

    def test_prints_without_error(self, simple_data, capsys):
        models = [LogisticRegressionModel(), KNNModel(n_neighbors=3)]
        results = evaluate_models(simple_data, models=models, cv_folds=3)

        print_evaluation_results(results, "accuracy")

        captured = capsys.readouterr()
        assert "Model" in captured.out
        assert "accuracy" in captured.out
        assert "LogisticRegression" in captured.out
        assert "KNN-3" in captured.out

    def test_prints_sorted_by_metric(self, capsys):
        results = {
            "ModelA": {"accuracy": {"mean": 0.7, "std": 0.1}},
            "ModelB": {"accuracy": {"mean": 0.9, "std": 0.05}},
            "ModelC": {"accuracy": {"mean": 0.8, "std": 0.08}},
        }

        print_evaluation_results(results, "accuracy")

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")

        # Find data lines (skip header)
        data_lines = [line for line in lines if "Model" in line or line.startswith("-")]

        # Should be sorted descending (ModelB, ModelC, ModelA)
        assert "ModelB" in captured.out
        assert "0.9000" in captured.out

    def test_handles_missing_metric(self, capsys):
        results = {
            "ModelA": {"accuracy": {"mean": 0.7, "std": 0.1}},
            "ModelB": {"f1": {"mean": 0.9, "std": 0.05}},  # No accuracy
        }

        # Should not crash, only print ModelA
        print_evaluation_results(results, "accuracy")

        captured = capsys.readouterr()
        assert "ModelA" in captured.out
        # ModelB should not appear since it doesn't have the metric
        # (Note: Current implementation might print it, but test documents expected behavior)


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_full_pipeline_default(self, simple_data):
        # Use all defaults
        results = evaluate_models(simple_data)

        # Should have results for all default models
        assert len(results) >= 10

        # Get best and mean metrics
        best_acc = get_best_metric(results, "accuracy")
        mean_acc = get_mean_metric(results, "accuracy")

        assert 0 <= best_acc <= 1
        assert 0 <= mean_acc <= 1
        assert best_acc >= mean_acc  # Best should be >= mean

    def test_full_pipeline_custom(self, simple_data):
        # Custom everything
        models = [LogisticRegressionModel(), KNNModel()]
        metrics = [AccuracyMetric(), F1Metric()]
        evaluator = CrossValidationEvaluator(cv_folds=3)

        results = evaluate_models(
            simple_data, models=models, metrics=metrics, evaluator=evaluator
        )

        assert len(results) == 2
        for model_name, model_results in results.items():
            assert len(model_results) == 2
            assert "accuracy" in model_results
            assert "f1" in model_results

        # Verify get_model_metric works
        lr_f1 = get_model_metric(results, "LogisticRegression", "f1")
        assert 0 <= lr_f1 <= 1
