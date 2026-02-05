"""Tests for ML model classes."""
import pytest
import numpy as np
from sklearn.datasets import make_classification

from data_complexity.experiments.ml_models import (
    AbstractMLModel,
    LogisticRegressionModel,
    KNNModel,
    DecisionTreeModel,
    SVMModel,
    RandomForestModel,
    GradientBoostingModel,
    NaiveBayesModel,
    MLPModel,
    SCORING_METRICS,
    get_default_models,
    get_model_by_name,
    evaluate_models,
    get_best_metric,
    get_mean_metric,
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


@pytest.fixture
def simple_xy():
    """Generate simple X, y arrays."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )
    return X, y


class TestAbstractMLModel:
    """Tests for AbstractMLModel base class."""

    def test_cannot_instantiate_abstract_class(self):
        """AbstractMLModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractMLModel()

    def test_subclass_must_implement_name(self, simple_data):
        """Subclass must implement name property."""

        class IncompleteModel(AbstractMLModel):
            def _create_estimator(self):
                return None

        with pytest.raises(TypeError):
            IncompleteModel()

    def test_subclass_must_implement_create_estimator(self, simple_data):
        """Subclass must implement _create_estimator method."""

        class IncompleteModel(AbstractMLModel):
            @property
            def name(self):
                return "Incomplete"

        with pytest.raises(TypeError):
            IncompleteModel()


class TestLogisticRegressionModel:
    """Tests for LogisticRegressionModel."""

    def test_name(self):
        model = LogisticRegressionModel()
        assert model.name == "LogisticRegression"

    def test_evaluate(self, simple_data):
        model = LogisticRegressionModel()
        metrics = model.evaluate(simple_data, cv_folds=3)

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"]["mean"] <= 1
        assert metrics["accuracy"]["std"] >= 0

    def test_fit_predict(self, simple_xy):
        X, y = simple_xy
        model = LogisticRegressionModel()
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(y)

    def test_get_metric(self, simple_data):
        model = LogisticRegressionModel()
        model.evaluate(simple_data, cv_folds=3)

        acc = model.get_metric("accuracy")
        assert 0 <= acc <= 1


class TestKNNModel:
    """Tests for KNNModel."""

    def test_name_includes_neighbors(self):
        model3 = KNNModel(n_neighbors=3)
        model5 = KNNModel(n_neighbors=5)

        assert model3.name == "KNN-3"
        assert model5.name == "KNN-5"

    def test_evaluate(self, simple_data):
        model = KNNModel(n_neighbors=3)
        metrics = model.evaluate(simple_data, cv_folds=3)

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"]["mean"] <= 1


class TestDecisionTreeModel:
    """Tests for DecisionTreeModel."""

    def test_name(self):
        model = DecisionTreeModel()
        assert model.name == "DecisionTree"

    def test_no_scaling(self):
        model = DecisionTreeModel()
        assert model.scale_features is False

    def test_evaluate(self, simple_data):
        model = DecisionTreeModel(max_depth=3)
        metrics = model.evaluate(simple_data, cv_folds=3)

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"]["mean"] <= 1


class TestSVMModel:
    """Tests for SVMModel."""

    def test_name_includes_kernel(self):
        rbf = SVMModel(kernel="rbf")
        linear = SVMModel(kernel="linear")

        assert rbf.name == "SVM-RBF"
        assert linear.name == "SVM-LINEAR"

    def test_evaluate(self, simple_data):
        model = SVMModel(kernel="rbf")
        metrics = model.evaluate(simple_data, cv_folds=3)

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"]["mean"] <= 1


class TestRandomForestModel:
    """Tests for RandomForestModel."""

    def test_name(self):
        model = RandomForestModel()
        assert model.name == "RandomForest"

    def test_evaluate(self, simple_data):
        model = RandomForestModel(n_estimators=10, max_depth=3)
        metrics = model.evaluate(simple_data, cv_folds=3)

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"]["mean"] <= 1


class TestGradientBoostingModel:
    """Tests for GradientBoostingModel."""

    def test_name(self):
        model = GradientBoostingModel()
        assert model.name == "GradientBoosting"

    def test_evaluate(self, simple_data):
        model = GradientBoostingModel(n_estimators=10, max_depth=2)
        metrics = model.evaluate(simple_data, cv_folds=3)

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"]["mean"] <= 1


class TestNaiveBayesModel:
    """Tests for NaiveBayesModel."""

    def test_name(self):
        model = NaiveBayesModel()
        assert model.name == "NaiveBayes"

    def test_no_scaling(self):
        model = NaiveBayesModel()
        assert model.scale_features is False

    def test_evaluate(self, simple_data):
        model = NaiveBayesModel()
        metrics = model.evaluate(simple_data, cv_folds=3)

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"]["mean"] <= 1


class TestMLPModel:
    """Tests for MLPModel."""

    def test_name(self):
        model = MLPModel()
        assert model.name == "MLP"

    def test_evaluate(self, simple_data):
        model = MLPModel(hidden_layer_sizes=(10,), max_iter=100)
        metrics = model.evaluate(simple_data, cv_folds=3)

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"]["mean"] <= 1


class TestFactoryFunctions:
    """Tests for factory and utility functions."""

    def test_get_default_models(self):
        models = get_default_models()

        assert len(models) == 10
        assert all(isinstance(m, AbstractMLModel) for m in models)

        names = [m.name for m in models]
        assert "LogisticRegression" in names
        assert "KNN-5" in names
        assert "SVM-RBF" in names

    def test_get_model_by_name(self):
        model = get_model_by_name("logisticregression")
        assert isinstance(model, LogisticRegressionModel)

        model = get_model_by_name("knn", n_neighbors=7)
        assert isinstance(model, KNNModel)
        assert model.n_neighbors == 7

        model = get_model_by_name("svm", kernel="linear")
        assert isinstance(model, SVMModel)
        assert model.kernel == "linear"

    def test_get_model_by_name_unknown(self):
        with pytest.raises(ValueError):
            get_model_by_name("unknown_model")

    def test_evaluate_models(self, simple_data):
        models = [
            LogisticRegressionModel(),
            KNNModel(n_neighbors=3),
        ]
        results = evaluate_models(simple_data, models=models, cv_folds=3)

        assert "LogisticRegression" in results
        assert "KNN-3" in results
        assert "accuracy" in results["LogisticRegression"]

    def test_get_best_metric(self, simple_data):
        models = get_default_models()[:3]  # First 3 models for speed
        results = evaluate_models(simple_data, models=models, cv_folds=3)

        best = get_best_metric(results, "accuracy")
        assert 0 <= best <= 1

    def test_get_mean_metric(self, simple_data):
        models = get_default_models()[:3]
        results = evaluate_models(simple_data, models=models, cv_folds=3)

        mean = get_mean_metric(results, "accuracy")
        assert 0 <= mean <= 1


class TestModelRepr:
    """Tests for model string representation."""

    def test_repr(self):
        model = LogisticRegressionModel()
        assert "LogisticRegressionModel" in repr(model)
        assert "LogisticRegression" in repr(model)


class TestPredictBeforeFit:
    """Tests for prediction error handling."""

    def test_predict_before_fit_raises(self, simple_xy):
        X, y = simple_xy
        model = LogisticRegressionModel()

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(X)
