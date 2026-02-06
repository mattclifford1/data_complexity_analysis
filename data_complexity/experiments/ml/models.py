"""
ML Model classes for classifier evaluation.

Provides an abstract base class and concrete implementations for various
sklearn classifiers with a unified interface for training and prediction.
"""
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class AbstractMLModel(ABC):
    """
    Abstract base class for ML models.

    Provides common functionality for creating, training, and predicting
    with sklearn classifiers. Models are pure wrappers around sklearn
    estimators without evaluation logic.

    Parameters
    ----------
    random_state : int, optional
        Random state for reproducibility. Default: 42
    scale_features : bool, optional
        Whether to scale features using StandardScaler. Default: False
    **kwargs
        Additional parameters passed to the underlying sklearn estimator.
    """

    def __init__(self, random_state=42, scale_features=False, **kwargs):
        self.random_state = random_state
        self.scale_features = scale_features
        self.model_params = kwargs
        self._model = None
        self._is_fitted = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    def _create_estimator(self):
        """
        Create and return the sklearn estimator.

        Returns
        -------
        estimator
            A sklearn-compatible estimator.
        """
        pass

    def _create_model(self):
        """Create the model, optionally with feature scaling pipeline."""
        estimator = self._create_estimator()
        if self.scale_features:
            return make_pipeline(StandardScaler(), estimator)
        return estimator

    @property
    def model(self):
        """Get the sklearn model, creating it if necessary."""
        if self._model is None:
            self._model = self._create_model()
        return self._model

    def fit(self, X, y):
        """
        Fit the model to training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature matrix.
        y : array-like, shape (n_samples,)
            Training labels.

        Returns
        -------
        self
        """
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        array-like, shape (n_samples,)
            Predicted labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.model.predict(X)

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


# ============================================================================
# Concrete Model Implementations
# ============================================================================


class LogisticRegressionModel(AbstractMLModel):
    """
    Logistic Regression classifier.

    Parameters
    ----------
    max_iter : int, optional
        Maximum iterations. Default: 1000
    **kwargs
        Additional parameters for LogisticRegression.
    """

    def __init__(self, max_iter=1000, **kwargs):
        super().__init__(scale_features=True, **kwargs)
        self.max_iter = max_iter

    @property
    def name(self) -> str:
        return "LogisticRegression"

    def _create_estimator(self):
        return LogisticRegression(
            max_iter=self.max_iter, random_state=self.random_state
        )


class KNNModel(AbstractMLModel):
    """
    K-Nearest Neighbors classifier.

    Parameters
    ----------
    n_neighbors : int, optional
        Number of neighbors. Default: 5
    **kwargs
        Additional parameters for KNeighborsClassifier.
    """

    def __init__(self, n_neighbors=5, **kwargs):
        super().__init__(scale_features=True, **kwargs)
        self.n_neighbors = n_neighbors

    @property
    def name(self) -> str:
        return f"KNN-{self.n_neighbors}"

    def _create_estimator(self):
        return KNeighborsClassifier(n_neighbors=self.n_neighbors)


class DecisionTreeModel(AbstractMLModel):
    """
    Decision Tree classifier.

    Parameters
    ----------
    max_depth : int, optional
        Maximum tree depth. Default: 5
    **kwargs
        Additional parameters for DecisionTreeClassifier.
    """

    def __init__(self, max_depth=5, **kwargs):
        super().__init__(scale_features=False, **kwargs)
        self.max_depth = max_depth

    @property
    def name(self) -> str:
        return "DecisionTree"

    def _create_estimator(self):
        return DecisionTreeClassifier(
            max_depth=self.max_depth, random_state=self.random_state
        )


class SVMModel(AbstractMLModel):
    """
    Support Vector Machine classifier.

    Parameters
    ----------
    kernel : str, optional
        Kernel type ('rbf', 'linear', 'poly'). Default: 'rbf'
    C : float, optional
        Regularization parameter. Default: 1.0
    **kwargs
        Additional parameters for SVC.
    """

    def __init__(self, kernel="rbf", C=1.0, **kwargs):
        super().__init__(scale_features=True, **kwargs)
        self.kernel = kernel
        self.C = C

    @property
    def name(self) -> str:
        return f"SVM-{self.kernel.upper()}"

    def _create_estimator(self):
        return SVC(kernel=self.kernel, C=self.C, random_state=self.random_state)


class RandomForestModel(AbstractMLModel):
    """
    Random Forest classifier.

    Parameters
    ----------
    n_estimators : int, optional
        Number of trees. Default: 50
    max_depth : int, optional
        Maximum tree depth. Default: 5
    **kwargs
        Additional parameters for RandomForestClassifier.
    """

    def __init__(self, n_estimators=50, max_depth=5, **kwargs):
        super().__init__(scale_features=False, **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    @property
    def name(self) -> str:
        return "RandomForest"

    def _create_estimator(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )


class GradientBoostingModel(AbstractMLModel):
    """
    Gradient Boosting classifier.

    Parameters
    ----------
    n_estimators : int, optional
        Number of boosting stages. Default: 50
    max_depth : int, optional
        Maximum tree depth. Default: 3
    **kwargs
        Additional parameters for GradientBoostingClassifier.
    """

    def __init__(self, n_estimators=50, max_depth=3, **kwargs):
        super().__init__(scale_features=False, **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    @property
    def name(self) -> str:
        return "GradientBoosting"

    def _create_estimator(self):
        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )


class NaiveBayesModel(AbstractMLModel):
    """
    Gaussian Naive Bayes classifier.

    Parameters
    ----------
    **kwargs
        Additional parameters for GaussianNB.
    """

    def __init__(self, **kwargs):
        super().__init__(scale_features=False, **kwargs)

    @property
    def name(self) -> str:
        return "NaiveBayes"

    def _create_estimator(self):
        return GaussianNB()


class MLPModel(AbstractMLModel):
    """
    Multi-Layer Perceptron classifier.

    Parameters
    ----------
    hidden_layer_sizes : tuple, optional
        Sizes of hidden layers. Default: (50,)
    max_iter : int, optional
        Maximum iterations. Default: 500
    **kwargs
        Additional parameters for MLPClassifier.
    """

    def __init__(self, hidden_layer_sizes=(50,), max_iter=500, **kwargs):
        super().__init__(scale_features=True, **kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter

    @property
    def name(self) -> str:
        return "MLP"

    def _create_estimator(self):
        return MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )


# ============================================================================
# Factory Functions
# ============================================================================


def get_default_models():
    """
    Return a list of default ML model instances.

    Returns
    -------
    list of AbstractMLModel
        Default set of models for benchmarking.
    """
    return [
        LogisticRegressionModel(),
        KNNModel(n_neighbors=5),
        KNNModel(n_neighbors=3),
        DecisionTreeModel(),
        SVMModel(kernel="rbf"),
        SVMModel(kernel="linear"),
        RandomForestModel(),
        GradientBoostingModel(),
        NaiveBayesModel(),
        MLPModel(),
    ]


def get_model_by_name(name, **kwargs):
    """
    Get a model instance by name.

    Parameters
    ----------
    name : str
        Model name (case-insensitive).
    **kwargs
        Parameters to pass to the model constructor.

    Returns
    -------
    AbstractMLModel
        Model instance.

    Raises
    ------
    ValueError
        If model name is not recognized.
    """
    name_lower = name.lower()
    model_map = {
        "logisticregression": LogisticRegressionModel,
        "logistic": LogisticRegressionModel,
        "knn": KNNModel,
        "decisiontree": DecisionTreeModel,
        "tree": DecisionTreeModel,
        "svm": SVMModel,
        "randomforest": RandomForestModel,
        "rf": RandomForestModel,
        "gradientboosting": GradientBoostingModel,
        "gb": GradientBoostingModel,
        "naivebayes": NaiveBayesModel,
        "nb": NaiveBayesModel,
        "mlp": MLPModel,
    }

    if name_lower not in model_map:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(model_map.keys())}"
        )

    return model_map[name_lower](**kwargs)
