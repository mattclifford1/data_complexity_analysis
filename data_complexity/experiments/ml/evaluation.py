"""
Evaluation classes for ML model assessment.

Provides abstract base classes and concrete implementations for:
- Evaluators (cross-validation, train-test split)
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from .classification_metrics import (
    AbstractEvaluationMetric,
    get_default_metrics,
    get_metrics_dict,
)


# ============================================================================
# Evaluators
# ============================================================================


class AbstractEvaluator(ABC):
    """
    Abstract base class for model evaluators.

    Evaluators define how to train and assess models (e.g., cross-validation,
    train-test split).
    """

    @abstractmethod
    def evaluate(
        self,
        model,
        X,
        y,
        metrics: Optional[List[AbstractEvaluationMetric]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a model on data.

        Parameters
        ----------
        model : AbstractMLModel
            Model to evaluate.
        X : array-like, shape (n_samples, n_features)
            Feature matrix.
        y : array-like, shape (n_samples,)
            Target labels.
        metrics : list of AbstractEvaluationMetric, optional
            Metrics to compute. Default: get_default_metrics()

        Returns
        -------
        dict
            Metric name -> {'mean': float, 'std': float}
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class CrossValidationEvaluator(AbstractEvaluator):
    """
    Cross-validation evaluator using StratifiedKFold.

    Parameters
    ----------
    cv_folds : int, optional
        Number of cross-validation folds. Default: 5
    random_state : int, optional
        Random state for reproducibility. Default: 42
    """

    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state

    def evaluate(
        self,
        model,
        X,
        y,
        metrics: Optional[List[AbstractEvaluationMetric]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model using cross-validation.

        Parameters
        ----------
        model : AbstractMLModel
            Model to evaluate.
        X : array-like, shape (n_samples, n_features)
            Feature matrix.
        y : array-like, shape (n_samples,)
            Target labels.
        metrics : list of AbstractEvaluationMetric, optional
            Metrics to compute. Default: get_default_metrics()

        Returns
        -------
        dict
            Metric name -> {'mean': float, 'std': float}
        """
        if metrics is None:
            metrics = get_default_metrics()

        scoring = get_metrics_dict(metrics)

        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        # Create fresh model for cross-validation
        sklearn_model = model._create_model()

        try:
            cv_results = cross_validate(
                sklearn_model, X, y, cv=cv, scoring=scoring, return_train_score=False
            )

            results = {}
            for metric in metrics:
                scores = cv_results[f"test_{metric.name}"]
                results[metric.name] = {
                    "mean": scores.mean(),
                    "std": scores.std(),
                }

        except Exception as e:
            print(f"Warning: {model.name} evaluation failed: {e}")
            results = {
                metric.name: {"mean": np.nan, "std": np.nan} for metric in metrics
            }

        return results

    def __repr__(self):
        return f"CrossValidationEvaluator(cv_folds={self.cv_folds})"


class TrainTestSplitEvaluator(AbstractEvaluator):
    """
    Simple train-test split evaluator.

    Parameters
    ----------
    test_size : float, optional
        Fraction of data for test set. Default: 0.2
    random_state : int, optional
        Random state for reproducibility. Default: 42
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def evaluate(
        self,
        model,
        X,
        y,
        metrics: Optional[List[AbstractEvaluationMetric]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model using train-test split.

        Parameters
        ----------
        model : AbstractMLModel
            Model to evaluate.
        X : array-like, shape (n_samples, n_features)
            Feature matrix.
        y : array-like, shape (n_samples,)
            Target labels.
        metrics : list of AbstractEvaluationMetric, optional
            Metrics to compute. Default: get_default_metrics()

        Returns
        -------
        dict
            Metric name -> {'mean': float, 'std': 0.0} (std is 0 for single split)
        """
        if metrics is None:
            metrics = get_default_metrics()

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )

            # Create and train fresh model
            sklearn_model = model._create_model()
            sklearn_model.fit(X_train, y_train)
            y_pred = sklearn_model.predict(X_test)

            results = {}
            for metric in metrics:
                score = metric.compute(y_test, y_pred)
                results[metric.name] = {
                    "mean": score,
                    "std": 0.0,  # Single split has no std
                }

        except Exception as e:
            print(f"Warning: {model.name} evaluation failed: {e}")
            results = {
                metric.name: {"mean": np.nan, "std": np.nan} for metric in metrics
            }

        return results

    def __repr__(self):
        return f"TrainTestSplitEvaluator(test_size={self.test_size})"


# ============================================================================
# Evaluator Factory Functions
# ============================================================================


def get_default_evaluator(cv_folds: int = 5) -> CrossValidationEvaluator:
    """
    Return the default evaluator (cross-validation).

    Parameters
    ----------
    cv_folds : int, optional
        Number of cross-validation folds. Default: 5

    Returns
    -------
    CrossValidationEvaluator
        Default evaluator instance.
    """
    return CrossValidationEvaluator(cv_folds=cv_folds)
