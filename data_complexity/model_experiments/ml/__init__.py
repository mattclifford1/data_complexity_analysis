"""
ML model evaluation module.

Provides model classes, evaluation metrics, evaluators, and orchestration
functions for assessing classifier performance on datasets.
"""

# Model classes
from .models import (
    AbstractMLModel,
    LogisticRegressionModel,
    KNNModel,
    DecisionTreeModel,
    SVMModel,
    RandomForestModel,
    GradientBoostingModel,
    NaiveBayesModel,
    MLPModel,
    get_default_models,
    get_model_by_name,
)

# Classification metrics
from .classification_metrics import (
    AbstractEvaluationMetric,
    AccuracyMetric,
    AccuracyBalancedMetric,
    AccuracyMinorityMetric,
    AccuracyMajorityMetric,
    GeometricMeanMetric,
    GeometricMeanWeightedMetric,
    F1Metric,
    F1WeightedMetric,
    PrecisionMetric,
    Precision0Metric,
    Precision1Metric,
    PrecisionWeightedMetric,
    FScoreMetric,
    RecallMetric,
    RecallWeightedMetric,
    AucMetric,
    RocAucMetric,
    get_default_metrics,
    get_metrics_dict,
    get_metric_by_name,
    get_metrics_from_names,
)

# Evaluators
from .evaluation import (
    AbstractEvaluator,
    CrossValidationEvaluator,
    TrainTestSplitEvaluator,
    get_default_evaluator,
)

# Orchestration functions
from .model_pipeline import (
    evaluate_models,
    evaluate_models_train_test,
    evaluate_single_model,
    get_best_metric,
    get_mean_metric,
    get_model_metric,
    print_evaluation_results,
)

__all__ = [
    # Models
    "AbstractMLModel",
    "LogisticRegressionModel",
    "KNNModel",
    "DecisionTreeModel",
    "SVMModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "NaiveBayesModel",
    "MLPModel",
    "get_default_models",
    "get_model_by_name",

    # Metrics
    "AbstractEvaluationMetric",
    "AccuracyMetric",
    "AccuracyBalancedMetric",
    "AccuracyMinorityMetric",
    "AccuracyMajorityMetric",
    "GeometricMeanMetric",
    "GeometricMeanWeightedMetric",
    "F1Metric",
    "F1WeightedMetric",
    "PrecisionMetric",
    "Precision0Metric",
    "Precision1Metric",
    "PrecisionWeightedMetric",
    "FScoreMetric",
    "RecallMetric",
    "RecallWeightedMetric",
    "AucMetric",
    "RocAucMetric",
    "get_default_metrics",
    "get_metrics_dict",
    "get_metric_by_name",
    "get_metrics_from_names",

    # Evaluators
    "AbstractEvaluator",
    "CrossValidationEvaluator",
    "TrainTestSplitEvaluator",
    "get_default_evaluator",

    # Pipeline
    "evaluate_models",
    "evaluate_models_train_test",
    "evaluate_single_model",
    "get_best_metric",
    "get_mean_metric",
    "get_model_metric",
    "print_evaluation_results",
]
