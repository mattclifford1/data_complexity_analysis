"""
Pre-defined experiment configurations for common analysis scenarios.

Provides ready-to-use configurations for studying complexity vs ML performance
across different dataset types and parameter variations.
"""
from typing import Dict, List

from data_complexity.experiments.pipeline.utils import (
    DatasetSpec,
    ParameterSpec,
    ExperimentConfig,
)
from data_complexity.experiments.pipeline.experiment import Experiment
from data_complexity.experiments.pipeline.utils import PlotType


def datasets_from_sweep(
    base_spec: DatasetSpec,
    param_spec: ParameterSpec,
) -> List[DatasetSpec]:
    """
    Build a list of DatasetSpecs from a base spec and a parameter sweep.

    Each value in ``param_spec.values`` produces one DatasetSpec whose
    ``fixed_params`` extend the base spec's params with the swept parameter,
    and whose ``label`` is set by ``param_spec.format_label(value)``.

    Parameters
    ----------
    base_spec : DatasetSpec
        Base dataset specification (type and any fixed parameters).
    param_spec : ParameterSpec
        Parameter to sweep over (name, values, and label format).

    Returns
    -------
    list of DatasetSpec
        One spec per value in param_spec.values, ordered the same way.

    Examples
    --------
    >>> specs = datasets_from_sweep(
    ...     DatasetSpec("Moons", {}),
    ...     ParameterSpec("moons_noise", [0.05, 0.1, 0.2], "noise={value}"),
    ... )
    >>> [s.label for s in specs]
    ['noise=0.05', 'noise=0.1', 'noise=0.2']
    """
    specs = []
    for value in param_spec.values:
        params = dict(base_spec.fixed_params)
        params[param_spec.name] = value
        specs.append(DatasetSpec(
            dataset_type=base_spec.dataset_type,
            fixed_params=params,
            label=param_spec.format_label(value),
        ))
    return specs


def gaussian_variance_config() -> ExperimentConfig:
    """
    Configuration for studying Gaussian covariance scale (variance) effects.

    Varies cov_scale from 0.5 to 5.0 with fixed class separation.
    """
    return ExperimentConfig(
        datasets=datasets_from_sweep(
            DatasetSpec("Gaussian", {"class_separation": 4.0, "cov_type": "spherical"}),
            ParameterSpec("cov_scale", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0], "scale={value}"),
        ),
        x_label="cov_scale",
        name="gaussian_variance",
    )


def gaussian_separation_config() -> ExperimentConfig:
    """
    Configuration for studying Gaussian class separation effects.

    Varies class_separation from 1.0 to 8.0 with fixed variance.
    """
    return ExperimentConfig(
        datasets=datasets_from_sweep(
            DatasetSpec("Gaussian", {"cov_type": "spherical", "cov_scale": 1.0}),
            ParameterSpec("class_separation", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0], "sep={value}"),
        ),
        x_label="class_separation",
        name="gaussian_separation",
    )


def gaussian_correlation_config() -> ExperimentConfig:
    """
    Configuration for studying Gaussian feature correlation effects.

    Varies cov_correlation from -0.8 to 0.8 with symmetric covariance.
    """
    return ExperimentConfig(
        datasets=datasets_from_sweep(
            DatasetSpec("Gaussian", {"class_separation": 4.0, "cov_type": "symmetric"}),
            ParameterSpec("cov_correlation", [-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8], "corr={value}"),
        ),
        x_label="cov_correlation",
        name="gaussian_correlation",
    )


def gaussian_imbalance_config() -> ExperimentConfig:
    """
    Configuration for studying Gaussian class imbalance effects.

    Varies minority_reduce_scaler from 1 (balanced) to 16 (extreme imbalance).
    Imbalance is applied to the training set after the train/test split,
    so complexity metrics reflect the actual imbalanced training data.

    - 1 = balanced (50%-50%)
    - 2 = 67%-33%
    - 4 = 80%-20%
    - 8 = 89%-11%
    - 16 = 94%-6%
    """
    return ExperimentConfig(
        datasets=datasets_from_sweep(
            DatasetSpec("Gaussian", {"class_separation": 4.0, "cov_type": "spherical", "cov_scale": 1.0}),
            ParameterSpec("minority_reduce_scaler", [1, 2, 4, 8, 16], "imbalance={value}x"),
        ),
        x_label="minority_reduce_scaler",
        name="gaussian_imbalance",
    )


def moons_noise_config() -> ExperimentConfig:
    """
    Configuration for studying moons dataset noise effects.

    Varies noise from 0.05 to 0.5.
    """
    return ExperimentConfig(
        datasets=datasets_from_sweep(
            DatasetSpec("Moons", {}),
            ParameterSpec("moons_noise", [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5], "noise={value}"),
        ),
        x_label="moons_noise",
        name="moons_noise",
    )


def circles_noise_config() -> ExperimentConfig:
    """
    Configuration for studying circles dataset noise effects.

    Varies noise from 0.02 to 0.3.
    """
    return ExperimentConfig(
        datasets=datasets_from_sweep(
            DatasetSpec("Circles", {}),
            ParameterSpec("circles_noise", [0.02, 0.05, 0.1, 0.15, 0.2, 0.3], "noise={value}"),
        ),
        x_label="circles_noise",
        name="circles_noise",
    )


def blobs_features_config() -> ExperimentConfig:
    """
    Configuration for studying blobs dimensionality effects.

    Varies number of features from 2 to 20.
    """
    return ExperimentConfig(
        datasets=datasets_from_sweep(
            DatasetSpec("Blobs", {}),
            ParameterSpec("blobs_features", [2, 3, 5, 10, 15, 20], "d={value}"),
        ),
        x_label="blobs_features",
        name="blobs_features",
    )


EXPERIMENT_CONFIGS: Dict[str, callable] = {
    "gaussian_variance": gaussian_variance_config,
    "gaussian_separation": gaussian_separation_config,
    "gaussian_correlation": gaussian_correlation_config,
    "gaussian_imbalance": gaussian_imbalance_config,
    "moons_noise": moons_noise_config,
    "circles_noise": circles_noise_config,
    "blobs_features": blobs_features_config,
}


def get_config(name: str) -> ExperimentConfig:
    """
    Get a pre-defined experiment configuration by name.

    Parameters
    ----------
    name : str
        Configuration name (e.g., 'gaussian_variance', 'moons_noise').

    Returns
    -------
    ExperimentConfig
        The experiment configuration.

    Raises
    ------
    ValueError
        If configuration name is not recognized.
    """
    if name not in EXPERIMENT_CONFIGS:
        available = ", ".join(EXPERIMENT_CONFIGS.keys())
        raise ValueError(f"Unknown config: {name}. Available: {available}")
    return EXPERIMENT_CONFIGS[name]()


def list_configs() -> list:
    """
    List all available configuration names.

    Returns
    -------
    list of str
        Available configuration names.
    """
    return list(EXPERIMENT_CONFIGS.keys())


def run_experiment(name: str, verbose: bool = True, save: bool = True) -> Experiment:
    """
    Run a pre-defined experiment by name.

    Parameters
    ----------
    name : str
        Configuration name.
    verbose : bool
        Print progress. Default: True
    save : bool
        Save results to disk. Default: True

    Returns
    -------
    Experiment
        The completed experiment.
    """
    config = get_config(name)
    exp = Experiment(config)
    exp.run(verbose=verbose)
    exp.compute_distances()

    if save:
        exp.save()

    return exp


def run_all_experiments(verbose: bool = True, save: bool = True) -> Dict[str, Experiment]:
    """
    Run all pre-defined experiments.

    Parameters
    ----------
    verbose : bool
        Print progress. Default: True
    save : bool
        Save results to disk. Default: True

    Returns
    -------
    dict
        Configuration name -> completed Experiment.
    """
    experiments = {}

    for name in EXPERIMENT_CONFIGS.keys():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running experiment: {name}")
            print("=" * 60)

        experiments[name] = run_experiment(name, verbose=verbose, save=save)

    return experiments


if __name__ == "__main__":
    print("Available experiment configurations:")
    for name in list_configs():
        print(f"  - {name}")
