"""
Pre-defined experiment configurations for common analysis scenarios.

Provides ready-to-use configurations for studying complexity vs ML performance
across different dataset types and parameter variations.
"""
from typing import Dict

from data_complexity.experiments.experiment import (
    DatasetSpec,
    ParameterSpec,
    ExperimentConfig,
    Experiment,
    PlotType,
)


def gaussian_variance_config() -> ExperimentConfig:
    """
    Configuration for studying Gaussian covariance scale (variance) effects.

    Varies cov_scale from 0.5 to 5.0 with fixed class separation.
    """
    return ExperimentConfig(
        dataset=DatasetSpec(
            dataset_type="Gaussian",
            fixed_params={"class_separation": 4.0, "cov_type": "spherical"},
        ),
        vary_parameter=ParameterSpec(
            name="cov_scale",
            values=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
            label_format="scale={value}",
        ),
        name="gaussian_variance",
    )


def gaussian_separation_config() -> ExperimentConfig:
    """
    Configuration for studying Gaussian class separation effects.

    Varies class_separation from 1.0 to 8.0 with fixed variance.
    """
    return ExperimentConfig(
        dataset=DatasetSpec(
            dataset_type="Gaussian",
            fixed_params={"cov_type": "spherical", "cov_scale": 1.0},
        ),
        vary_parameter=ParameterSpec(
            name="class_separation",
            values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
            label_format="sep={value}",
        ),
        name="gaussian_separation",
    )


def gaussian_correlation_config() -> ExperimentConfig:
    """
    Configuration for studying Gaussian feature correlation effects.

    Varies cov_correlation from -0.8 to 0.8 with symmetric covariance.
    """
    return ExperimentConfig(
        dataset=DatasetSpec(
            dataset_type="Gaussian",
            fixed_params={"class_separation": 4.0, "cov_type": "symmetric"},
        ),
        vary_parameter=ParameterSpec(
            name="cov_correlation",
            values=[-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8],
            label_format="corr={value}",
        ),
        name="gaussian_correlation",
    )


def moons_noise_config() -> ExperimentConfig:
    """
    Configuration for studying moons dataset noise effects.

    Varies noise from 0.05 to 0.5.
    """
    return ExperimentConfig(
        dataset=DatasetSpec(
            dataset_type="Moons",
            fixed_params={},
        ),
        vary_parameter=ParameterSpec(
            name="moons_noise",
            values=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
            label_format="noise={value}",
        ),
        name="moons_noise",
    )


def circles_noise_config() -> ExperimentConfig:
    """
    Configuration for studying circles dataset noise effects.

    Varies noise from 0.02 to 0.3.
    """
    return ExperimentConfig(
        dataset=DatasetSpec(
            dataset_type="Circles",
            fixed_params={},
        ),
        vary_parameter=ParameterSpec(
            name="circles_noise",
            values=[0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
            label_format="noise={value}",
        ),
        name="circles_noise",
    )


def blobs_features_config() -> ExperimentConfig:
    """
    Configuration for studying blobs dimensionality effects.

    Varies number of features from 2 to 20.
    """
    return ExperimentConfig(
        dataset=DatasetSpec(
            dataset_type="Blobs",
            fixed_params={},
        ),
        vary_parameter=ParameterSpec(
            name="blobs_features",
            values=[2, 3, 5, 10, 15, 20],
            label_format="d={value}",
        ),
        name="blobs_features",
    )


EXPERIMENT_CONFIGS: Dict[str, callable] = {
    "gaussian_variance": gaussian_variance_config,
    "gaussian_separation": gaussian_separation_config,
    "gaussian_correlation": gaussian_correlation_config,
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
    exp.compute_correlations()

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
