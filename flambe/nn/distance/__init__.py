# type: ignore[attr-defined]

from flambe.nn.distance.distance import DistanceModule, MeanModule
from flambe.nn.distance.euclidean import EuclideanDistance, EuclideanMean
from flambe.nn.distance.cosine import CosineDistance, CosineMean
from flambe.nn.distance.hyperbolic import HyperbolicDistance, HyperbolicMean


def get_distance_module(metric: str) -> DistanceModule:
    """Get the distance module from a string alias.

    Currently available:
    . `euclidean`
    . `cosine`
    . `hyperbolic`

    Parameters
    ----------
    metric : str
        The distance metric to use

    Raises
    ------
    ValueError
        Unvalid distance string alias provided

    Returns
    -------
    DistanceModule
        The instantiated distance module

    """
    if metric == 'euclidean':
        module = EuclideanDistance()
    elif metric == 'cosine':
        module = CosineDistance()
    elif metric == 'hyperbolic':
        module = HyperbolicDistance()
    else:
        raise ValueError(f"Unknown distance alias: {metric}")

    return module


def get_mean_module(metric: str) -> MeanModule:
    """Get the mean module from a string alias.

    Currently available:
    . `euclidean`
    . `cosine`
    . `hyperbolic`

    Parameters
    ----------
    metric : str
        The distance metric to use

    Raises
    ------
    ValueError
        Unvalid distance string alias provided

    Returns
    -------
    DistanceModule
        The instantiated distance module

    """
    if metric == 'euclidean':
        module = EuclideanMean()
    elif metric == 'cosine':
        module = CosineMean()
    elif metric == 'hyperbolic':
        module = HyperbolicMean()
    else:
        raise ValueError(f"Unknown distance alias: {metric}")

    return module
