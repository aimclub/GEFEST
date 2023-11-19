from typing import Callable

from hyperopt import hp

from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain

VarianceGeneratorType = Callable[[Structure], list[float]]


def _get_uniform_args(mode: float, variance: float) -> tuple[float, float]:
    return (mode - variance, mode + variance)


def _get_norm_args(mode: float, variance: float) -> tuple[float, float]:
    return (mode, variance)


def average_edge_variance(
    structure: Structure,
    domain: Domain,
    distrib: Callable,
) -> list[float]:
    """Generates tuning variance for each point.

    Variance is equal to half the average edge length of the polygon.

    Returns:
        list[float]: list of variances for each point in structure
    """
    if distrib is hp.uniform:
        get_args = _get_uniform_args
    elif distrib is hp.normal:
        get_args = _get_norm_args
    else:
        raise ValueError(
            f'Invalid distribution function: {distrib}, only hp.uniform and hp.normal allowed.',
        )

    geom = domain.geometry
    variances = []
    for poly in structure:
        avg = 0.5 * geom.get_length(poly) / (len(poly) - int(geom.is_closed))
        for point in poly:
            for coord in point.coords:
                variances.append(get_args(coord, avg))

    return variances
