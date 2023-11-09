from typing import Callable

from hyperopt import hp

from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.opt.objective.objective import Objective

VarianceGeneratorType = Callable[[Structure], list[float]]


def objective_validation_wrap(
    struct: Structure,
    objectives: Objective,
    validator: Callable,
) -> float:
    """Applys validation rules to structure.

    Used for GOLEM tuner as objective to filter out
    invalid cases in tuning process.

    Args:
        struct (Structure): Changed stucture.
        fitness_fun (Fitness): Fitness instance for current task.
        validator (Callable): Util for structure validation.

    Returns:
        float: None if structure invalid else fintess function output
    """


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
            f'Invalin distribution function: {distrib}, only hp.uniform and hp.normal allowed.'
        )

    geom = domain.geometry
    variances = []
    for poly in structure:
        avg = 0.5 * geom.get_length(poly) / (len(poly) - int(geom.is_closed))
        for point in poly:
            for coord in point.coords:
                variances.append(get_args(coord, avg))

    return variances