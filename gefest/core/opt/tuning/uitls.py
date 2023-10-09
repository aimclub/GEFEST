from typing import Callable

from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain
from gefest.tools.fitness import Fitness

VarianceGeneratorType = Callable[[Structure], list[float]]


def fitness_validation_wrap(
    struct: Structure,
    fitness_fun: Fitness,
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
    if validator(struct):
        fitness = fitness_fun.fitness(struct)
    else:
        fitness = 1.e42
    return fitness


def average_edge_variance(
    structure: Structure,
    domain: Domain,
) -> list[float]:
    """Generates tuning variance for each point.
    Variance is equal to half the average edge length of the polygon.

    Returns:
        list[float]: list of variances for each point in structure
    """
    geom = domain.geometry
    variances = []
    for poly in structure:
        variances.extend(
            [0.5 * geom.get_length(poly) / (len(poly) - int(geom.is_closed))] * len(poly),
        )
    return variances
