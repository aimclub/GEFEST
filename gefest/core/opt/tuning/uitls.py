from typing import Any, Callable, Dict, Optional, Union

from golem.core.dag.graph import Graph
from golem.core.optimisers.fitness import Fitness, MultiObjFitness, SingleObjFitness
from golem.core.optimisers.objective.objective import Objective as GolemObjective
from hyperopt import hp

from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.opt.adapters.structure import StructureAdapter
from gefest.tools.objective import Objective, ObjectivesEvaluator

VarianceGeneratorType = Callable[[Structure], list[float]]


class GolemObjectiveWithPreValidation(GolemObjective):
    def __init__(
        self,
        quality_metrics: Union[Callable, Dict[Any, Callable]],
        validator: Callable,
        adapter: StructureAdapter,
        complexity_metrics: Optional[Dict[Any, Callable]] = None,
        is_multi_objective: bool = False,
    ) -> None:
        super().__init__(quality_metrics, complexity_metrics, is_multi_objective)
        self.validator = validator
        self.adapter = adapter

    def __call__(self, graph: Graph, **metrics_kwargs: Any) -> Fitness:
        if self.validator(self.adapter.restore(graph)):
            res = super().__call__(graph, **metrics_kwargs)
            return res
        else:
            if self.is_multi_objective:
                return MultiObjFitness(values=[1.0e42] * len(self.quality_metrics), weights=1.0)
            else:
                return SingleObjFitness(1.0e42)


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
        raise ValueError('Invalin distribution function, only hp.uniform and hp.normal allowed.')
    geom = domain.geometry
    variances = []
    for poly in structure:
        avg = 0.5 * geom.get_length(poly) / (len(poly) - int(geom.is_closed))
        for point in poly:
            for coord in point.coords:
                variances.append(get_args(coord, avg))
    return variances
