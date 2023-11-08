from typing import Any, Callable, Dict, Optional, Union

from golem.core.dag.graph import Graph
from golem.core.optimisers.fitness import Fitness, MultiObjFitness, SingleObjFitness
from golem.core.optimisers.objective.objective import Objective as GolemObjective

from gefest.core.opt.adapters.structure import StructureAdapter


class GolemObjectiveWithPreValidation(GolemObjective):
    """GOLEM objective with GEFEST validation filtering."""

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
        """Evaluates objective for GOLEM graph representtion of GEFEST structure.

        If structure invalid, fintes will be set to high value.
        This class allows filter out invalid variants on tuning.

        """
        if self.validator(self.adapter.restore(graph)):
            res = super().__call__(graph, **metrics_kwargs)
            return res
        else:
            if self.is_multi_objective:
                return MultiObjFitness(values=[1.0e42] * len(self.quality_metrics), weights=1.0)
            else:
                return SingleObjFitness(1.0e42)
