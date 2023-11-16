from __future__ import annotations

from typing import TYPE_CHECKING

from gefest.core.geometry.datastructs.structure import Structure

if TYPE_CHECKING:
    from gefest.core.configs.optimization_params import OptimizationParams

from golem.core.optimisers.meta.surrogate_optimizer import SurrogateEachNgenOptimizer
from golem.core.optimisers.objective import Objective

from gefest.core.opt.adapters.configuration_mapping import (
    map_into_gpa,
    map_into_graph_generation_params,
    map_into_graph_requirements,
)
from gefest.tools.optimizers.optimizer import Optimizer


class SurrogateOptimizer(Optimizer):
    """Wrapper for GOLEM SurrogateEachNgenOptimizer.

    Provides optimization strategy using both physical simulator and surrogate model.

    """

    def __init__(self, opt_params: OptimizationParams, initial_population=None, **kwargs) -> None:
        super().__init__(opt_params.log_dispatcher, **kwargs)
        self.opt_params = opt_params
        self.objective = Objective(
            quality_metrics={obj.__class__.__name__: obj for obj in opt_params.objectives},
            is_multi_objective=len(opt_params.objectives) > 1,
        )
        self.requirements = map_into_graph_requirements(opt_params)
        self.ggp = map_into_graph_generation_params(opt_params)
        self.gpa = map_into_gpa(opt_params)

        if initial_population:
            self.initial_pop = initial_population
        else:
            self.initial_pop: list[Structure] = list(
                map(opt_params.golem_adapter.adapt, opt_params.sampler(opt_params.pop_size)),
            )

        self.__surrogate_opt = SurrogateEachNgenOptimizer(
            objective=self.objective,
            initial_graphs=self.initial_pop,
            requirements=self.requirements,
            graph_generation_params=self.ggp,
            graph_optimizer_params=self.gpa,
            surrogate_each_n_gen=self.opt_params.golem_surrogate_each_n_gen,
        )

    def optimize(self) -> list[Structure]:
        """Optimizes population.

        Returns:
            list[Structure]: Optimized population.
        """
        optimized_graphs = self.__surrogate_opt.optimise(self.objective)
        optimized_pop = list(map(self.opt_params.golem_adapter.restore, optimized_graphs))
        return optimized_pop
