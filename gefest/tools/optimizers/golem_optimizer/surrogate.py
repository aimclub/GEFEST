from golem.core.optimisers.meta.surrogate_optimizer import SurrogateEachNgenOptimizer
from golem.core.optimisers.objective import Objective

from gefest.core.configs.optimization_params import OptimizationParams
from gefest.core.opt.adapters.configuration_mapping import (
    map_into_gpa,
    map_into_graph_generation_params,
    map_into_graph_requirements,
)
from gefest.tools.optimizers.optimizer import Optimizer


class SurrogateOptimizer(Optimizer):
    def __init__(self, opt_params: OptimizationParams, **kwargs) -> None:
        super().__init__(opt_params.log_dispatcher, **kwargs)
        self.objective = Objective(
            quality_metrics={obj.__class__.__name__: obj for obj in opt_params.objectives},
            is_multi_objective=len(opt_params.objectives) > 1,
        )
        self.requirements = map_into_graph_requirements(opt_params)
        self.ggp = map_into_graph_generation_params(opt_params)
        self.gpa = map_into_gpa(opt_params)
        self.initial_pop = list(
            map(opt_params.golem_adapter.adapt, opt_params.sampler(opt_params.pop_size)),
        )
        self.__surrogate_opt = SurrogateEachNgenOptimizer(
            objective=self.objective,
            initial_graphs=self.initial_pop,
            requirements=self.requirements,
            graph_generation_params=self.ggp,
            graph_optimizer_params=self.gpa,
            surrogate_each_n_gen=5,  # make surrogate config
        )

    def optimize(self):
        return self.__surrogate_opt.optimise(self.objective)
