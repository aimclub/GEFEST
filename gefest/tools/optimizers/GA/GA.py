from typing import Callable
from venv import logger

from tqdm import tqdm

from gefest.core.geometry import Structure
from gefest.core.opt import strategies
from gefest.core.opt.objective.objective_eval import ObjectivesEvaluator
from gefest.tools.optimizers.optimizer import Optimizer


class BaseGA(Optimizer):
    """Implemets default genetic optimization algorithm steps.

    Can be used as base class for others GA based optimizers.
    Can be configured using modules with different realization of operations,
    e.g. crossover, mutation, selection operations or crossover, mutation strategies.

    """

    def __init__(
        self,
        opt_params,
        initial_population=None,
        **kwargs,
    ):
        super().__init__(opt_params.log_dispatcher)
        self.opt_params = opt_params
        self.crossover = getattr(strategies, opt_params.crossover_strategy)(opt_params=opt_params)
        self.mutation = getattr(strategies, opt_params.mutation_strategy)(opt_params=opt_params)
        self.sampler: Callable = opt_params.sampler
        self.objectives_evaluator: ObjectivesEvaluator = ObjectivesEvaluator(opt_params.objectives)
        self.pop_size = opt_params.pop_size
        self.n_steps = opt_params.n_steps
        self.domain = self.opt_params.domain
        if initial_population:
            self._pop = initial_population
        else:
            self._pop: list[Structure] = self.sampler(self.opt_params.pop_size)

        self._pop = self.objectives_evaluator(self._pop)

        self.selector: Callable = opt_params.selector.value
        if len(self.opt_params.objectives) > 1:
            if self.opt_params.multiobjective_selector.__name__ == 'MOEAD':
                self.opt_params.extra = 0
                logger.warning('For moead extra not available.')

        self.selector = self.opt_params.multiobjective_selector(
            single_demention_selection=self.selector,
            init_pop=self._pop,
            moead_n_neighbors=self.opt_params.moead_multi_objective_selector_neighbors,
            steps=self.n_steps,
        )

        self.log_dispatcher.log_pop(self._pop, '00000_init')

    def optimize(self) -> list[Structure]:
        """Optimizes population.

        Returns:
            list[Structure]: Optimized population.
        """
        for step in tqdm(range(self.n_steps)):
            self._pop = self.crossover(self._pop)
            self._pop = self.mutation(self._pop)
            self._pop.extend(self.sampler(self.opt_params.extra))
            self._pop = self.objectives_evaluator(self._pop)
            self._pop = self.selector(self._pop, self.pop_size)
            self.log_dispatcher.log_pop(self._pop, str(step + 1))

        return self._pop
