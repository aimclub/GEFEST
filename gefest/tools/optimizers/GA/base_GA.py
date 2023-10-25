from random import randint
from typing import Callable

from tqdm import tqdm

from gefest.core.geometry import Structure
from gefest.core.opt import strategies
from gefest.core.utils.logger import LogDispatcher
from gefest.tools.objective import ObjectivesEvaluator
from gefest.tools.optimizers.optimizer import Optimizer


class BaseGA(Optimizer):
    def __init__(
        self,
        opt_params,
        **kwargs,
    ):
        super().__init__(opt_params.log_dispatcher)
        self.opt_params = opt_params
        self.crossover = getattr(strategies, opt_params.crossover_strategy)(opt_params=opt_params)
        self.mutation = getattr(strategies, opt_params.mutation_strategy)(opt_params=opt_params)
        self.sampler: Callable = opt_params.sampler
        self.objectives_evaluator: ObjectivesEvaluator = ObjectivesEvaluator(opt_params.objectives)
        self.selector: Callable = opt_params.selector
        self.pop_size = opt_params.pop_size
        self.n_steps = opt_params.n_steps
        self.domain = self.opt_params.domain
        self._pop: list[Structure] = self.sampler(self.opt_params.pop_size)
        self._pop = self.objectives_evaluator(self._pop)
        self.log_dispatcher.log_pop(self._pop, '00000_init')

    def optimize(self) -> list[Structure]:
        for step in tqdm(range(self.n_steps)):
            self._pop = self.crossover(self._pop)
            self._pop = self.mutation(self._pop)
            self._pop.extend(self.sampler(self.opt_params.extra))
            self._pop = self.objectives_evaluator(self._pop)
            self._pop = self.selector(self._pop, self.opt_params.pop_size)
            self.log_dispatcher.log_pop(self._pop, str(step + 1))
        return self._pop
