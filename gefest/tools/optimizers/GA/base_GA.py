from random import randint
from typing import Callable

from tqdm import tqdm

from gefest.core.geometry import Structure
from gefest.core.opt.strategies.strategy import Strategy
from gefest.tools.optimizers.optimizer import Optimizer
from refactored import LogDispatcher


class BaseGA(Optimizer):
    def __init__(
        self,
        opt_params,
        **kwargs,
    ):
        super().__init__(LogDispatcher())
        self.opt_params = opt_params
        self.crossover: Strategy = opt_params.crossover_strategy
        self.mutation: Strategy = opt_params.mutation_strategy
        self.sampler: Callable = opt_params.sampler
        self.estimator: Callable[[list[Structure]], list[Structure]] = opt_params.estimator
        self.selector: Callable = opt_params.selector
        self.pop_size = opt_params.pop_size
        self.n_steps = opt_params.n_steps
        self.domain = self.opt_params.domain
        self._pop: list[Structure] = self.sampler(self.opt_params.pop_size)
        self._pop = self.estimator(self._pop)
        self.logger.log_pop(self._pop, 'init')

    def optimize(self) -> list[Structure]:
        for step in tqdm(range(self.n_steps)):
            self._pop = self.selector(self._pop, self.opt_params.pop_size)
            self._pop = self.crossover(self._pop)
            self._pop = self.mutation(self._pop)
            self._pop.extend(self.sampler(5))
            self._pop = self.estimator(self._pop)
            self.logger.log_pop(self._pop, step)
        self._pop = sorted(self._pop, key=lambda x: x.fitness)
        return self._pop
