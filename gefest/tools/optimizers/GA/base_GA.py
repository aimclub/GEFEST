import copy
from random import randint
from typing import Callable

from tqdm import tqdm

from gefest.core.geometry import Structure
from gefest.core.opt import strategies
from gefest.core.utils.logger import LogDispatcher
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
        self.estimator: Callable[[list[Structure]], list[Structure]] = opt_params.estimator
        self.selector: Callable = opt_params.selector
        self.pop_size = opt_params.pop_size
        self.n_steps = opt_params.n_steps
        self.domain = self.opt_params.domain
        self._pop = self.sampler(self.opt_params.pop_size)
        print('Длина популяции инит', len(self._pop))
        self._pop = self.estimator(self._pop)
        #self._pop = sorted(self._pop,key = lambda x: x.fitness[0])
        self.log_dispatcher.log_pop(self._pop, '00000_init')

    def optimize(self) -> list[Structure]:
        for step in tqdm(range(self.n_steps)):
            best = copy.deepcopy(self._pop[0])
            self._pop = self.crossover(self._pop)
            print('Длина популяции после crossover', len(self._pop))
            self._pop = self.mutation(self._pop)
            print('Длина популяции после мутации',len(self._pop))
            self._pop.extend(self.sampler(self.opt_params.extra))
            print('Длина популяции после экстры',len(self._pop))
            self._pop = self.estimator(self._pop)
            self._pop = self.selector(self._pop, self.opt_params.pop_size)
            self._pop.append(best)
            #self._pop = sorted(self._pop, key=lambda x: x.fitness[0])
            print('Длина популяции после selector', len(self._pop))
            self.log_dispatcher.log_pop(self._pop, str(step + 1))
        return self._pop
