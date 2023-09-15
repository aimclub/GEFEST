import copy
from functools import partial
from typing import Callable

import numpy as np

from gefest.core.geometry import Structure
from gefest.core.utils import chain, where
from gefest.core.utils.parallel_manager import BaseParallelDispatcher

from .strategy import Strategy


class CrossoverStrategy(Strategy):
    def __init__(self, opt_params):

        self.prob = opt_params.crossover_prob
        self.crossovers = opt_params.crossovers
        self.each_prob = opt_params.crossover_each_prob
        self.postprocess: Callable = opt_params.postprocessor
        self.parent_pairs_selector: Callable = opt_params.pair_selector
        self.sampler: Callable = opt_params.sampler
        self.postprocess_attempts = opt_params.postprocess_attempts
        self._pm = BaseParallelDispatcher(opt_params.n_jobs)

    def __call__(self, pop: list[Structure]) -> list[Structure]:
        return self.crossover(pop=pop)

    def crossover(self, pop: list[Structure]):

        chosen_crossover = np.random.choice(
            a=self.crossovers,
            size=1,
            p=self.each_prob,
        )[0]
        pairs = copy.deepcopy(self.parent_pairs_selector(pop))

        new_generation = self._pm.exec_parallel(
            func=chain(chosen_crossover, partial(self.postprocess, attempts=3)),
            arguments=pairs,
            use=True,
        )

        idx_failed = where(new_generation, lambda ind: ind is None)
        if len(idx_failed) > 0:
            generated = self.sampler(len(idx_failed))
            for enum_id, idx in enumerate(idx_failed):
                new_generation[idx] = generated[enum_id]

        pop.extend(new_generation)
        return pop
