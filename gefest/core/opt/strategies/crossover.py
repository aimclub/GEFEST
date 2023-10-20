import copy
from functools import partial
from typing import Callable

from gefest.core.configs.optimization_params import OptimizationParams
from gefest.core.geometry import Structure
from gefest.core.opt.operators.crossovers import crossover_structures
from gefest.core.utils import where
from gefest.core.utils.parallel_manager import BaseParallelDispatcher

from .strategy import Strategy


class CrossoverStrategy(Strategy):
    def __init__(self, opt_params: OptimizationParams):

        self.prob = opt_params.crossover_prob
        self.crossovers = opt_params.crossovers
        self.crossovers_probs = opt_params.crossover_each_prob
        self.crossover_chacne = opt_params.crossover_prob
        self.postprocess: Callable = opt_params.postprocessor
        self.parent_pairs_selector: Callable = opt_params.pair_selector
        self.sampler: Callable = opt_params.sampler
        self.domain = opt_params.domain
        self.postprocess_attempts = opt_params.postprocess_attempts
        self._pm = BaseParallelDispatcher(opt_params.n_jobs)

    def __call__(self, pop: list[Structure]) -> list[Structure]:
        return self.crossover(pop=pop)

    def crossover(self, pop: list[Structure]):

        crossover = partial(
            crossover_structures,
            domain=self.domain,
            operations=self.crossovers,
            operation_chance=self.crossover_chacne,
            operations_probs=self.crossovers_probs,
        )

        pairs = copy.deepcopy(self.parent_pairs_selector(pop))

        new_generation = self._pm.exec_parallel(
            func=crossover,
            arguments=pairs,
            use=True,
        )
        new_generation = self._pm.exec_parallel(
            func=self.postprocess,
            arguments=[(ind,) for ind in new_generation],
            use=True,
            flatten=True,
        )

        idx_failed = where(new_generation, lambda ind: ind is None)
        if len(idx_failed) > 0:
            generated = self.sampler(len(idx_failed))
            for enum_id, idx in enumerate(idx_failed):
                new_generation[idx] = generated[enum_id]
        pop =new_generation
        #pop.extend(new_generation)
        return pop
