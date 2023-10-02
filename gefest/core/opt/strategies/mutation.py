import copy
from functools import partial
from typing import Callable

from gefest.core.geometry import Structure
from gefest.core.opt.operators.mutations import mutate_structure
from gefest.core.utils import chained_call, where
from gefest.core.utils.parallel_manager import BaseParallelDispatcher

from .strategy import Strategy


class MutationStrategy(Strategy):
    def __init__(self, opt_params):

        self.domain = opt_params.domain
        self.mutation_prob = opt_params.mutation_prob
        self.mutations = opt_params.mutations
        self.each_prob = opt_params.mutation_each_prob
        self.postprocess: Callable = opt_params.postprocessor
        self.sampler = opt_params.sampler
        self.postprocess_attempts = opt_params.postprocess_attempts
        self._pm = BaseParallelDispatcher(opt_params.n_jobs)

    def __call__(self, pop: list[Structure]) -> list[Structure]:
        return self.mutate(pop=pop)

    def mutate(self, pop: list[Structure]):

        mutator = partial(
            mutate_structure,
            domain=self.domain,
            mutations=self.mutations,
            mutation_chance=self.mutation_prob,
            mutations_probs=self.each_prob,
        )
        pop_ = copy.deepcopy(pop)

        mutated_pop = self._pm.exec_parallel(
            func=chained_call(mutator, partial(self.postprocess, attempts=3)),
            arguments=pop_,
            use=True,
        )

        idx_failed = where(mutated_pop, lambda ind: ind is None)
        if len(idx_failed) > 0:
            generated = self.sampler(len(idx_failed))
            for enum_id, idx in enumerate(idx_failed):
                mutated_pop[idx] = generated[enum_id]

        return mutated_pop
