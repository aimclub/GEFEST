from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gefest.core.configs.optimization_params import OptimizationParams

import copy
from functools import partial
from typing import Callable

from gefest.core.geometry import Structure
from gefest.core.opt.operators.mutations import mutate_structure
from gefest.core.utils import where
from gefest.core.utils.parallel_manager import BaseParallelDispatcher

from .strategy import Strategy


class MutationStrategy(Strategy):
    """Default mutation strategy."""

    def __init__(self, opt_params: OptimizationParams):

        self.domain = opt_params.domain
        self.mutation_chance = opt_params.mutation_prob
        self.mutations = opt_params.mutations
        self.mutations_probs = opt_params.mutation_each_prob
        self.postprocess: Callable = opt_params.postprocessor
        self.sampler = opt_params.sampler
        self.postprocess_attempts = opt_params.postprocess_attempts
        self._pm = BaseParallelDispatcher(opt_params.n_jobs)

    def __call__(self, pop: list[Structure]) -> list[Structure]:
        """Calls mutate method."""
        return self.mutate(pop=pop)

    def mutate(self, pop: list[Structure]) -> list[Structure]:
        """Mutates provided population.

        Args:
            pop (list[Structure]): Given population.

        Returns:
            list[Structure]: Mutated population.
        """
        mutator = partial(
            mutate_structure,
            domain=self.domain,
            operations=self.mutations,
            operation_chance=self.mutation_chance,
            operations_probs=self.mutations_probs,
        )
        pop_ = copy.deepcopy(pop)

        mutated_pop = self._pm.exec_parallel(
            func=mutator,
            arguments=[(ind,) for ind in pop_],
            use=True,
            flatten=False,
        )

        mutated_pop = self._pm.exec_parallel(
            func=partial(self.postprocess, attempts=3),
            arguments=[(ind,) for ind in mutated_pop],
            use=True,
            flatten=True,
        )

        idx_failed = where(mutated_pop, lambda ind: ind is None)
        if len(idx_failed) > 0:
            generated = self.sampler(len(idx_failed))
            for enum_id, idx in enumerate(idx_failed):
                mutated_pop[idx] = generated[enum_id]

        return mutated_pop
