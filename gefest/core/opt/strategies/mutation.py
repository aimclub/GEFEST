import copy
from functools import partial
from typing import Callable

from gefest.core.geometry import Structure
from gefest.core.opt.operators.mutations import mutate_structure
from gefest.core.utils import WorkerData, where

from .strategy import Strategy


class MutationStrategy(Strategy):
    def __init__(self, opt_params):
        super().__init__(opt_params.workers_manager)
        self.domain = opt_params.domain
        self.mutation_prob = opt_params.mutation_prob
        self.mutations = opt_params.mutations
        self.each_prob = opt_params.mutation_each_prob
        self.postprocess: Callable = opt_params.postprocessor
        self.sampler = opt_params.sampler
        self.postprocess_attempts = 3

    def __call__(self, pop: list[Structure]) -> list[Structure]:
        return self.mutate(pop=pop)

    def mutate(self, pop: list[Structure]):

        mutated_pop = copy.deepcopy(pop)
        mutator = partial(
            mutate_structure,
            domain=self.domain,
            mutations=self.mutations,
            mutation_chance=self.mutation_prob,
            mutations_probs=self.each_prob,
        )
        chosen_mutations = [(mutator, self.postprocess) for _ in range(len(pop))]

        mutated, _ = self._mp(
            [
                WorkerData(funcs, idx, args)
                for funcs, idx, args in zip(
                    chosen_mutations,
                    range(len(pop)),
                    pop,
                )
            ],
        )

        succes_mutated_ids = where(mutated, lambda ind: ind != None)
        for idx in succes_mutated_ids:
            mutated_pop[idx] = mutated[idx]

        for _ in range(self.postprocess_attempts):
            failed_idx = where(mutated, lambda ind: ind == None)
            if len(failed_idx) > 0:
                mutated, _ = self._mp(
                    [
                        WorkerData(funcs, idx, args)
                        for funcs, idx, args in zip(
                            [(self.postprocess,) for idx in failed_idx],
                            failed_idx,
                            [mutated_pop[idx] for idx in failed_idx],
                        )
                    ],
                )

                succes_mutated_ids = where(mutated, lambda ind: ind != None)
                for idx in succes_mutated_ids:
                    mutated_pop[idx] = mutated[idx]

        failed_idx = where(mutated, lambda ind: ind == None)
        if len(failed_idx) > 0:
            generated = self.sampler(len(failed_idx))
            for enum_id, idx in enumerate(failed_idx):
                mutated_pop[idx] = generated[enum_id]

        return mutated_pop
