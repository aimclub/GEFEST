import copy
from typing import Callable

import numpy as np

from gefest.core.geometry import Structure
from gefest.core.opt.abstract.strategy import Strategy
from gefest.core.utils import WorkerData, where


class MutationStrategy(Strategy):
    def __init__(self, opt_params):
        super().__init__()
        self.mutation_prob = opt_params.mutation_prob
        self.mutations = opt_params.mutations
        self.each_prob = opt_params.mutation_each_prob
        self.postprocess: Callable = opt_params.postprocessor
        self.sampler = opt_params.sampler
        self.postprocess_attempts = 3

    def __call__(self, pop: list[Structure]) -> list[Structure]:
        return self.mutate(pop=pop)

    def mutate(self, pop: list[Structure]):

        individuals_to_mutate_ids = np.random.choice(
            a=[0, 1], size=len(pop), p=[1 - self.mutation_prob, self.mutation_prob]
        )
        chosen_mutations = np.random.choice(
            a=self.mutations,
            size=len(individuals_to_mutate_ids),
            p=self.each_prob,
        )

        chosen_mutations = [(cm, self.postprocess) for cm in chosen_mutations]
        individuals_to_mutate = copy.deepcopy([pop[idx] for idx in individuals_to_mutate_ids])

        mutated, _ = self._mp(
            [
                WorkerData(funcs, idx, args)
                for funcs, idx, args in zip(
                    chosen_mutations,
                    range(len(individuals_to_mutate_ids)),
                    individuals_to_mutate,
                )
            ]
        )

        succes_mutated_ids = where(mutated, lambda ind: ind != None)
        for idx in succes_mutated_ids:
            individuals_to_mutate[idx] = mutated[idx]
        # individuals_to_mutate[succes_mutated_ids] = mutated[succes_mutated_ids]

        for _ in range(self.postprocess_attempts):
            failed_idx = where(mutated, lambda ind: ind == None)
            if len(failed_idx) > 0:
                mutated, _ = self._mp(
                    [
                        WorkerData(funcs, idx, args)
                        for funcs, idx, args in zip(
                            [chosen_mutations[idx] for idx in failed_idx],
                            failed_idx,
                            [individuals_to_mutate[idx] for idx in failed_idx],
                        )
                    ]
                )

                succes_mutated_ids = where(mutated, lambda ind: ind != None)
                for idx in succes_mutated_ids:
                    individuals_to_mutate[idx] = mutated[idx]

        failed_idx = where(mutated, lambda ind: ind == None)
        if len(failed_idx) > 0:
            generated = self.sampler(len(failed_idx))
            for enum_id, idx in enumerate(failed_idx):
                individuals_to_mutate[idx] = generated[enum_id]
        for mutated_id, idx in enumerate(individuals_to_mutate_ids):
            pop[idx] = individuals_to_mutate[mutated_id]

        return pop
