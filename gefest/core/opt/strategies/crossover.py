import copy
from typing import Callable

import numpy as np

from gefest.core.geometry import Structure
from gefest.core.opt.abstract.strategy import Strategy
from gefest.core.utils import WorkerData, where


class CrossoverStrategy(Strategy):
    def __init__(self, opt_params):
        super().__init__()

        self.prob = opt_params.crossover_prob
        self.crossovers = opt_params.crossovers
        self.each_prob = opt_params.crossover_each_prob
        self.postprocess: Callable = opt_params.postprocessor
        self.parent_pairs_selector: Callable = opt_params.pair_selector
        self.sampler: Callable = opt_params.sampler
        self.attempts = 3

    def __call__(self, pop: list[Structure]) -> list[Structure]:
        return self.crossover(pop=pop)

    def crossover(self, pop: list[Structure]):
        pairs_to_crossover = copy.deepcopy(self.parent_pairs_selector(pop))
        new_generation = np.full(len(pairs_to_crossover), None)
        chosen_crossovers = np.random.choice(
            a=self.crossovers,
            size=len(pairs_to_crossover),
            p=self.each_prob,
        )

        chosen_crossovers = [(cm, self.postprocess) for cm in chosen_crossovers]
        children, _ = self._mp(
            [
                WorkerData(funcs, idx, args)
                for funcs, idx, args in zip(
                    chosen_crossovers,
                    range(len(pairs_to_crossover)),
                    pairs_to_crossover,
                )
            ]
        )

        succes_crossover_ids = where(children, lambda ind: ind != None)
        for idx in succes_crossover_ids:
            new_generation[idx] = children[idx]

        for _ in range(self.attempts):
            failed_idx = where(new_generation, lambda ind: ind == None)
            if len(failed_idx) > 0:
                children, _ = self._mp(
                    [
                        WorkerData(funcs, idx, args)
                        for funcs, idx, args in zip(
                            [chosen_crossovers[idx] for idx in failed_idx],
                            failed_idx,
                            [pairs_to_crossover[idx] for idx in failed_idx],
                        )
                    ]
                )
                succes_crossover_ids = where(children, lambda ind: ind != None)
                for idx in succes_crossover_ids:
                    new_generation[idx] = children[idx]
            else:
                break
        failed_idx = where(new_generation, lambda ind: ind == None)
        if len(failed_idx) > 0:
            new_generation[failed_idx] = self.sampler(len(failed_idx))
        pop.extend(new_generation)

        return pop
