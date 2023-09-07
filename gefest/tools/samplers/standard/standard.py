from functools import partial
from typing import Callable

from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.geometry.utils import get_random_structure
from gefest.core.opt.strategies.strategy import Strategy
from gefest.core.utils.mp_manager import WorkerData


class StandardSampler(Strategy):
    def __init__(self, opt_params) -> None:
        super().__init__(opt_params.workers_manager)
        self.postprocessor: Callable = opt_params.postprocessor
        self.domain: Domain = opt_params.domain

    def __call__(self, n_samples: int) -> list[Structure]:
        return self.sample(n_samples=n_samples)

    def sample(self, n_samples: int) -> list[Structure]:
        
        parallel_funcs = [
            (
                (
                    partial(get_random_structure, domain=self.domain),
                    self.postprocessor,
                ),
                idx,
            )
            for _, idx in enumerate(range(n_samples))
        ]
        if self._mp:
            workers_data = [WorkerData(funcs, idx) for funcs, idx in parallel_funcs]
            random_pop, _ = self._mp.multiprocess(workers_data)
        else:
            random_pop = [func() for func in parallel_funcs]
        return random_pop
