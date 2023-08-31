from functools import partial
from typing import Callable

from gefest.core.geometry import Structure
from gefest.core.geometry.utils import get_random_structure
from gefest.core.opt.strategies.strategy import Strategy
from gefest.core.opt.domain import Domain
from gefest.core.utils.mp_manager import WorkerData


class StandardSampler(Strategy):
    def __init__(self, opt_params) -> None:
        super().__init__()
        self.postprocessor: Callable = opt_params.postprocessor
        self.domain: Domain = opt_params.domain

    def __call__(self, n_samples: int) -> list[Structure]:
        return self.sample(n_samples=n_samples)

    def sample(self, n_samples: int) -> list[Structure]:
        data = [
            (
                (
                    partial(get_random_structure, domain=self.domain),
                    self.postprocessor,
                ),
                idx,
            )
            for _, idx in enumerate(range(n_samples))
        ]

        parallel_generators = [WorkerData(funcs, idx) for funcs, idx in data]

        random_pop, _ = self._mp.multiprocess(parallel_generators)
        return random_pop
