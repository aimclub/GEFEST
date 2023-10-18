from functools import partial
from typing import Callable

from loguru import logger

from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain
from gefest.core.geometry.utils import get_random_structure
from gefest.core.utils.parallel_manager import BaseParallelDispatcher
from gefest.tools.samplers.sampler import Sampler


class StandardSampler(Sampler):
    def __init__(self, opt_params) -> None:
        super().__init__(
            samples_generator=get_random_structure,
            domain=opt_params.domain,
        )
        self.domain: Domain = opt_params.domain
        self.postprocessor: Callable = opt_params.postprocessor
        self.postprocess_attempts: int = opt_params.postprocess_attempts
        self._pm = BaseParallelDispatcher(opt_params.n_jobs)

    def __call__(self, n_samples: int) -> list[Structure]:
        return self.sample(n_samples=n_samples)

    def sample(self, n_samples: int) -> list[Structure]:
        random_pop = self._pm.exec_parallel(
            partial(get_random_structure, domain=self.domain),
            tuple(range(n_samples + 1)),
            False,
            False,
        )
        corrected = self._pm.exec_parallel(
            self.postprocessor,
            [(ind,) for ind in random_pop],
        )

        random_pop = [ind for ind in corrected if ind is not None]

        logger.info(f'{n_samples}, {len(random_pop)}')
        pop = random_pop[:n_samples]
        return pop
