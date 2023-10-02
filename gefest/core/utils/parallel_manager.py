from typing import Any, Callable

from joblib import Parallel, cpu_count, delayed
from loguru import logger


class BaseParallelDispatcher:
    """Provides interface for parallel execution."""

    def __init__(self, n_jobs: int = -1):
        self.n_jobs = self._determine_n_jobs(n_jobs)

    # def __call__(self, *args, **kwargs) -> list[Any]:
    #     return self.exec_parallel(*args, **kwargs)

    def _determine_n_jobs(self, n_jobs: int = -1):
        if n_jobs > cpu_count() or n_jobs == -1:
            n_jobs = cpu_count()
        return n_jobs

    @logger.catch
    def exec_parallel(
        self,
        func: Callable,
        arguments: list[Any],
        use: bool = True,
    ) -> list[Any]:
        """Executes provided function in parallel.

        Args:
            func (Callable): Function to execute.
            arguments (list[Any]): Each element in list is arguments for separate func call.
            use (bool, optional): If True, each arg in args will be used as func argument,
                otherwise func will be called without arguments len(args) times. Defaults to True.

        Returns:
            list[Any]
        """
        if self.n_jobs == 0:
            if use:
                result = [func(arg) for arg in arguments]
            else:
                result = [func() for _ in arguments]
        else:
            parallel = Parallel(n_jobs=self.n_jobs, verbose=0, pre_dispatch='2*n_jobs')
            if use:
                result = parallel(delayed(func)(arg) for arg in arguments)
            else:
                result = parallel(delayed(func)() for _ in arguments)
        return result
