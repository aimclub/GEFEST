from typing import Any, Callable

from joblib import Parallel, cpu_count, delayed
from loguru import logger


class BaseParallelDispatcher:
    """Provides api for easy to use parallel execution with joblib backernd."""

    def __init__(self, n_jobs: int = -1):
        """Inits parallel dispatcher.

        Args:
            n_jobs (int, optional): Set 0 to debug withou joblib.
                For other values see joblib docs. Defaults to -1.

        """
        self.n_jobs = self._determine_n_jobs(n_jobs)

    def _determine_n_jobs(self, n_jobs: int = -1):
        if n_jobs > cpu_count() or n_jobs == -1:
            n_jobs = cpu_count()

        return n_jobs

    @logger.catch
    def exec_parallel(
        self,
        func: Callable,
        arguments: list[tuple[Any]],
        use: bool = True,
        flatten: bool = True,
    ) -> list[Any]:
        """Executes provided function in parallel.

        Args:
            func (Callable): Function to execute.
            arguments (list[tuple[Any]]): Each tuple in list is *args for separate func call.
            use (bool, optional): If True, each arg in args will be used as func argument,
                otherwise func will be called without arguments len(args) times. Defaults to True.
            flatten (bool): If true, makes flatten list from function ouput lists.

        Returns:
            list[Any]
        """
        if self.n_jobs == 0:
            if use:
                result = [func(*args) for args in arguments]
            else:
                result = [func() for _ in arguments]
        else:
            parallel = Parallel(n_jobs=self.n_jobs, verbose=0, pre_dispatch='2*n_jobs')
            if use:
                result = parallel(delayed(func)(*args) for args in arguments)
            else:
                result = parallel(delayed(func)() for _ in arguments)

        if flatten:
            if not all(isinstance(res, (tuple, list)) for res in result):
                raise ValueError(
                    f"""Got an unexpected outputs from {func}.
                    Got: {[type(res) for res in result]}. Expected tuples or lists.""",
                )

            result = [item for separate_output in result for item in separate_output]

        return result
