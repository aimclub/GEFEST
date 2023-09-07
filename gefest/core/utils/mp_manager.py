import multiprocessing
from typing import Any, Callable, Optional

from loguru import logger
from pydantic.dataclasses import dataclass

from gefest.core.utils import chain


@dataclass
class WorkerData:
    funcs: list[Callable]
    idx: int
    initial_args: Optional[Any] = None


class WorkersManager:
    def __init__(
        self,
        num_workers: Optional[int] = None,
        worker_func: Optional[Callable] = None,
    ) -> None:
        self.worker_func = worker_func if worker_func else WorkersManager.worker
        self.num_workers = num_workers if num_workers else multiprocessing.cpu_count()
        self.buffer_size = self.num_workers
        self.pool = multiprocessing.Pool(processes=self.num_workers)
        self.mpm = multiprocessing.Manager()
        self.processing_queue = self.mpm.Queue()
        self.result_queue = self.mpm.Queue()
        self.__out__queue_expected_size = 0
        self.workers = [
            self.pool.apply_async(
                self.worker_func,
                (self.processing_queue, self.result_queue),
            )
            for _ in range(self.num_workers)
        ]

    def __del__(self, *args):
        self.pool.close()
        self.pool.terminate()

    def get_queue_size(self) -> int:
        return self.__out__queue_expected_size

    def _queue_pop(self, args) -> None:
        self.__out__queue_expected_size += 1
        self.processing_queue.put(args)

    @logger.catch
    def _queue_get(self):
        self.__out__queue_expected_size -= 1
        return self.result_queue.get()

    @staticmethod
    @logger.catch
    def worker(
        processing_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
    ) -> None:
        """Executes given func with it params.
        Args:
            processing_queue: multiprocessing.Queue - input queue
            result_queue: multiprocessing.Queue - output queue
        Processing_queue provides data of the form:
            tuple(list[Callable], int, Optional[Any]) with
                  funcs,          idx, funcs arguments
        """
        while True:
            data: WorkerData = processing_queue.get()
            if data.initial_args is None:
                output = chain(*data.funcs)()
            else:
                output = chain(*data.funcs)(data.initial_args)

            result_queue.put((output, data.idx))

    def __call__(self, workers_data: list[WorkerData]):
        return self.multiprocess(workers_data)

    def multiprocess(self, workers_data: list[WorkerData]):
        """Executes provided functions with args in parallel.
        Args:
            args: tuple, contains some func, (e.g. mutation, crossover operators),
                 its arguments and index
        Returns:
            results: [list, list], contains provided functions outputs and indices they sorted by
        """
        res_ids_pairs = []

        for idx, operand in enumerate(workers_data):

            self._queue_pop(operand)
            if idx < self.buffer_size:
                continue

            else:
                res_ids_pairs.append(self._queue_get())

        for _ in range(self.get_queue_size()):
            res_ids_pairs.append(self._queue_get())

        if len(res_ids_pairs) == 0:
            print('err')
        res_ids_pairs = sorted(res_ids_pairs, key=lambda x: x[1])
        res, ids = list(zip(*res_ids_pairs))

        return list(res), list(ids)
