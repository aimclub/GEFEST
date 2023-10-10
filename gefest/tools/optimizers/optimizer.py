from abc import ABCMeta, abstractmethod

from gefest.core.geometry import Structure


class Optimizer(metaclass=ABCMeta):
    def __init__(
        self,
        log_dispatcher=None,
        **kwargs,
    ) -> None:
        self.log_dispatcher = log_dispatcher

    @abstractmethod
    def optimize(
        self,
        n_steps: int,
        **kwargs,
    ) -> list[Structure]:
        ...
