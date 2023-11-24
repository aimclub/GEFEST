from abc import ABCMeta, abstractmethod

from gefest.core.geometry import Structure


class Optimizer(metaclass=ABCMeta):
    """Interface for optimizers."""

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
        """Must implement optimization logic.

        Args:
            n_steps (int): Number of optimization steps.

        Returns:
            list[Structure]: Optimized population.
        """
        ...
