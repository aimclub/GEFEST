from abc import ABCMeta, abstractmethod
from typing import Any, Callable

from gefest.core.geometry import Structure
from gefest.core.geometry.domain import Domain


class Sampler(metaclass=ABCMeta):
    """Interface for samplers."""

    def __init__(
        self,
        samples_generator: Callable[[Any], Structure],
        domain: Domain,
    ) -> None:
        self.samples_generator = samples_generator
        self.domain = domain

    def __call__(
        self,
        n_samples: int,
        **kwargs,
    ) -> list[Structure]:
        """Simplifies usage of samplers.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            list[Structure]: Generated samples.
        """
        return self.sample(n_samples)

    @abstractmethod
    def sample(self, n_samples: int) -> list[Structure]:
        """Must implement sampling logic.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            list[Structure]: Generated samples.
        """
        ...
