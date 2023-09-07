from typing import Any

from gefest.core.geometry.domain import Domain


class Sampler:
    """
    ::TODO::
    make abstract class for all samplers
    """

    def __init__(self, sampler, domain: "Domain"):
        """
        Base sampler class
        :param sampler: (Object) object with method sample
        :param domain: (Domain) design domain
        """
        self.sampler = sampler
        self.domain = domain

    def sample(self, n_samples: int):
        """
        Sampling from certain sampler
        :param n_samples: (Int) number of samples
        :return: (List(Structure)) sample n_samples structures
        """
        samples = self.sampler.sample(n_samples=n_samples, domain=self.domain)

        return samples

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.sample(args)
