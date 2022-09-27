from gefest.tools.samplers.standard.standard import StandardSampler
from gefest.tools.samplers.sampler import Sampler


def configurate_sampler(domain):
    """
    ::TODO::
    Create abstract interface for configurations
    """
    # ------------
    # User-defined sampler
    # it should be created as object with .sample() method
    # ------------
    standard_sampler = StandardSampler()

    # ------------
    # GEFEST sampler,
    # it consist of user defined sampler and configurated domain
    # ------------
    sampler = Sampler(sampler=standard_sampler,
                      domain=domain)

    return sampler
