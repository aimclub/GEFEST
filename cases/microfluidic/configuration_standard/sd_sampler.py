from gefest.tools.samplers.standard.standard import StandardSampler
from gefest.tools.samplers.sampler import Sampler


def configurate_sampler(domain):
    # ------------
    # User-defined sampler
    # it should be created as object with .sample() method
    # ------------
    sampler = StandardSampler()

    # ------------
    # GEFEST sampler,
    # it consist of user defined sampler and configurated domain
    # ------------
    sampler = Sampler(sampler=sampler,
                      domain=domain)

    return sampler
