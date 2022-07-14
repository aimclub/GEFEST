from gefest.tools.samplers.DL.microfluid.microfluid_sampler import DeepSampler
from gefest.tools.samplers.sampler import Sampler


def configurate_sampler(domain):
    # ------------
    # User-defined sampler
    # it should be created as object with .sample() method
    # ------------
    sampler = DeepSampler()

    # ------------
    # GEFEST sampler,
    # it consist of user defined sampler and configurated domain
    # ------------
    sampler = Sampler(sampler=sampler,
                      domain=domain)

    return sampler
