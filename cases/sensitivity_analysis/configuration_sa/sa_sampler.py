from gefest.tools.samplers.sens_analysis.sens_sampler import SensitivitySampler
from gefest.tools.samplers.sampler import Sampler


def configurate_sampler(domain, path):
    # ------------
    # User-defined sampler
    # it should be created as object with .sample() method
    # ------------
    sensitivity_sampler = SensitivitySampler(path=path)

    # ------------
    # GEFEST sampler,
    # it consist of user defined sampler and configurated domain
    # ------------
    sampler = Sampler(sampler=sensitivity_sampler,
                      domain=domain)

    return sampler
