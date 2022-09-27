from pathlib import Path

from gefest.tools.samplers.DL.heat.heat_sampler import DeepSampler
from gefest.tools.samplers.sampler import Sampler


def configurate_sampler(domain, path_to_sampler=False):
    # ------------
    # User-defined sampler
    # it should be created as object with .sample() method
    # ------------
    if not path_to_sampler:
        root_path = Path(__file__).parent.parent.parent.parent
        path_to_sampler = f'{root_path}/gefest/tools/samplers/DL/heat/DL_sampler'

    sampler = DeepSampler(path_to_sampler)

    # ------------
    # GEFEST sampler,
    # it consist of user defined sampler and configurated domain
    # ------------
    sampler = Sampler(sampler=sampler,
                      domain=domain)

    return sampler
