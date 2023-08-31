import copy
from multiprocessing import Pool

import numpy as np

from gefest.core.geometry import Structure
from gefest.core.opt.domain import Domain


# pairs for crossover selection
def panmixis(pop: list[Structure]) -> list[tuple[Structure, Structure]]:
    np.random.shuffle(list(pop))
    return [(pop[idx], pop[idx + 1]) for idx in range(len(pop) - 1)]


# best indivisual selection
def structure_level_crossover(
    operands: tuple[Structure, Structure],
    domain: Domain,
    **kwargs,
):
    s1, s2 = operands
    new_structure = copy.deepcopy(s1)

    crossover_point = np.random.randint(
        1,
        len(new_structure.polygons) + 1,
    )

    # Crossover conversion
    part_1 = s1.polygons[0:crossover_point]
    if not isinstance(part_1, list):
        part_1 = [part_1]
    part_2 = s2.polygons[crossover_point : len(s1.polygons)]
    if not isinstance(part_2, list):
        part_2 = [part_2]

    result = copy.deepcopy(part_1)
    result.extend(copy.deepcopy(part_2))

    new_structure.polygons = result

    return new_structure
