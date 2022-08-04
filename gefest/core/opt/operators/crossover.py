import copy
import random
from multiprocessing import Pool

from gefest.core.algs.postproc.resolve_errors import postprocess
from gefest.core.opt.constraints import check_constraints
from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Structure, shuffle_structures

MAX_ITER = 50000
NUM_PROC = 1


def crossover_worker(args):
    """
    One point crossover between two selected structures
    Polygons are exchanged between structures
    """

    s1, s2, domain = args[0], args[1], args[2]

    new_structure = copy.deepcopy(s1)
    s1 = copy.deepcopy(s1)
    s2 = copy.deepcopy(s2)

    # Checking if at least one Structure does not have any polygons
    if not all([len(s1.polygons), len(s2.polygons)]):
        # All polygons are shuffling between Structures in random way
        s1, s2 = shuffle_structures(s1, s2)

    crossover_point = random.randint(1, min(len(s1.polygons), len(s2.polygons)))  # Choosing crossover point randomly

    # Crossover conversion
    part_1 = s1.polygons[:crossover_point]
    if not isinstance(part_1, list):
        part_1 = [part_1]
    part_2 = s2.polygons[crossover_point:]
    if not isinstance(part_2, list):
        part_2 = [part_2]

    result = copy.deepcopy(part_1)
    result.extend(copy.deepcopy(part_2))

    new_structure.polygons = result

    # Postprocessing for new structure
    new_structure = postprocess(new_structure, domain)
    constraints = check_constraints(structure=new_structure, domain=domain)
    max_attempts = 3  # Number of postprocessing attempts
    while not constraints:
        new_structure = postprocess(new_structure, domain)
        constraints = check_constraints(structure=new_structure, domain=domain)
        max_attempts -= 1
        if max_attempts == 0:
            # If the number of attempts is over,
            # the transformation is considered unsuccessful
            # and one of the structures is returned
            return s1
    return new_structure


def crossover(s1: Structure, s2: Structure, domain: Domain, rate: float = 0.4) -> Structure:
    """the crossover proccess method

    Args:
        s1: the firts generative :obj:`Structure`
        s2: the second generative :obj:`Structure`
        domain: the :obj:`Domain` that use for optimization process
        rate: likelihood for success crossover. Defaults to 0.4.

    Returns:
        if crossover was finished succsess - new structure, born from parents
        :obj:`s1` and :obj:`s2`; otherwise will randomly return one of the given :obj:`Structure`
    """

    random_val = random.random()
    if random_val >= rate or all([len(s1.polygons) <= 1, len(s2.polygons) <= 1]):
        # In the case when all of structures consist one polygon or less,
        # the transformation is not performed
        return random.choice([s1, s2])

    new_structure = copy.deepcopy(s1)

    if NUM_PROC > 1:
        # Calculations on different processor cores
        with Pool(NUM_PROC) as p:
            new_items = p.map(crossover_worker,
                              [[s1, s2, domain] for _ in range(NUM_PROC)])
    else:
        new_items = [crossover_worker([s1, s2, domain]) for _ in range(NUM_PROC)]

    for structure in new_items:
        if structure is not None:
            new_structure = structure
            break

    return new_structure
