import copy
import random
from multiprocessing import Pool

from gefest.core.algs.postproc.resolve_errors import postprocess
from gefest.core.opt.constraints import check_constraints
from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Structure

MAX_ITER = 50000
NUM_PROC = 1


def crossover_worker(args):
    """
    One point crossover between two selected structures
    Polygons are exchanged between structures
    """

    s1, s2, domain = args[0], args[1], args[2]

    new_structure = copy.deepcopy(s1)

    crossover_point = random.randint(1, len(new_structure.polygons) + 1)  # Choosing crossover point randomly

    # Crossover conversion
    part_1 = s1.polygons[0:crossover_point]
    if not isinstance(part_1, list):
        part_1 = [part_1]
    part_2 = s2.polygons[crossover_point:len(s1.polygons)]
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


def crossover(s1: Structure, s2: Structure, domain: Domain, rate=0.4):
    random_val = random.random()
    if random_val >= rate or len(s1.polygons) == 1 or len(s2.polygons) == 1:
        # In the case when the structures consist of only one polygon,
        # the transformation is not performed
        if random.random() > 0.5:
            return s1
        else:
            return s2
    elif len(s1.polygons) == 0:
        return s2
    elif len(s2.polygons) == 0:
        return s1

    new_structure = s2

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
