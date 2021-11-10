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
    s1, s2, domain = args[0], args[1], args[2]

    new_structure = copy.deepcopy(s1)

    crossover_point = random.randint(1, len(new_structure.polygons))

    part_1 = s1.polygons[0:crossover_point]
    if not isinstance(part_1, list):
        part_1 = [part_1]
    part_2 = s2.polygons[crossover_point:len(s1.polygons)]
    if not isinstance(part_2, list):
        part_2 = [part_2]

    result = copy.deepcopy(part_1)
    result.extend(copy.deepcopy(part_2))

    new_structure.polygons = result

    new_structure = postprocess(new_structure, domain)
    is_correct = check_constraints(new_structure, is_lightweight=True, domain=domain)
    if not is_correct:
        return None

    return new_structure


def crossover(s1: Structure, s2: Structure, domain: Domain, rate=0.4):
    random_val = random.random()
    if random_val >= rate or len(s1.polygons) == 1 or len(s2.polygons) == 1:
        if random.random() > 0.5:
            return s1
        else:
            return s2

    is_correct = False
    n_iter = 0

    new_structure = s1

    while not is_correct and n_iter < MAX_ITER:
        n_iter += 1
        print('cross', n_iter)
        if NUM_PROC > 1:
            with Pool(NUM_PROC) as p:
                new_items = p.map(crossover_worker,
                                  [[s1, s2, domain] for _ in range(NUM_PROC)])
        else:
            new_items = [crossover_worker([s1, s2, domain]) for _ in range(NUM_PROC)]

        for structure in new_items:
            if structure is not None:
                # is_correct = check_constraints(structure, domain=domain, is_lightweight=True)
                # if is_correct:
                new_structure = structure
                break

    return new_structure
