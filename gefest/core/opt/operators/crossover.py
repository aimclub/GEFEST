import copy
import random
from multiprocessing import Pool

from fedot.core.optimisers.optimizer import GraphGenerationParams

from gefest.core.algs.postproc.resolve_errors import postprocess, iterative_postprocess
from gefest.core.opt.constraints import check_constraints
from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Structure

MAX_ITER = 50000
NUM_PROC = 1


def one_point_crossover(**kwargs):
    """
    One point crossover between two selected structures
    Polygons are exchanged between structures
    """

    s1 = kwargs['graph_first']
    s2 = kwargs['graph_second']
    domain = kwargs['params'].custom['domain']

    new_structure = copy.deepcopy(s1)

    crossover_point = random.randint(1, len(new_structure.polygons))  # Choosing crossover point randomly

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
    new_structure = iterative_postprocess(new_structure=new_structure, default_structure=s1,
                                          domain=domain,
                                          max_attempts=4)
    return [new_structure]
