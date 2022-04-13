from copy import deepcopy
from multiprocessing import Pool

from typing import List

from gefest.core.algs.postproc.resolve_errors import postprocess
from gefest.core.opt.constraints import check_constraints
from gefest.core.opt.individual import Individual
from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import get_random_structure

MAX_ITER = 50000
NUM_PROC = 1


def initial_pop_random(size: int, domain: Domain, initial_state=None) -> List[Individual]:
    """
    Initialises the first population
    :param size: population dise
    :param domain: description of domain
    :param initial_state: pre-defined initial assumption for population
    :return: population
    """

    print('Start init')
    population_new = []

    if initial_state is None:
        while len(population_new) < size:
            if NUM_PROC > 1:
                with Pool(NUM_PROC) as p:
                    new_items = p.map(get_pop_worker, [domain] * size)
            else:
                new_items = []
                for i in range(size):
                    new_items.append(get_pop_worker(domain))
                    print(f'Initial created: {i} from {size}')

            for structure in new_items:
                population_new.append(structure)
                if len(population_new) == size:
                    return population_new
        print('End init')
    else:
        for _ in range(size):
            population_new.append(deepcopy(initial_state))
    population_new = [Individual(genotype=gen) for gen in population_new]
    return population_new


def get_pop_worker(domain):
    # Create a random structure and postprocess it
    structure = get_random_structure(domain=domain)
    structure = postprocess(structure, domain)
    constraints = check_constraints(structure=structure, domain=domain)
    max_attempts = 3  # Number of postprocessing attempts
    while not constraints:
        structure = postprocess(structure, domain)
        constraints = check_constraints(structure=structure, domain=domain)
        max_attempts -= 1
        if max_attempts < 0:
            # If the number of attempts is over,
            # a new structure is created on which postprocessing is performed
            structure = get_random_structure(domain=domain)
            structure = postprocess(structure, domain)
            constraints = check_constraints(structure=structure, domain=domain)
    return structure
