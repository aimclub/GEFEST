from copy import deepcopy
from multiprocessing import Pool

from gefest.core.algs.postproc.resolve_errors import postprocess
from gefest.core.opt.constraints import check_constraints
from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import get_random_structure

MAX_ITER = 50000
NUM_PROC = 1


def initial_pop_random(size: int, domain: Domain, initial_state=None):
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
    return population_new


def get_pop_worker(domain):
    # print(f'Try to create size {structure_size}')

    is_correct = False
    while not is_correct:
        structure = get_random_structure(domain=domain)
        # structure.plot(structure, title='Initial')
        structure = postprocess(structure, domain)
        # structure.plot(structure, title='Initial post')
        is_correct = check_constraints(structure, is_lightweight=True, domain=domain)

        if is_correct:
            # structure.plot(title='Initial correct')
            print(f'Created, domain {domain.name}')
            return structure
