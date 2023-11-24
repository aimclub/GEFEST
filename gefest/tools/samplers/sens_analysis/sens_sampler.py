from copy import deepcopy
from multiprocessing import Pool

from gefest.core.opt.constraints import check_constraints
from gefest.core.opt.operators.sensitivity_methods import get_structure_for_analysis
from gefest.core.opt.postproc.resolve_errors import postprocess
from gefest.core.structure.domain import Domain
from gefest.core.utils import project_root

MAX_ITER = 50000
NUM_PROC = 1


class SensitivitySampler:
    def __init__(self, path: str):
        self.path = path
        pass

    def sample(self, n_samples: int, domain: Domain, initial_state=None):
        # Method for initialization of population
        population_new = []

        if initial_state is None:
            while len(population_new) < n_samples:
                if NUM_PROC > 1:
                    with Pool(NUM_PROC) as p:
                        new_items = p.map(self.get_pop_worker, [domain] * n_samples)
                else:
                    new_items = []
                    for _ in range(n_samples):
                        new_items.append(self.get_pop_worker(domain))

                for structure in new_items:
                    population_new.append(structure)
                    if len(population_new) == n_samples:
                        return population_new
        else:
            for _ in range(n_samples):
                population_new.append(deepcopy(initial_state))
        return population_new

    def get_pop_worker(self, domain: Domain):
        # Create a random structure and postprocess it
        structure = get_structure_for_analysis(self.path)
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
                structure = get_structure_for_analysis(self.path)
                structure = postprocess(structure, domain)
                constraints = check_constraints(structure=structure, domain=domain)

        return structure
