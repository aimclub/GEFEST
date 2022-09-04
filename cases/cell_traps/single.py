import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import mph

from cases.cell_traps.comsol import simulate_hydrodynamics
from gefest.core.opt.analytics import EvoAnalytics
from gefest.core.opt.optimize import optimize
from gefest.core.opt.setup import Setup
from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Structure
from gefest.core.viz.struct_vizualizer import StructVizualizer

random.seed(42)
np.random.seed(42)


if __name__ == '__main__':
    EvoAnalytics.clear()

    comsol_objective = partial(simulate_hydrodynamics, client=mph.Client(cores=12))

    domain = Domain(allowed_area=[(-125, 100),
                                  (-75, 155),
                                  (15, 155),
                                  (40, 90),
                                  (-10, -130),
                                  (-10, -155),
                                  (-125, -155)],
                    max_poly_num=3,
                    min_poly_num=1,
                    )

    task_setup = Setup(domain=domain)

    optimized_structure = optimize(task_setup=task_setup,
                                   objective_function=comsol_objective,
                                   pop_size=100,
                                   max_gens=200)

    visualiser = StructVizualizer(task_setup.domain)
    plt.figure(figsize=(7, 7))

    info = {'fitness': comsol_objective(optimized_structure),
            'type': 'prediction'}
    visualiser.plot_structure(optimized_structure, info)

    EvoAnalytics.create_boxplot()
