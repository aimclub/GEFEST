from gefest.tools.optimizers.SPEA2.SPEA2 import SPEA2
from gefest.core.opt.operators.operators import default_operators
from gefest.tools.optimizers.optimizer import Optimizer


def configurate_optimizer(pop_size: int,
                          crossover_rate: int,
                          mutation_rate: int,
                          task_setup):
    # ------------
    # User-defined optimizer (SPEA2 in this case)
    # it should be created as object with .step() method
    # ------------
    params = SPEA2.Params(pop_size=pop_size,
                          crossover_rate=crossover_rate,
                          mutation_rate=mutation_rate,
                          mutation_value_rate=[])

    spea2_optimizer = SPEA2(params=params,
                            evolutionary_operators=default_operators(),
                            task_setup=task_setup)

    # ------------
    # GEFEST optimizer
    # ------------
    optimizer = Optimizer(optimizer=spea2_optimizer)

    return optimizer
