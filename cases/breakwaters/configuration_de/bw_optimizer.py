from gefest.tools.optimizers.GA.GA import GA
from gefest.core.opt.operators.operators import default_operators
from gefest.tools.optimizers.optimizer import Optimizer


def configurate_optimizer(pop_size: int,
                          crossover_rate: int,
                          mutation_rate: int,
                          task_setup):
    """
    ::TODO::
    Create abstract interface for configurations
    """
    # ------------
    # User-defined optimizer
    # it should be created as object with .step() method
    # ------------
    params = GA.Params(pop_size=pop_size,
                       crossover_rate=crossover_rate,
                       mutation_rate=mutation_rate,
                       mutation_value_rate=[])

    ga_optimizer = GA(params=params,
                      evolutionary_operators=default_operators(),
                      task_setup=task_setup)

    # ------------
    # GEFEST optimizer
    # ------------
    optimizer = Optimizer(optimizer=ga_optimizer)

    return optimizer
