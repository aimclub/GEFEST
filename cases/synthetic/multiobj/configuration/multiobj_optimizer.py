from functools import partial

from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.regularization import RegularizationTypesEnum
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.objective import Objective

from gefest.core.opt.adapter import StructureAdapter
from gefest.core.opt.constraints import check_constraints
from gefest.tools.optimizers.optimizer import Optimizer

from gefest.tools.optimizers.golem_optimizer.nsga2 import NSGA2
from gefest.tools.optimizers.golem_optimizer.moead import MOEAD
from gefest.tools.optimizers.golem_optimizer.age import AGE


def configurate_optimizer(pop_size: int,
                          crossover_rate: float,
                          mutation_rate: float,
                          task_setup):
    # ------------
    # User-defined optimizer
    # it should be created as object with .step() method
    # ------------
    rules = [partial(check_constraints, domain=task_setup.domain)]

    requirements = GraphRequirements()

    optimiser_parameters = GPAlgorithmParameters(
        crossover_prob=crossover_rate,
        mutation_prob=mutation_rate,
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        regularization_type=RegularizationTypesEnum.none)

    graph_generation_params = GraphGenerationParams(
        adapter=StructureAdapter(),
        rules_for_constraint=rules)

    adapter = StructureAdapter()
    params = {'objective': Objective(metrics=2),
              'graph_generation_params': graph_generation_params,
              'graph_optimizer_params': optimiser_parameters,
              'requirements': requirements,
              'initial_graphs': [1, 2, 3]}

    age_optimizer = AGE(adapter=adapter,
                        crossover_rate=crossover_rate,
                        mutation_rate=mutation_rate,
                        pop_size=pop_size,
                        task_setup=task_setup,
                        params=params)

    # ------------
    # GEFEST optimizer
    # ------------
    optimizer = Optimizer(optimizer=age_optimizer)

    return optimizer
