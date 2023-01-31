from functools import partial

from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.optimisers.gp_comp.gp_optimizer import GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.optimisers.objective import Objective

from gefest.core.opt.adapter import StructureAdapter
from gefest.core.opt.constraints import check_constraints
from gefest.tools.optimizers.optimizer import Optimizer

from gefest.tools.optimizers.fedot_optimizer.nsga2 import NSGA2
from gefest.tools.optimizers.fedot_optimizer.moead import MOEAD
from gefest.tools.optimizers.fedot_optimizer.age import AGE


def configurate_optimizer(pop_size: int,
                          crossover_rate: float,
                          mutation_rate: float,
                          task_setup):
    # ------------
    # User-defined optimizer
    # it should be created as object with .step() method
    # ------------
    rules = [partial(check_constraints, domain=task_setup.domain)]

    requirements = PipelineComposerRequirements(
        primary=['point'],
        secondary=['point'],
        max_arity=1,
        max_depth=100,
        num_of_generations=1,
        timeout=None)

    optimiser_parameters = GPGraphOptimizerParameters(
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
