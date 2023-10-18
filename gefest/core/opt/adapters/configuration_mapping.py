from functools import partial

from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams

from gefest.core.algs.postproc.resolve_errors import validate
from gefest.core.configs.optimization_params import OptimizationParams
from gefest.core.opt.adapters.factories import StructureFactory
from gefest.core.opt.adapters.operator import OperationWrap
from gefest.core.opt.operators.crossovers import crossover_structures
from gefest.core.opt.operators.mutations import mutate_structure


def map_into_graph_requirements(
    opt_params: OptimizationParams,
) -> GraphRequirements:
    return GraphRequirements(
        early_stopping_timeout=opt_params.early_stopping_timeout,
        early_stopping_iterations=opt_params.early_stopping_iterations,
        keep_n_best=opt_params.pop_size,
        keep_history=False,
        num_of_generations=opt_params.n_steps,
        n_jobs=opt_params.n_jobs,
        history_dir=opt_params.log_dir,
    )


def map_into_graph_generation_params(
    opt_params: OptimizationParams,
) -> GraphGenerationParams:
    return GraphGenerationParams(
        adapter=opt_params.golem_adapter,
        rules_for_constraint=[
            partial(validate, rules=opt_params.postprocess_rules, domain=opt_params.domain),
        ],
        random_graph_factory=StructureFactory(opt_params.sampler, opt_params.golem_adapter),
    )


def map_into_gpa(
    opt_params: OptimizationParams,
) -> GPAlgorithmParameters:
    return GPAlgorithmParameters(
        multi_objective=False,
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        mutation_types=[
            OperationWrap(
                executor=mutate_structure,
                operations=[mut],
                operation_chance=opt_params.mutation_prob,
                operations_probs=[1],
                domain=opt_params.domain,
                postproc_func=opt_params.postprocessor,
                postprocess_rules=opt_params.postprocess_rules,
                attempts=opt_params.postprocess_attempts,
            )
            for mut in opt_params.mutations
        ],
        crossover_types=[
            OperationWrap(
                executor=crossover_structures,
                operations=[opt_params.crossovers[1]],
                operation_chance=opt_params.crossover_prob,
                operations_probs=[1],
                domain=opt_params.domain,
                postproc_func=opt_params.postprocessor,
                postprocess_rules=opt_params.postprocess_rules,
                attempts=opt_params.postprocess_attempts,
            ),
        ],
        pop_size=opt_params.pop_size,
        max_pop_size=int(opt_params.pop_size * 1.5),
        crossover_prob=opt_params.crossover_prob,
        mutation_prob=1,
        adaptive_mutation_type=MutationAgentTypeEnum.default,
    )
