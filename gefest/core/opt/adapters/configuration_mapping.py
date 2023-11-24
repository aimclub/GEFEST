from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gefest.core.configs.optimization_params import OptimizationParams

from functools import partial

from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams

from gefest.core.opt.adapters.factories import StructureFactory
from gefest.core.opt.adapters.operator import OperationWrap
from gefest.core.opt.operators.crossovers import crossover_structures
from gefest.core.opt.operators.mutations import mutate_structure
from gefest.core.opt.postproc.resolve_errors import validate


def map_into_graph_requirements(
    opt_params: OptimizationParams,
) -> GraphRequirements:
    """Translates OptimizationParams into GraphRequirements."""
    return GraphRequirements(
        early_stopping_timeout=opt_params.early_stopping_timeout,
        early_stopping_iterations=opt_params.early_stopping_iterations,
        keep_n_best=opt_params.pop_size,
        keep_history=opt_params.golem_keep_histoy,
        num_of_generations=opt_params.n_steps,
        n_jobs=opt_params.estimation_n_jobs,
        history_dir=opt_params.log_dir,
    )


def map_into_graph_generation_params(
    opt_params: OptimizationParams,
) -> GraphGenerationParams:
    """Translates OptimizationParams into GraphGenerationParams."""
    return GraphGenerationParams(
        adapter=opt_params.golem_adapter,
        rules_for_constraint=[
            partial(
                validate,
                rules=opt_params.postprocess_rules,
                domain=opt_params.domain,
            ),
        ],
        random_graph_factory=StructureFactory(opt_params.sampler, opt_params.golem_adapter),
    )


def map_into_gpa(
    opt_params: OptimizationParams,
) -> GPAlgorithmParameters:
    """Translates OptimizationParams into GPAlgorithmParameters."""
    return GPAlgorithmParameters(
        multi_objective=False,
        genetic_scheme_type=getattr(
            GeneticSchemeTypesEnum,
            opt_params.golem_genetic_scheme_type.name,
        ),
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
        selection_types=[getattr(SelectionTypesEnum, opt_params.golem_selection_type)],
        pop_size=opt_params.pop_size,
        max_pop_size=opt_params.pop_size,
        crossover_prob=opt_params.crossover_prob,
        mutation_prob=1,
        adaptive_mutation_type=getattr(
            MutationAgentTypeEnum,
            opt_params.golem_adaptive_mutation_type,
        ),
    )
