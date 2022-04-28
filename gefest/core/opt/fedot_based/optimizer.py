import datetime
import os
import random
from functools import partial

import numpy as np
import pandas as pd

from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.log import default_log
from fedot.core.optimisers.adapters import DirectAdapter
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser, GPGraphOptimiserParameters, \
    GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from fedot.core.utils import fedot_project_root
from typing import Callable

from gefest.core.opt.fedot_based.adapter import StructureAdapter

from gefest.core.opt.constraints import check_constraints
from gefest.core.opt.operators.crossover import one_point_crossover
from gefest.core.opt.operators.initial import initial_pop_random
from gefest.core.opt.operators.mutation import mutation
from gefest.core.opt.operators.operators import default_operators
from gefest.core.opt.setup import Setup
from gefest.core.structure.point import Point
from gefest.core.structure.structure import Structure

random.seed(1)
np.random.seed(1)


def optimize(task_setup: Setup, objective_function: Callable, max_gens, pop_size):
    rules = [partial(check_constraints, domain=task_setup.domain)]

    initial = initial_pop_random(size=pop_size, domain=task_setup.domain)
    requirements = PipelineComposerRequirements(
        primary=['point'],
        secondary=['point'], max_arity=1,
        max_depth=100, pop_size=50, num_of_generations=50,
        crossover_prob=0.8, mutation_prob=0.9, timeout=None)

    optimiser_parameters = GPGraphOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        mutation_types=[mutation],
        crossover_types=[one_point_crossover],
        regularization_type=RegularizationTypesEnum.none)

    graph_generation_params = GraphGenerationParams(
        adapter=StructureAdapter(),
        rules_for_constraint=rules)
    graph_generation_params.custom['domain'] = task_setup.domain

    optimiser = EvoGraphOptimiser(
        graph_generation_params=graph_generation_params,
        metrics=[],
        parameters=optimiser_parameters,
        requirements=requirements, initial_graph=initial,
        log=default_log(logger_name='Structural', verbose_level=1))

    optimized_graph = optimiser.optimise(objective_function)
    optimized_structure = optimiser.graph_generation_params.adapter.restore(optimized_graph)

    return optimized_structure, optimiser.history
