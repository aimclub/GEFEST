from functools import partial

import pandas as pd
from examples.synthetic_graph_evolution.experiment_setup import run_experiments
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from golem.core.adapter.adapter import BaseOptimizationAdapter
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.graph import OptGraph, OptNode

from golem.core.optimisers.meta.surrogate_optimizer import SurrogateEachNgenOptimizer
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.random_graph_factory import RandomGraphFactory

from gefest.core.algs.geom.validation import *
from gefest.core.algs.postproc.resolve_errors import Rules, postprocess, validate
from gefest.core.configs.optimization_params import OptimizationParams
from gefest.core.geometry import Structure
from gefest.core.geometry.datastructs.point import Point
from gefest.core.geometry.datastructs.polygon import Polygon
from gefest.core.geometry.domain import Domain
from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.geometry.utils import get_random_structure
from gefest.core.opt.adapters.structure import StructureAdapter
from gefest.core.opt.operators.crossovers import (
    panmixis,
    polygon_level_crossover,
    structure_level_crossover,
)
from gefest.core.opt.operators.mutations import (
    add_point,
    add_poly,
    drop_point,
    drop_poly,
    mutate_structure,
    pos_change_point_mutation,
    resize_poly,
    rotate_poly,
)
from gefest.core.structure.prohibited import create_prohibited
from gefest.tools.estimators.simulators.sound_wave.sound_interface import (
    SoundSimulator,
    generate_map,
)
from gefest.tools.fitness import Fitness
from gefest.tools.samplers.sampler import Sampler

# pre domain params
grid_resolution_x = 300  # Number of points on x-axis
grid_resolution_y = 300  # Number of points on y-axis
coord_X = np.linspace(20, 100, grid_resolution_x + 1)  # X coordinate for spatial grid
coord_Y = np.linspace(20, 100, grid_resolution_y + 1)  # Y coordinate for spatial grid
grid = [grid_resolution_x, grid_resolution_y]  # points grid
fixed_area = [[[45, 55], [55, 55], [55, 45], [45, 45], [45, 55]]]

#  domain configuration
geometry = Geometry2D(is_closed=True, is_convex=True)
prohibited = create_prohibited(1, [], [], fixed_area=fixed_area)
domain = Domain(
    allowed_area=[
        (min(coord_X), min(coord_Y)),
        (min(coord_X), max(coord_Y)),
        (max(coord_X), max(coord_Y)),
        (max(coord_X), min(coord_Y)),
        (min(coord_X), min(coord_Y)),
    ],
    geometry=geometry,
    max_poly_num=5,
    min_poly_num=1,
    max_points_num=30,
    min_points_num=20,
    prohibited_area=prohibited,
    polygon_side=0.0001,
)


def poly_from_comsol_txt(path='figures/bottom_square.txt'):
    """

    Args:
        path: path to txt file with comsol points

    Returns:

    """
    res = pd.read_csv(path, sep=' ', header=None)
    points = [[int(round(res.iloc[i, 0], 2)), int(round(res.iloc[i, 1], 2))] for i in res.index]
    points = [Point(i[0], i[1]) for i in np.array(points)]
    poly = Polygon(points=points)
    struct = Structure(polygons=[poly])
    return struct


class SoundFieldFitness(Fitness):
    def __init__(self, domain, estimator, n_jobs, path_best_struct=None):
        super().__init__(domain, estimator)
        self.path_best_struct = path_best_struct
        self.n_jobs = n_jobs
        if self.path_best_struct is None:
            print('please, set up the best spl matrix into configuration')
            print('the best structure will be generated randomly')
            rnd_structure = get_random_structure(domain)
            best_spl = generate_map(domain, rnd_structure)
        else:
            best_structure = poly_from_comsol_txt(path_best_struct)
            best_spl = self.estimator(best_structure)
            best_spl = np.nan_to_num(best_spl, nan=0, neginf=0, posinf=0)

        self.best_spl = best_spl

    def fitness(self, ind: Structure):
        spl = self.estimator(ind)
        current_spl = np.nan_to_num(spl, nan=0, neginf=0, posinf=0)
        l_f = np.sum(np.abs(self.best_spl - current_spl)) / (120 * 120)
        return l_f


path_to_init_figure = f'figures/bottom_square.txt'
estimator = SoundFieldFitness(
    domain,
    SoundSimulator(domain, 10),
    -1,
    'F:\\Git_Repositories\\gef_ref\\GEFEST\\cases\\sound_waves\\figures\\bottom_square.txt',
)


#  RandomGraphFactory sampler wrap
class StructFactory(RandomGraphFactory):
    """Simple GEFEST sampler wrap for GOLEM RandomGraphFactory compatibility."""

    def __init__(
        self,
        sampler: Sampler,
        adapter: BaseOptimizationAdapter,
    ):
        self.sampler = sampler
        self.adapter = adapter

    def __call__(self, *args, **kwargs) -> OptGraph:
        samples = self.sampler(1)
        return self.adapter(samples[0])


if __name__ == '__main__':

    opt_params = OptimizationParams(
        crossovers=[
            partial(polygon_level_crossover, domain=domain),
            partial(structure_level_crossover, domain=domain),
        ],
        crossover_each_prob=[0.0, 1.0],
        mutations=[
            rotate_poly,
            resize_poly,
            add_point,
            drop_point,
            add_poly,
            drop_poly,
            pos_change_point_mutation,
        ],
        mutation_each_prob=[0.125, 0.125, 0.25, 0.25, 0.00, 0.00, 0.25],
        pair_selector=panmixis,
        postprocess_attempts=3,
        domain=domain,
        postprocessor=postprocess,
        estimator=estimator,
        postprocess_rules=[
            Rules.not_out_of_bounds.value,
            Rules.not_closed_polygon.value,
            Rules.not_self_intersects.value,
            Rules.not_too_close_polygons.value,
            Rules.not_overlaps_prohibited.value,
            Rules.not_too_close_points.value,
        ],
        extra=3,
        n_jobs=1,
        golem_adapter=StructureAdapter,
        tuner_cfg=None,
        n_steps=30,
        pop_size=10,
        log_dir='saved_pop',
        run_name='test',
    )

    objective = Objective(
        quality_metrics={
            type(opt_params.estimator).__name__: opt_params.estimator.fitness,
        },
    )

    requirements = GraphRequirements(
        early_stopping_timeout=5,
        early_stopping_iterations=1000,
        keep_n_best=4,
        # timeout=timeout,
        keep_history=True,
        num_of_generations=opt_params.n_steps,
        n_jobs=opt_params.n_jobs,
        history_dir='surrogate_logs',
    )

    ggp = GraphGenerationParams(
        adapter=opt_params.golem_adapter,
        rules_for_constraint=[
            partial(validate, rules=opt_params.postprocess_rules, domain=opt_params.domain),
        ],
        random_graph_factory=StructFactory(opt_params.sampler, opt_params.golem_adapter),
    )

    pop = list(map(opt_params.golem_adapter.adapt, opt_params.sampler(opt_params.pop_size)))

    def fake_crossover(struct1, struct2, *args, **kwargs):
        return struct1, struct2

    ########################################
    class MutationWrap:
        def __init__(self, mutations, mutation_chance, mutations_probs, domain):
            self.mutations = mutations
            self.mutation_chance = mutation_chance
            self.mutations_probs = mutations_probs
            self.domain = domain

        def __call__(self, struct, **kwargs):
            return mutate_structure(
                structure=struct,
                mutations=self.mutations,
                mutation_chance=self.mutation_chance,
                mutations_probs=self.mutations_probs,
                domain=self.domain,
            )

    gpparams = GPAlgorithmParameters(
        multi_objective=False,
        genetic_scheme_type=GeneticSchemeTypesEnum.generational,
        ########################################
        mutation_types=[
            MutationWrap(
                mutations=[mut],
                mutation_chance=opt_params.mutation_prob,
                mutations_probs=[1],
                domain=domain,
            )
            for mut in opt_params.mutations
        ],
        ########################################
        crossover_types=[fake_crossover],
        pop_size=opt_params.pop_size,
        max_pop_size=int(opt_params.pop_size * 1.5),
        crossover_prob=opt_params.crossover_prob,
        mutation_prob=1,
    )

    surrogate = SurrogateEachNgenOptimizer(
        objective=objective,
        initial_graphs=pop,
        requirements=requirements,
        graph_generation_params=ggp,
        graph_optimizer_params=gpparams,
        #  surrogate_model=,  # random
        surrogate_each_n_gen=8,
    )

    surrogate.optimise(objective=objective)
