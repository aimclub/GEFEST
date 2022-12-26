from ast import Break
from enum import unique
from shutil import move
from numpy import poly1d
import sys
sys.path.append('C:/Users/user2/GEFEST')

import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
import itertools

from gefest.core.structure.structure import Structure
from gefest.core.structure.point import Point
from gefest.core.algs.geom.validation import out_of_bound, too_close, intersection
from gefest.core.opt.gen_design import design
from cases.breakwaters.configuration_de import bw_domain
from cases.breakwaters.configuration_spea2 import bw_optimizer, bw_sampler, bw_estimator
from cases.main_conf import opt_params

opt_params.path_to_sim = False
opt_params.path_to_sur = False

domain, task_setup = bw_domain.configurate_domain(poly_num=opt_params.n_polys,
                                                  points_num=opt_params.n_points,
                                                  is_closed=opt_params.is_closed)

estimator = bw_estimator.configurate_estimator(domain=domain,
                                               path_sim=opt_params.path_to_sim)

# sampler = bw_sampler.configurate_sampler(domain=domain)

# optimizer = bw_optimizer.configurate_optimizer(pop_size=opt_params.pop_size,
#                                                crossover_rate=opt_params.c_rate,
#                                                mutation_rate=opt_params.m_rate,
#                                                task_setup=task_setup)

# optimized_population = design(n_steps=opt_params.n_steps,
#                        pop_size=opt_params.pop_size,
#                        estimator=estimator,
#                        sampler=sampler,
#                        optimizer=optimizer,
#                        extra=True)

file_name = "optimized.pkl"
# open_file = open(file_name, "wb")
# pickle.dump(optimized_population, open_file)
# open_file.close()

open_file = open(file_name, "rb")
optimized_pop = pickle.load(open_file)
open_file.close()

class SA_methods():

    def __init__(self):
        self.optimized_structure = optimized_pop[0]
        self.cost = estimator.estimate
        self.input_domain = domain

    def moving_position(self):
        structure = self.optimized_structure
        for poly_num, poly in enumerate(structure.polygons):
            poly.id = 'poly_' + str(poly_num)
        init_fitnes = round(self.cost([structure])[0][0], 3) #only high of wave in multicreterial loss

        fitnes_history = []
        structure_history = []
        polygon_history = []
        fitnes_history.append(init_fitnes)
        structure_history.append(structure)
        polygon_history.append('init_moving')
        current_fitnes = init_fitnes

        for poly_num, poly in enumerate(structure.polygons):
            step_fitnes = 0
            max_attempts = 3

            if poly.id != 'fixed':
                moving_step = self.input_domain.geometry.get_length(polygon=poly)*0.2

                while step_fitnes <= current_fitnes and max_attempts > 0:
                    step_structure, step_fitnes, worse_res = self._moving_for_one_step(structure=structure,
                                                                            poly_number=poly_num,
                                                                            moving_step=moving_step,
                                                                            init_fitnes=current_fitnes)
                    structure_history.append(step_structure)
                    fitnes_history.append(step_fitnes)
                    
                    if worse_res:
                        fitnes_diff = round(100 * ((worse_res - current_fitnes)/current_fitnes), 1)
                        polygon_history.append(str(poly.id) + ', cannot be moved, fitnes +' + str(fitnes_diff) + '%')
                    else:
                        fitnes_diff = round(100 * ((step_fitnes - current_fitnes)/current_fitnes), 1)
                        polygon_history.append(str(poly.id) + ', fitnes ' + str(fitnes_diff) + '%')

                    if step_fitnes >= current_fitnes:
                        max_attempts -= 1
                        moving_step = moving_step/2
                    else:
                        current_fitnes = step_fitnes
                        structure = deepcopy(step_structure)

        return fitnes_history, structure_history, polygon_history

    def _moving_for_one_step(self,
                             structure: Structure,
                             poly_number: int,
                             moving_step,
                             init_fitnes):
        moved_init_poly = structure.polygons[poly_number]
        directions = ['north', 'south', 'east', 'west', 'n-w', 's-w', 'n-e', 's-e']
        results = {}
        worse_results = {}

        for direct in directions:
            moved_poly = deepcopy(moved_init_poly)
            for idx, point in enumerate(moved_poly.points):
                moved_poly.points[idx] = self._moving_point(direct, point, moving_step)

            tmp_structure = deepcopy(structure)
            tmp_structure.polygons[poly_number] = moved_poly
            fitnes = round(self.cost([tmp_structure])[0][0], 3)
            non_unvalid = not any([out_of_bound(tmp_structure, self.input_domain),
                                too_close(tmp_structure, self.input_domain),
                                intersection(tmp_structure, self.input_domain)])
            if fitnes < init_fitnes and non_unvalid:
                results[fitnes] = tmp_structure
            else:
                worse_results[fitnes] = tmp_structure

        if results:
            best_structure = results[min(results)]
            best_fitnes = min(results)
            return best_structure, best_fitnes, 0
        else:
            best_worse_fitnes = min(worse_results)
            return structure, init_fitnes, best_worse_fitnes


    def _moving_point(self, direction: str, point: Point, moving_step) -> Point:
        directions = {'north': Point(point.x + moving_step, point.y),
                      'south': Point(point.x - moving_step, point.y),
                      'east': Point(point.x, point.y + moving_step),
                      'west': Point(point.x, point.y - moving_step),
                      'n-w': Point(point.x + moving_step, point.y - moving_step),
                      's-w': Point(point.x - moving_step, point.y + moving_step),
                      'n-e': Point(point.x + moving_step, point.y + moving_step),
                      's-e': Point(point.x - moving_step, point.y - moving_step)}
        return directions[direction]

    def exploring_combinations(self, structure: Structure, init_fitnes):
        current_fitnes = init_fitnes

        fitnes_history = []
        structure_history = []
        polygon_history = []

        fitnes_history.append(init_fitnes)
        structure_history.append(structure)
        polygon_history.append('init_del')

        for length in range(1, len(structure.polygons)+1):

            for unique_comb in itertools.combinations(structure.polygons, length):

                tmp_structure = deepcopy(structure)
                tmp_structure.polygons = unique_comb
                fitnes = round(self.cost([tmp_structure])[0][0], 3)

                if fitnes < current_fitnes:
                    fitnes_diff = round(100 * ((fitnes - current_fitnes)/current_fitnes), 1)
                    current_fitnes = fitnes
                    structure_history.append(tmp_structure)
                    fitnes_history.append(fitnes)
                    ids = []
                    for polygon in tmp_structure.polygons:
                        ids.append(polygon.id)
                    polygon_history.append(str(ids) + ', fitnes ' + str(fitnes_diff) + '%')
                else:
                    structure_history.append(tmp_structure)
                    fitnes_history.append(current_fitnes)
                    fitnes_diff = round(100 * ((fitnes - current_fitnes)/current_fitnes), 1)
                    ids = []
                    for polygon in tmp_structure.polygons:
                        ids.append(polygon.id)
                    polygon_history.append(str(ids) + ', cannot be used, fitnes +' + str(fitnes_diff) + '%')

        return fitnes_history, structure_history, polygon_history


    def removing_points(self, structure: Structure, init_fitnes):
        fitnes_history = []
        structure_history = []
        polygon_history = []

        fitnes_history.append(init_fitnes)
        structure_history.append(structure)
        polygon_history.append('init_rm_points')

        current_fitnes = init_fitnes

        for poly_number, polygon in enumerate(structure.polygons):

            closed = bool(polygon.points[0] == polygon.points[-1])

            if closed:
                for idx_point in range(1, len(polygon.points)):
                    exploring_polygon = deepcopy(polygon)
                    tmp_structure = deepcopy(structure)

                    exploring_point_coords = tmp_structure.polygons[poly_number].points[idx_point].coords()
                    exploring_polygon.points.pop(idx_point)
                    tmp_structure.polygons[poly_number] = exploring_polygon

                    fitnes = round(self.cost([tmp_structure])[0][0], 3)

                    if fitnes < current_fitnes:
                        current_fitnes = fitnes

                        structure_history.append(tmp_structure)
                        fitnes_history.append(fitnes)
                        polygon_history.append(str(polygon.id) + ', del_point=' + str(exploring_point_coords))

            else:
                for idx, point in enumerate(polygon.points):
                    exploring_polygon = deepcopy(polygon)
                    tmp_structure = deepcopy(structure)

                    exploring_point_coords = point.coords()
                    exploring_polygon.points.pop(idx)
                    tmp_structure.polygons[poly_number] = exploring_polygon

                    fitnes = round(self.cost([tmp_structure])[0][0], 3)

                    if fitnes < current_fitnes:
                        current_fitnes = fitnes

                        structure_history.append(tmp_structure)
                        fitnes_history.append(fitnes)
                        polygon_history.append(str(polygon.id) + ', del_point=' + str(exploring_point_coords))

        return fitnes_history, structure_history, polygon_history


class SA(SA_methods):

    def __init__(self) -> Structure:
        super().__init__()
        
    def analysis(self):
        mov_fitnes, mov_structure, mov_poly = self.moving_position()
        del_fitnes, del_structure, del_poly = self.exploring_combinations(mov_structure[-1], mov_fitnes[-1])
        rm_points_fitnes, rm_point_structure, rm_points_poly = self.removing_points(del_structure[-1], del_fitnes[-1])

        fitnes_history = mov_fitnes + del_fitnes + rm_points_fitnes
        structure_history = mov_structure + del_structure + rm_point_structure
        poly_history = mov_poly + del_poly + rm_points_poly
        
        return fitnes_history, structure_history, poly_history


sensitivity = SA()
fitnes, structure, poly = sensitivity.analysis()
descriptions = poly
x= list(range(len(poly)))
y= fitnes

fig, axd = plt.subplot_mosaic([['upper', 'upper'],
                               ['lower left', 'lower right']],
                               figsize=(10, 10), height_ratios=[1, 3])

fig.suptitle('SA of moving objects')

structure[0].plot(color = 'r', ax=axd['lower left'], legend=True)
axd['lower left'].set_title('Initial structure')
structure[-1].plot(ax=axd['lower right'], legend=True)
axd['lower right'].set_title('Processed structure')

axd['upper'].plot(fitnes, c='c')
axd['upper'].scatter(x,y, marker='o', c='c')
for idx,text in enumerate(descriptions):
    axd['upper'].annotate(text, (x[idx]+0.01, y[idx]+0.01), rotation=45.0)
axd['upper'].set_xlabel('iteration of senitivity analysis')
axd['upper'].set_ylabel('loss - height of waves')

fig.tight_layout()
fig.savefig('experiment_moving.png')
