import itertools
import time
from copy import deepcopy

import matplotlib.pyplot as plt

from gefest.core.geometry import Point, Structure
from gefest.core.opt.objective.objective_eval import ObjectivesEvaluator
from gefest.core.opt.postproc.validation import validate


class SensitivityAnalysis:
    """Base class consists transformation methods for sensitivity-based optimization."""

    def __init__(self, optimized_pop, opt_params):
        self.optimized_structure = optimized_pop[0]
        self.opt_params = opt_params
        self.cost = ObjectivesEvaluator(
            opt_params.objectives,
            opt_params.estimation_n_jobs,
        )
        self.domain = opt_params.domain
        self.sa_time_history = [0]
        self.start_time = time.time()

    @property
    def get_time_history(self):
        """Returns time history of optimization process."""
        return self.sa_time_history

    def moving_position(self):
        """Analysis of moving polygons around by different distances."""
        structure = self.optimized_structure
        print(structure)
        for poly_num, poly in enumerate(structure.polygons):
            poly.id = 'poly_' + str(poly_num)

        structure = self.cost(structure)[0]
        init_fitness = round(structure.fitness[0], 3)  # only high of wave in multicreterial loss

        fitness_history = []
        structure_history = []
        polygon_history = []
        fitness_history.append(init_fitness)
        structure_history.append(structure)
        polygon_history.append(f'init_moving, fitness={init_fitness}')
        current_fitness = init_fitness

        for poly_num, poly in enumerate(structure.polygons):
            step_fitness = 0
            max_attempts = 10

            if poly.id != 'fixed':
                moving_step = self.domain.geometry.get_length(polygon=poly) * 0.2

                while step_fitness <= current_fitness and max_attempts > 0:
                    step_structure, step_fitness, worse_res = self._moving_for_one_step(
                        structure=structure,
                        poly_number=poly_num,
                        moving_step=moving_step,
                        init_fitness=current_fitness,
                    )
                    structure_history.append(step_structure)
                    fitness_history.append(step_fitness)
                    end_step_time = time.time()
                    self.sa_time_history.append(end_step_time - self.start_time)

                    if worse_res:
                        diff = round(
                            100 * ((worse_res - current_fitness) / current_fitness), 1,
                        )
                        polygon_history.append(
                            f'{str(poly.id)}, step={round(moving_step)}, fitness=+{str(diff)}%'
                        )
                    else:
                        diff = round(
                            100 * ((step_fitness - current_fitness) / current_fitness), 1,
                        )
                        polygon_history.append(
                            f'{str(poly.id)}, step={round(moving_step)}, fitness={str(diff)}%'
                        )

                    if step_fitness >= current_fitness:
                        max_attempts -= 1
                        moving_step = moving_step / 2
                    else:
                        current_fitness = step_fitness
                        structure = deepcopy(step_structure)

        return fitness_history, structure_history, polygon_history

    def _moving_for_one_step(
        self, structure: Structure, poly_number: int, moving_step, init_fitness
    ):
        moved_init_poly = structure.polygons[poly_number]
        directions = ['north', 'south', 'east', 'west', 'n-w', 's-w', 'n-e', 's-e']
        results = {}
        worse_results = {}

        for direct in directions:
            moved_poly = deepcopy(moved_init_poly)
            for idx, point in enumerate(moved_poly.points):
                moved_poly.points[idx] = self._moving_point(direct, point, moving_step)

            tmp_structure = deepcopy(structure)
            tmp_structure[poly_number] = moved_poly
            tmp_structure = self.cost(tmp_structure)[0]
            fitness = round(tmp_structure.fitness[0], 3)
            non_invalid = validate(tmp_structure, self.opt_params.postprocess_rules, self.domain)

            if fitness < init_fitness and non_invalid:
                results[fitness] = tmp_structure
            elif fitness >= init_fitness and non_invalid:
                worse_results[fitness] = tmp_structure
            else:
                worse_results[init_fitness] = tmp_structure

        if results:
            best_structure = results[min(results)]
            best_fitness = min(results)
            return best_structure, best_fitness, 0
        else:
            best_worse_fitness = min(worse_results)
            return structure, init_fitness, best_worse_fitness

    def _moving_point(self, direction: str, point: Point, moving_step) -> Point:
        directions = {
            'north': Point(point.x + moving_step, point.y),
            'south': Point(point.x - moving_step, point.y),
            'east': Point(point.x, point.y + moving_step),
            'west': Point(point.x, point.y - moving_step),
            'n-w': Point(point.x + moving_step, point.y - moving_step),
            's-w': Point(point.x - moving_step, point.y + moving_step),
            'n-e': Point(point.x + moving_step, point.y + moving_step),
            's-e': Point(point.x - moving_step, point.y - moving_step),
        }
        return directions[direction]

    def exploring_combinations(self, structure: Structure, init_fitness):
        """Analysis of polygons necessity, looking for the best combination of polys."""
        best_fitness = []
        best_structures = []
        best_description = []

        fitness_history = []
        structure_history = []
        polygon_history = []

        fitness_history.append(init_fitness)
        structure_history.append(structure)
        polygon_history.append(f'init_combinations, fitness={init_fitness}')
        end_step_time = time.time()
        self.sa_time_history.append(end_step_time - self.start_time)

        for length in range(1, len(structure.polygons) + 1):
            for unique_comb in itertools.combinations(structure.polygons, length):
                tmp_structure = deepcopy(structure)
                tmp_structure.polygons = list(unique_comb)
                tmp_structure = self.cost(tmp_structure)[0]
                fitness = round(tmp_structure.fitness[0], 3)

                structure_history.append(tmp_structure)
                fitness_history.append(init_fitness)
                diff = round(100 * ((fitness - init_fitness) / init_fitness), 1)

                if fitness <= init_fitness * 1.01:
                    best_fitness.append(fitness)
                    best_structures.append(tmp_structure)

                    ids = []
                    for polygon in tmp_structure.polygons:
                        ids.append(polygon.id)

                    polygon_history.append(f'{str(ids)}, fitness={str(diff)}%')
                    best_description.append(f'{str(ids)}, fitness={str(diff)}%')
                else:
                    ids = []
                    for polygon in tmp_structure.polygons:
                        ids.append(polygon.id)

                    polygon_history.append(f'{str(ids)}, fitness=+{str(diff)}%')

                end_step_time = time.time()
                self.sa_time_history.append(end_step_time - self.start_time)

        if min(best_fitness) < init_fitness:
            best_samples = list(zip(best_fitness, best_structures, best_description))
            best_samples.sort(key=lambda x: x[0])
            finish_sample = best_samples[0]
        else:
            length = [len(struct.polygons) for struct in best_structures]
            best_samples = list(zip(best_fitness, best_structures, best_description, length))
            best_samples.sort(key=lambda x: x[3])
            finish_sample = best_samples[0][:-1]

        fitness_history.append(finish_sample[0])
        structure_history.append(finish_sample[1])
        polygon_history.append(finish_sample[2])

        return fitness_history, structure_history, polygon_history

    def removing_points(self, structure: Structure, init_fitness):
        """Analysis of the points removal."""
        fitness_history = []
        structure_history = []
        polygon_history = []

        fitness_history.append(init_fitness)
        structure_history.append(structure)
        polygon_history.append(f'init_removing_points, fitness={init_fitness}')
        end_step_time = time.time()
        self.sa_time_history.append(end_step_time - self.start_time)

        current_fitness = init_fitness

        new_structure = deepcopy(structure)

        for poly_number, polygon in enumerate(structure.polygons):
            if len(polygon.points) > 2:
                closed = bool(polygon.points[0] == polygon.points[-1])

                new_polygon = deepcopy(polygon)
                tmp_structure = deepcopy(new_structure)

                if closed:
                    for point in polygon.points[1:-1]:
                        tmp_polygon = deepcopy(new_polygon)

                        tmp_points = tmp_polygon.points
                        tmp_points.remove(point)
                        tmp_polygon.points = tmp_points

                        tmp_structure[poly_number] = tmp_polygon
                        tmp_structure = self.cost(tmp_structure)[0]
                        fitness = round(tmp_structure.fitness[0], 3)

                        diff = round(
                            100 * ((fitness - current_fitness) / current_fitness), 1,
                        )

                        structure_history.append(tmp_structure)

                        if fitness <= current_fitness * 1.01:
                            current_fitness = fitness
                            new_polygon = tmp_polygon
                            fitness_history.append(fitness)
                            polygon_history.append(
                                f'{str(polygon.id)}, del={str(point.coords)}, fitness={str(diff)}'
                            )
                        else:
                            fitness_history.append(current_fitness)
                            polygon_history.append(
                                f'{str(polygon.id)}, del={str(point.coords)}, fitness=+{str(diff)}%'
                            )

                        end_step_time = time.time()
                        self.sa_time_history.append(end_step_time - self.start_time)

                    new_structure = tmp_structure
                else:
                    for point in polygon.points:
                        tmp_polygon = deepcopy(new_polygon)

                        tmp_points = tmp_polygon.points
                        tmp_points.remove(point)
                        tmp_polygon.points = tmp_points

                        tmp_structure[poly_number] = tmp_polygon

                        tmp_structure = self.cost(tmp_structure)[0]
                        fitness = round(tmp_structure.fitness[0], 3)
                        diff = round(
                            100 * ((fitness - current_fitness) / current_fitness), 1,
                        )

                        structure_history.append(tmp_structure)

                        if fitness <= current_fitness * 1.01:
                            current_fitness = fitness
                            new_polygon = tmp_polygon
                            fitness_history.append(fitness)
                            polygon_history.append(
                                f'{str(polygon.id)}, del={str(point.coords)}, fitness={str(diff)}%'
                            )
                        else:
                            fitness_history.append(current_fitness)
                            polygon_history.append(
                                f'{str(polygon.id)}, del={str(point.coords)}, fitness=+{str(diff)}%'
                            )

                        end_step_time = time.time()
                        self.sa_time_history.append(end_step_time - self.start_time)

                    new_structure = tmp_structure

        return fitness_history, structure_history, polygon_history

    def rotate_objects(self, structure: Structure, init_fitness: int):
        """Analysis of rotating polygons."""
        rotate_func = self.domain.geometry.rotate_poly
        fitness_history = []
        structure_history = []
        polygon_history = []

        fitness_history.append(init_fitness)
        structure_history.append(structure)
        polygon_history.append(f'init_rotates, fitness={init_fitness}')
        end_step_time = time.time()
        self.sa_time_history.append(end_step_time - self.start_time)

        curent_fitness = init_fitness

        for poly_num, poly in enumerate(structure.polygons):
            tmp_fit_history = []
            tmp_str_history = []

            angles = list(range(45, 360, 45))

            if poly.id != 'fixed':
                for angle in angles:
                    tmp_structure = deepcopy(structure)
                    rotated_poly = deepcopy(poly)

                    rotated_poly = rotate_func(rotated_poly, angle=angle)
                    tmp_structure[poly_num] = rotated_poly
                    tmp_structure = self.cost(tmp_structure)[0]
                    fitness = round(tmp_structure.fitness[0], 3)
                    tmp_fit_history.append([fitness, angle])
                    tmp_str_history.append(tmp_structure)

                best_poly_fit = min(tmp_fit_history)
                idx_best = tmp_fit_history.index(best_poly_fit)
                diff = round(
                    100 * ((best_poly_fit[0] - curent_fitness) / curent_fitness), 1,
                )

                if best_poly_fit[0] < curent_fitness:
                    curent_fitness = best_poly_fit[0]
                    best_tmp_structure = tmp_str_history[idx_best]
                    structure[poly_num] = best_tmp_structure.polygons[poly_num]
                    fitness_history.append(best_poly_fit[0])
                    structure_history.append(best_tmp_structure)
                    polygon_history.append(
                        f'{str(poly.id)}, best_angle={best_poly_fit[1]}, fitnesss={diff}%'
                    )
                else:
                    fitness_history.append(curent_fitness)
                    structure_history.append(tmp_str_history[idx_best])
                    polygon_history.append(
                        f'{str(poly.id)}, best_angle={best_poly_fit[1]}, fitnesss=+{diff}%'
                    )

                end_step_time = time.time()
                self.sa_time_history.append(end_step_time - self.start_time)

        best_fitness = min(fitness_history)
        best_idx = fitness_history.index(best_fitness)
        best_structure = structure_history[best_idx]

        fitness_history.append(best_fitness)
        structure_history.append(best_structure)
        polygon_history.append('best_structure after rotating polygons')

        end_step_time = time.time()
        self.sa_time_history.append(end_step_time - self.start_time)

        return fitness_history, structure_history, polygon_history

    def analysis(self):
        """Method for sensitivity-based optimization.

        Returns:
            List: fitness history, structure history, description for an each step of analysis,
              time history
        """
        mov_fitness, mov_structure, mov_poly = self.moving_position()
        rotated_fitness, rotated_structure, rotated_poly = self.rotate_objects(
            mov_structure[-1], mov_fitness[-1]
        )
        del_fitness, del_structure, del_poly = self.exploring_combinations(
            rotated_structure[-1], rotated_fitness[-1]
        )
        rm_points_fitness, rm_points_structure, rm_points_poly = self.removing_points(
            del_structure[-1], del_fitness[-1]
        )

        fitness_history = mov_fitness + rotated_fitness + del_fitness + rm_points_fitness
        structure_history = mov_structure + rotated_structure + del_structure + rm_points_structure
        poly_history = mov_poly + rotated_poly + del_poly + rm_points_poly

        time_history = self.get_time_history

        return fitness_history, structure_history, poly_history, time_history

    @property
    def get_improved_structure(self):
        """Getter method if needed to recieve only improved structure.

        Returns:
            Structure: improved structure by sensitivity-based optimization
        """
        _, structure, _, _ = self.analysis()

        return structure[-1]


def report_viz(analysis_result):
    """Plots a picture-report of sensitivity-based optimization."""
    fitness_history = analysis_result[0]
    descriptions = analysis_result[2]
    time_history = analysis_result[3]

    x = list(range(len(descriptions)))
    y = fitness_history

    spend_time = round(time_history[-1] - time_history[0])

    start_fit = fitness_history[0]
    end_fit = fitness_history[-1]
    difference = round(100 * (start_fit - end_fit) / start_fit, 1)

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f'SA report, spend={spend_time}sec, fitness improved on {difference}%'
    )

    plt.plot(fitness_history, c='c')
    plt.scatter(x, y, marker='o', c='c')
    for idx, text in enumerate(descriptions):
        plt.annotate(text, (x[idx], y[idx]), rotation=75.0)

    plt.xlabel('iteration of senitivity analysis')
    plt.ylabel('fitness')
    fig.tight_layout()

    fig.savefig('sensitivity_report.png')
