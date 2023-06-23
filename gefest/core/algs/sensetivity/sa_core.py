import itertools
import time
from copy import deepcopy
import matplotlib.pyplot as plt

from gefest.core.structure.structure import Structure
from gefest.core.structure.point import Point
from gefest.core.algs.geom.validation import out_of_bound, too_close, intersection


class SA_methods():
    """Base class consists transformation methods for sensitivity-based optimization
    """

    def __init__(self, optimized_pop, estimator, domain, *args, **kwargs):
        self.optimized_structure = optimized_pop[0]
        self.cost = estimator.estimate
        self.input_domain = domain
        self.sa_time_history = [0]
        self.start_time = time.time()

    @property
    def get_time_history(self):
        '''Returns time history of optimization process
        '''
        return self.sa_time_history

    def moving_position(self):
        '''Analysis of moving polygons around by different distances 
        '''
        structure = self.optimized_structure
        print(structure)
        for poly_num, poly in enumerate(structure.polygons):
            poly.id = 'poly_' + str(poly_num)
        init_fitnes = round(self.cost([structure])[0], 3) #only high of wave in multicreterial loss

        fitnes_history = []
        structure_history = []
        polygon_history = []
        fitnes_history.append(init_fitnes)
        structure_history.append(structure)
        polygon_history.append(f'init_moving, fitnes={init_fitnes}')
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
                    end_step_time = time.time()
                    self.sa_time_history.append(end_step_time - self.start_time)

                    if worse_res:
                        fitnes_diff = round(100 * ((worse_res - current_fitnes)/current_fitnes), 1)
                        polygon_history.append(f'{str(poly.id)}, step={round(moving_step)},\
                                                fitnes=+{str(fitnes_diff)}%')
                    else:
                        fitnes_diff = round(100 * ((step_fitnes - current_fitnes)/current_fitnes), 1)
                        polygon_history.append(f'{str(poly.id)}, step={round(moving_step)}, fitnes={str(fitnes_diff)}%')

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
            fitnes = round(self.cost([tmp_structure])[0], 3)
            non_unvalid = not any([out_of_bound(tmp_structure, self.input_domain),
                                too_close(tmp_structure, self.input_domain),
                                intersection(tmp_structure, self.input_domain)])
            if fitnes < init_fitnes and non_unvalid:
                results[fitnes] = tmp_structure
            elif fitnes >= init_fitnes and non_unvalid:
                worse_results[fitnes] = tmp_structure
            else:
                worse_results[init_fitnes] = tmp_structure

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
        '''Analysis of polygons necessity, looking for the best combination of polys
        '''

        best_fitnes = []
        best_structures = []
        best_description = []

        fitnes_history = []
        structure_history = []
        polygon_history = []

        fitnes_history.append(init_fitnes)
        structure_history.append(structure)
        polygon_history.append(f'init_combinations, fitnes={init_fitnes}')
        end_step_time = time.time()
        self.sa_time_history.append(end_step_time - self.start_time)

        for length in range(1, len(structure.polygons)+1):

            for unique_comb in itertools.combinations(structure.polygons, length):

                tmp_structure = deepcopy(structure)
                tmp_structure.polygons = list(unique_comb)
                fitnes = round(self.cost([tmp_structure])[0], 3)

                structure_history.append(tmp_structure)
                fitnes_history.append(init_fitnes)
                fitnes_diff = round(100 * ((fitnes - init_fitnes)/init_fitnes), 1)

                if fitnes <= init_fitnes*1.01:

                    best_fitnes.append(fitnes)
                    best_structures.append(tmp_structure)

                    ids = []
                    for polygon in tmp_structure.polygons:
                        ids.append(polygon.id)
                    polygon_history.append(f'{str(ids)}, fitnes={str(fitnes_diff)}%')
                    best_description.append(f'{str(ids)}, fitnes={str(fitnes_diff)}%')
                else:
                    ids = []
                    for polygon in tmp_structure.polygons:
                        ids.append(polygon.id)
                    polygon_history.append(f'{str(ids)}, fitnes=+{str(fitnes_diff)}%')

                end_step_time = time.time()
                self.sa_time_history.append(end_step_time - self.start_time)

        if min(best_fitnes) < init_fitnes:
            best_samples = list(zip(best_fitnes, best_structures, best_description))
            best_samples.sort(key = lambda x : x[0])
            finish_sample = best_samples[0]
        else:
            length =[len(struct.polygons) for struct in best_structures]
            best_samples = list(zip(best_fitnes, best_structures, best_description, length))
            best_samples.sort(key = lambda x : x[3])
            finish_sample = best_samples[0][:-1]

        fitnes_history.append(finish_sample[0])
        structure_history.append(finish_sample[1])
        polygon_history.append(finish_sample[2])

        return fitnes_history, structure_history, polygon_history

    def removing_points(self, structure: Structure, init_fitnes):
        fitnes_history = []
        structure_history = []
        polygon_history = []

        fitnes_history.append(init_fitnes)
        structure_history.append(structure)
        polygon_history.append(f'init_removing_points, fitnes={init_fitnes}')
        end_step_time = time.time()
        self.sa_time_history.append(end_step_time - self.start_time)

        current_fitnes = init_fitnes

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

                        tmp_structure.polygons[poly_number] = tmp_polygon

                        fitnes = round(self.cost([tmp_structure])[0], 3)
                        fitnes_diff = round(100 * ((fitnes - current_fitnes)/current_fitnes), 1)

                        structure_history.append(tmp_structure)

                        if fitnes <= current_fitnes*1.01:
                            current_fitnes = fitnes
                            new_polygon = tmp_polygon
                            fitnes_history.append(fitnes)
                            polygon_history.append(f'{str(polygon.id)}, del={str(point.coords())},\
                                                    fitnes={str(fitnes_diff)}%')
                        else:
                            fitnes_history.append(current_fitnes)
                            polygon_history.append(f'{str(polygon.id)}, del={str(point.coords())},\
                                                    fitnes=+{str(fitnes_diff)}%')

                        end_step_time = time.time()
                        self.sa_time_history.append(end_step_time - self.start_time)
                    new_structure = tmp_structure
                else:
                    for point in polygon.points:
                        tmp_polygon = deepcopy(new_polygon)

                        tmp_points = tmp_polygon.points
                        tmp_points.remove(point)
                        tmp_polygon.points = tmp_points

                        tmp_structure.polygons[poly_number] = tmp_polygon

                        fitnes = round(self.cost([tmp_structure])[0], 3)
                        fitnes_diff = round(100 * ((fitnes - current_fitnes)/current_fitnes), 1)

                        structure_history.append(tmp_structure)

                        if fitnes <= current_fitnes*1.01:
                            current_fitnes = fitnes
                            new_polygon = tmp_polygon
                            fitnes_history.append(fitnes)
                            polygon_history.append(f'{str(polygon.id)}, del={str(point.coords())},\
                                                    fitnes={str(fitnes_diff)}%')
                        else:
                            fitnes_history.append(current_fitnes)
                            polygon_history.append(f'{str(polygon.id)}, del={str(point.coords())},\
                                                    fitnes=+{str(fitnes_diff)}%')

                        end_step_time = time.time()
                        self.sa_time_history.append(end_step_time - self.start_time)
                    new_structure = tmp_structure

        return fitnes_history, structure_history, polygon_history

    def rotate_objects(self, structure: Structure, init_fitnes: int):
        '''Analysis of rotating polygons
        '''
        rotate_func = self.input_domain.geometry.rotate_poly
        fitnes_history = []
        structure_history = []
        polygon_history = []

        fitnes_history.append(init_fitnes)
        structure_history.append(structure)
        polygon_history.append(f'init_rotates, fitnes={init_fitnes}')
        end_step_time = time.time()
        self.sa_time_history.append(end_step_time - self.start_time)

        curent_fitnes = init_fitnes

        for poly_num, poly in enumerate(structure.polygons):
            tmp_fit_history = []
            tmp_str_history = []

            angles = list(range(45, 360, 45))

            if poly.id != 'fixed':
                for angle in angles:
                    tmp_structure = deepcopy(structure)
                    rotated_poly = deepcopy(poly)

                    rotated_poly = rotate_func(rotated_poly, angle=angle)
                    tmp_structure.polygons[poly_num] = rotated_poly

                    fitnes = round(self.cost([tmp_structure])[0], 3)
                    tmp_fit_history.append([fitnes, angle])
                    tmp_str_history.append(tmp_structure)

                best_poly_fit = min(tmp_fit_history)
                idx_best = tmp_fit_history.index(best_poly_fit)
                fitnes_diff = round(100 * ((best_poly_fit[0] - curent_fitnes)/curent_fitnes), 1)

                if best_poly_fit[0] < curent_fitnes:
                    curent_fitnes = best_poly_fit[0]
                    best_tmp_structure = tmp_str_history[idx_best]
                    structure.polygons[poly_num] = best_tmp_structure.polygons[poly_num]
                    fitnes_history.append(best_poly_fit[0])
                    structure_history.append(best_tmp_structure)
                    polygon_history.append(f'{str(poly.id)}, best_angle={best_poly_fit[1]}, fitness={fitnes_diff}%')
                else:
                    fitnes_history.append(curent_fitnes)
                    structure_history.append(tmp_str_history[idx_best])
                    polygon_history.append(f'{str(poly.id)}, best_angle={best_poly_fit[1]}, fitness=+{fitnes_diff}%')

                end_step_time = time.time()
                self.sa_time_history.append(end_step_time - self.start_time)

        best_fitnes = min(fitnes_history)
        best_idx = fitnes_history.index(best_fitnes)
        best_structure = structure_history[best_idx]

        fitnes_history.append(best_fitnes)
        structure_history.append(best_structure)
        polygon_history.append('best_structure after rotating polygons')

        end_step_time = time.time()
        self.sa_time_history.append(end_step_time - self.start_time)

        return fitnes_history, structure_history, polygon_history


class SA(SA_methods):
    """The class for doing sensitivity-based optimization for structures

        Parameters:
            optimized_pop: optimized ''Structure'' from generative design process
            estimator: physical process simulator (SWAN, COMSOL, etc.)
            domain: ''Domain'' class (same with initial ''Domain'' in the previous gen. design process)

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def analysis(self):
        """Method for sensitivity-based optimization 

        Returns:
            List: fitnes history, structure history, description for an each step of analysis,
              time history
        """

        mov_fitnes, mov_structure, mov_poly = self.moving_position()
        rotated_fitnes, rotated_structure, rotated_poly = self.rotate_objects(mov_structure[-1], mov_fitnes[-1])
        del_fitnes, del_structure, del_poly = self.exploring_combinations(rotated_structure[-1], rotated_fitnes[-1])
        rm_points_fitnes, rm_points_structure, rm_points_poly = self.removing_points(del_structure[-1], del_fitnes[-1])

        fitnes_history = mov_fitnes + rotated_fitnes + del_fitnes + rm_points_fitnes
        structure_history = mov_structure + rotated_structure + del_structure + rm_points_structure
        poly_history = mov_poly + rotated_poly + del_poly + rm_points_poly

        time_history = self.get_time_history

        return fitnes_history, structure_history, poly_history, time_history

    @property
    def get_improved_structure(self):
        """Getter method if needed to recieve only improved structure

        Returns:
            Structure: improved structure by sensitivity-based optimization
        """

        _, structure, _, _ = self.analysis()

        return structure[-1]


def report_viz(analysis_result):
    """Generates a picture-report of sensitivity-based optimization

    Args:
        analysis_result (List): results of sensitivity analysis of structure (from ''SA.analysis()'')
    """

    fitnes_history = analysis_result[0]
    structure_history = analysis_result[1]
    descriptions = analysis_result[2]
    time_history = analysis_result[3]

    initial_strucutre = structure_history[0]
    optimized_structure = structure_history[-1]

    x = list(range(len(descriptions)))
    y = fitnes_history

    spend_time = round(time_history[-1]-time_history[0])

    start_fit = fitnes_history[0]
    end_fit = fitnes_history[-1]
    fitnes_difference = round(100*(start_fit - end_fit)/start_fit, 1)

    fig, axd = plt.subplot_mosaic([['upper', 'upper'],
                                        ['lower left', 'lower right']],
                                        figsize=(15, 8), height_ratios=[1, 3])

    fig.suptitle(f'Sensitivity-based optimization report, spend={spend_time}sec,\
                  fitnes improved on {fitnes_difference}%')

    initial_strucutre.plot(color = 'r', ax=axd['lower left'], legend=True)
    axd['lower left'].set_title(f'Initial structure, fitnes={round(fitnes_history[0], 3)}')
    optimized_structure.plot(ax=axd['lower right'], legend=True)
    axd['lower right'].set_title(f'Processed structure, fitnes={round(fitnes_history[-1], 3)}')

    axd['upper'].plot(fitnes_history, c='c')
    axd['upper'].scatter(x,y, marker='o', c='c')
    for idx,text in enumerate(descriptions):
        axd['upper'].annotate(text, (x[idx]+0.01, y[idx]+0.01), rotation=45.0)
    axd['upper'].set_xlabel('iteration of senitivity analysis')
    axd['upper'].set_ylabel('loss - height of waves')

    fig.tight_layout()
    plt.legend()
    fig.savefig('sensitivity_report.png')
