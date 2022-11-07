from gefest.core.structure.structure import Structure
import matplotlib.pyplot as plt
from gefest.core.structure.point import Point
from gefest.core.algs.geom.validation import out_of_bound, too_close, intersection
from copy import deepcopy
from cases.breakwaters.main import optimized_pop, estimator, domain


class Breakwaters():

    def __init__(self) -> Structure:
        self.optimized_structure = optimized_pop[0]
        self.cost = estimator.estimate
        self.input_domain = domain

    def moving_position(self):
        structure = self.optimized_structure
        init_fitnes = round(self.cost([structure])[0][0], 3) #only high of wave in multicreterial loss

        fitnes_history = []
        structure_history = []
        polygon_history = []
        fitnes_history.append(init_fitnes)
        structure_history.append(structure)
        polygon_history.append('init_step')
        current_fitnes = init_fitnes

        for poly_num, poly in enumerate(structure.polygons):
            step_fitnes = 0
            max_attempts = 3

            if poly.id != 'fixed':
                moving_step = self.input_domain.geometry.get_length(polygon=poly)*0.2

                while step_fitnes <= current_fitnes and max_attempts > 0:
                    step_structure, step_fitnes = self._moving_for_one_step(structure=structure,
                                                                            poly_number=poly_num,
                                                                            moving_step=moving_step,
                                                                            init_fitnes=current_fitnes)
                    structure_history.append(step_structure)
                    fitnes_history.append(step_fitnes)
                    polygon_history.append('P' + str(poly_num) + ', step=' + str(round(moving_step, 1)))
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
                             init_fitnes) -> Structure:
        moved_poly = structure.polygons[poly_number]
        directions = ['north', 'south', 'east', 'west', 'n-w', 's-w', 'n-e', 's-e']
        results = {}

        for direct in directions:
            for idx, point in enumerate(moved_poly.points):
                moved_poly.points[idx] = self._moving_point(direct, point, moving_step)
                print(moved_poly.points[idx])

            tmp_structure = deepcopy(structure)
            tmp_structure.polygons[poly_number] = moved_poly
            fitnes = round(self.cost([tmp_structure])[0][0], 3)
            non_unvalid = not any([out_of_bound(tmp_structure, self.input_domain),
                                too_close(tmp_structure, self.input_domain),
                                intersection(tmp_structure, self.input_domain)])
            if fitnes < init_fitnes and non_unvalid:
                results[fitnes] = tmp_structure
                tmp_structure.plot(f'changed poly {poly_number} by direction {direct} TMP')
                # plt.savefig(f'tmp_struct- {poly_number} and {direct}.png')
                # plt.clf()
        if results:
            best_structure = results[min(results)]
            best_fitnes = min(results)
            # best_structure.plot(f'best struct for poly {poly_number}')
            # plt.savefig(f'best struct for poly {poly_number}')
            return best_structure, best_fitnes
        else:
            return structure, init_fitnes


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


breakw = Breakwaters()
fitnes, structure, poly = breakw.moving_position()
print('number of structures: ', len(structure))
print([struct.polygons for struct in structure])
descriptions = poly
x= list(range(len(poly)))
y= fitnes

fig, axd = plt.subplot_mosaic([['upper', 'upper'],
                               ['lower left', 'lower right']],
                               figsize=(10, 10), height_ratios=[1, 3])

fig.suptitle('SA of moving objects')

structure[0].plot(color = 'r', ax=axd['lower left'])
axd['lower left'].set_title('Initial structure')
structure[-1].plot(color='c', ax=axd['lower right'])
axd['lower right'].set_title('Processed structure')

axd['upper'].plot(fitnes, c='c')
axd['upper'].scatter(x,y, marker='o', c='c')
for idx,text in enumerate(descriptions):
    axd['upper'].annotate(text, (x[idx]+0.01, y[idx]+0.01), rotation=45.0)
axd['upper'].set_xlabel('iteration of senitivity analysis')
axd['upper'].set_ylabel('loss - height of waves')

fig.tight_layout()
fig.savefig('experiment_moving.png')

# for idx, obj_fit in enumerate(zip(structure, fitnes)):
#     obj_fit[0].plot(title=f'structure {idx} has fitnes {obj_fit[1]}')
#     plt.savefig(f'structure {idx}.png')
