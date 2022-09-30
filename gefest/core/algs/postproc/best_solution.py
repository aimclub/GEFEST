from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.structure.polygon import Polygon
from gefest.core.structure.structure import Structure
from gefest.core.structure.point import Point
from cases.breakwaters.one_segment import cost, optimized_structure
from gefest.core.algs.geom.validation import out_of_bound, too_close, intersection
import random

class Breakwaters():

    def __init__(self) -> Structure:
        self.optimized_structure = optimized_structure.best_structure
        self.cost = cost

    def moving_position(self):
        structure = self.optimized_structure
        best_fitnes = self.cost(structure)
        number_poly = len(structure.polygons)

        max_attempts = 50

        while max_attempts > 0:

            choosen_poly = random.choice(range(number_poly))
            moving_step = Geometry2D.get_square(structure.polygons[choosen_poly])*0.001

            best_structure, best_fitnes = _moving_for_one_step(structure=structure,
                                                               poly_number=choosen_poly,
                                                               moving_step=moving_step,
                                                               cost=self.cost,
                                                               current_fitnes=best_fitnes)

            return best_structure, best_fitnes


def _moving_for_one_step(structure: Structure, poly_number: int, moving_step, cost, current_fitnes) -> Structure:
    moved_poly = structure.polygons[poly_number]
    directions = {'north': Point(point.x + moving_step, point.y),
                  'south': Point(point.x - moving_step, point.y),
                  'east': Point(point.x, point.y + moving_step),
                  'west': Point(point.x, point.y - moving_step),
                  'n-w': Point(point.x + moving_step, point.y - moving_step),
                  's-w': Point(point.x - moving_step, point.y - moving_step),
                  'n-e': Point(point.x + moving_step, point.y + moving_step),
                  's-e': Point(point.x - moving_step, point.y - moving_step)}
    results = {}

    for direct in directions.keys():
        for idx, point in enumerate(moved_poly.points):
            moved_poly.points[idx] = directions[direct]

        tmp_structure = structure.copy()
        tmp_structure.polygons[poly_number] = moved_poly
        fitnes = cost(tmp_structure)
        non_unvalid = not any([out_of_bound(tmp_structure),
                               too_close(tmp_structure),
                               intersection(tmp_structure)])
        if fitnes < current_fitnes and non_unvalid:
            results[moved_poly] = fitnes

    best_poly = max(results, key=results.get)
    best_fitnes = max(results.values())
    if best_poly:
        structure.polygons[poly_number] = moved_poly

    return structure, best_fitnes


breakw = Breakwaters()
breakw_best = breakw.moving_position()
print(breakw_best)
