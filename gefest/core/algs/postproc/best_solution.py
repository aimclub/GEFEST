from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.structure.polygon import Polygon
from gefest.core.structure.structure import Structure
from gefest.core.structure.point import Point
from cases.breakwaters.one_segment import cost, optimized_structure


class Breakwaters():

    def __init__(self) -> Structure:
        self.optimized_structure = optimized_structure.best_structure
        self.cost = cost

    def moving_position(self):
        structure = optimized_structure
        init_fitnes = cost(structure)
        directions = ['north', 'south', 'east', 'west',
                      'n-w', 's-w', 'n-e', 's-e']

        best_structures = {}
        best_fitnes = init_fitnes

        for poly_id, poly in enumerate(structure.polygons):
            moving_step = Geometry2D.get_square(poly)*0.001
            fitnes = best_fitnes

            for direction in directions:
                best_directions = {}
                moved_poly_to_dir = _moving_for_one_step(poly, moving_step, direction)
                moved_structure = structure.copy()
                moved_structure.polygons[poly_id] = moved_poly_to_dir
                fitnes_dir = cost(moved_structure)

                if fitnes_dir < fitnes:
                    best_directions[direction] = fitnes_dir

            if best_directions:
                fitnes_tmp = fitnes
                best_dir = min(best_directions, key=best_directions.get)
                moved_structure = structure.copy()
                while fitnes_tmp < fitnes:

                    moved_poly_to_dir = _moving_for_one_step(poly, moving_step, best_dir)              
                    moved_structure.polygons[poly_id] = moved_poly_to_dir
                    fitnes_tmp = cost(moved_structure)
                    if fitnes_tmp < fitnes:
                        fitnes = fitnes_tmp
                        poly = moved_poly_to_dir

            best_structures[moved_structure] = fitnes

            best_structure = min(best_structures, key=best_structures.get)
            best_fitnes = best_structures[best_structure]

            return init_fitnes, best_fitnes, best_structure


def _moving_for_one_step(poly: Polygon, moving_step, direction):
    moved_poly = poly.copy()
    directions = {'north': Point(point.x + moving_step, point.y),
                  'south': Point(point.x - moving_step, point.y),
                  'east': Point(point.x, point.y + moving_step),
                  'west': Point(point.x, point.y - moving_step),
                  'n-w': Point(point.x + moving_step, point.y - moving_step),
                  's-w': Point(point.x - moving_step, point.y - moving_step),
                  'n-e': Point(point.x + moving_step, point.y + moving_step),
                  's-e': Point(point.x - moving_step, point.y - moving_step)}

    for idx, point in enumerate(moved_poly.points):
        moved_poly.points[idx] = directions[direction]

    return moved_poly


breakw = Breakwaters()
breakw_best = breakw.moving_position()
print(breakw_best)
