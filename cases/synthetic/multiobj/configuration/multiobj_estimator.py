from shapely.geometry import Polygon, Point
from types import SimpleNamespace

from gefest.core.structure.structure import Structure
from gefest.tools.estimators.estimator import Estimator
from gefest.core.structure.point import Point

target = Point(150, 150)


def configurate_estimator(domain):
    # ------------
    # User-defined estimator
    # it should be created as object with .estimate() method
    # ------------
    def len_num_ratio(struct: Structure):
        if len(struct.polygons) > 1 or len(struct.polygons[0].points) < 3:
            return [100000, 100000]
        else:
            poly = struct.polygons[0]

            l_dist = domain.geometry.min_distance(poly, target)
            length = domain.geometry.get_length(poly)

            is_cont = domain.geometry.is_contain_point(poly=poly,
                                                       point=target)

        if not is_cont:
            length += 10500

        objective = [length, l_dist]

        return objective

    estimator = SimpleNamespace()
    estimator.estimate = len_num_ratio

    # ------------
    # GEFEST estimator
    # ------------

    estimator = Estimator(estimator=estimator)

    return estimator
