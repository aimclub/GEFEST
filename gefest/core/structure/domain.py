from typing import List, Optional, Tuple

from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon


class Domain:
    def __init__(self, name='main', allowed_area: Optional[List[Tuple]] = None,
                 max_poly_num=4, min_dist=15, fixed_points: Optional[List[Tuple]] = None,
                 geometry=None):
        self.name = name
        if geometry is None:
            self.geometry = Geometry2D()
        else:
            self.geometry = geometry

        if allowed_area is None:
            allowed_area = [(0, 0),
                            (0, 100),
                            (100, 100),
                            (100, 0)]
        else:
            allowed_area = [(int(round(p[0], 0)), int(round(p[1], 0))) for p in allowed_area]

        self.allowed_area = allowed_area

        self.max_poly_num = max_poly_num
        self.min_dist = min_dist

        self.fixed_points = [Point(p[0], p[1]) for p in fixed_points] \
            if fixed_points is not None else []

    @property
    def min_x(self):
        return min(p[0] for p in self.allowed_area) + self.min_dist

    @property
    def max_x(self):
        return max(p[0] for p in self.allowed_area) - self.min_dist

    @property
    def min_y(self):
        return min(p[1] for p in self.allowed_area) + self.min_dist

    @property
    def max_y(self):
        return max(p[1] for p in self.allowed_area) - self.min_dist

    @property
    def len_x(self):
        return abs(self.max_x - self.min_x)

    @property
    def len_y(self):
        return abs(self.max_y - self.min_y)

    @property
    def bound_poly(self):
        if self.allowed_area is None or len(self.allowed_area) <= 2:
            raise ValueError('Not enough points for domain')
        bnd_points = [Point(*pt_coords) for pt_coords in self.allowed_area]
        return Polygon(polygon_id=f'bnd_{self.name}', points=bnd_points)
