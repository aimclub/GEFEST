from typing import List, Optional, Tuple

from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon


class Domain:
    def __init__(self, name='main', allowed_area: Optional[List[Tuple]] = None,
                 max_poly_num=4, min_poly_num=2,
                 max_points_num=50, min_points_num=20,
                 fixed_points=None,
                 is_closed=True,
                 geometry=None):
        self.name = name
        self.is_closed = is_closed
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
        self.min_poly_num = min_poly_num

        self.max_points_num = max_points_num
        self.min_points_num = min_points_num

        self.min_dist = max(self.max_x - self.min_x, self.max_y - self.min_y) / 35

        self.fixed_points = [Polygon(polygon_id='fixed', points=[Point(p[0], p[1]) for p in poly]) for poly in
                             fixed_points] \
            if fixed_points is not None else []

    @property
    def min_x(self):
        return min(p[0] for p in self.allowed_area)

    @property
    def max_x(self):
        return max(p[0] for p in self.allowed_area)

    @property
    def min_y(self):
        return min(p[1] for p in self.allowed_area)

    @property
    def max_y(self):
        return max(p[1] for p in self.allowed_area)

    @property
    def len_x(self):
        return abs(self.max_x - self.min_x)

    @property
    def len_y(self):
        return abs(self.max_y - self.min_y)

    def contains(self, point: Point):
        geom_poly_allowed = Polygon(polygon_id=f'bnd_{self.name}',
                                    points=[Point(pt[0], pt[1]) for pt in self.allowed_area])
        return self.geometry.is_contain_point(geom_poly_allowed, point)

    @property
    def bound_poly(self):
        if self.allowed_area is None or len(self.allowed_area) <= 2:
            raise ValueError('Not enough points for domain')
        bnd_points = [Point(*pt_coords) for pt_coords in self.allowed_area]
        return Polygon(polygon_id=f'bnd_{self.name}', points=bnd_points)
