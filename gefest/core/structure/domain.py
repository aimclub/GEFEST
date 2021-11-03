from typing import List, Optional, Tuple

from shapely.geometry import Point as GeomPoint, Polygon as GeomPolygon

from gefest.core.structure.polygon import PolygonPoint


class Domain:
    def __init__(self, name='main', allowed_area: Optional[List[Tuple]] = None,
                 max_poly_num=4, min_dist=15, fixed_points: Optional[List[Tuple]] = None):
        self.name = name
        if allowed_area is None:
            allowed_area = [(-125, 100),
                            (-75, 155),
                            (15, 155),
                            (40, 90),
                            (-10, -130),
                            (-10, -155),
                            (-125, -155)]
        else:
            allowed_area = [(int(round(p[0], 0)), int(round(p[1], 0))) for p in allowed_area]
        self.allowed_area = allowed_area
        self.max_poly_num = max_poly_num
        self.min_dist = min_dist
        self.fixed_points = [PolygonPoint(p[0], p[1]) for p in fixed_points] \
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

    def contains(self, point: PolygonPoint):
        geom_poly_allowed = GeomPolygon([GeomPoint(pt[0], pt[1]) for pt in self.allowed_area])
        geom_pt = GeomPoint(point.x, point.y)
        return geom_poly_allowed.contains(geom_pt)

    def as_geom(self):
        if self.allowed_area is None or len(self.allowed_area) <= 2:
            raise ValueError('Not enough points for domain')
        return GeomPolygon([GeomPoint(pt[0], pt[1]) for pt in self.allowed_area])
