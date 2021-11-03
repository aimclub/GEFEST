from abc import abstractmethod
from math import sqrt

from shapely import affinity
from shapely.geometry import Point as GeomPoint
from shapely.geometry import Polygon as GeomPolygon

from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon


class Geometry:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def resize_poly(self, poly: Polygon, x_scale: float, y_scale: float):
        pass

    @abstractmethod
    def rotate_poly(self, poly: Polygon, angle: float):
        pass

    def get_length(self, polygon: 'Polygon'):
        if len(polygon.points) < 1:
            return 0

        total_length = 0
        for i in range(1, len(polygon.points)):
            total_length += sqrt(
                (polygon.points[i - 1].x - polygon.points[i].x) ** 2 +
                (polygon.points[i - 1].y - polygon.points[i].y) ** 2 +
                (polygon.points[i - 1].z - polygon.points[i].z) ** 2)

        total_length += sqrt(
            (polygon.points[len(polygon.points) - 1].x - polygon.points[0].x) ** 2 +
            (polygon.points[len(polygon.points) - 1].y - polygon.points[0].y) ** 2 +
            (polygon.points[len(polygon.points) - 1].z - polygon.points[0].z) ** 2)

        return total_length

    @abstractmethod
    def get_square(self, polygon: 'Polygon'):
        pass

    @abstractmethod
    def is_contain_point(self, poly: 'Polygon', point: 'Point'):
        pass

    def _poly_to_geom(self, poly: Polygon):
        pass

class Geometry2D:
    def _poly_to_geom(self, poly: Polygon):
        return GeomPolygon([GeomPoint(pt.x, pt.y) for pt in poly.points])

    def resize_poly(self, poly: Polygon, x_scale: float, y_scale: float):
        geom_polygon = self._poly_to_geom(poly)

        rescaled_geom_polygon = affinity.scale(geom_polygon,
                                               x_scale, y_scale)

        poly.points = [Point(x, y) for x, y in
                       zip(list(rescaled_geom_polygon.exterior.xy[0]),
                           list(rescaled_geom_polygon.exterior.xy[1]))]

        return poly

    def rotate_poly(self, poly: Polygon, angle: float):
        geom_polygon = self._poly_to_geom(poly)

        rotated_geom_polygon = affinity.rotate(geom_polygon, angle, 'center')

        poly.points = [Point(x, y) for x, y in
                       zip(list(rotated_geom_polygon.exterior.xy[0]),
                           list(rotated_geom_polygon.exterior.xy[1]))]

        return poly

    def get_square(self, polygon: 'Polygon'):
        if len(polygon.points) <= 1:
            return 0

        geom_polygon = self._poly_to_geom(polygon)

        return geom_polygon.area

    def is_contain_point(self, poly: 'Polygon', point: 'Point'):
        geom_poly_allowed = self._poly_to_geom(poly)
        geom_pt = GeomPoint(point.x, point.y)
        return geom_poly_allowed.contains(geom_pt)
