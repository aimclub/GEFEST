from shapely import affinity
from shapely.geometry import Point as GeomPoint, Polygon as GeomPolygon
from shapely.geometry.multipolygon import MultiPolygon as ShapelyMultiPolygon
from shapely.geometry.polygon import Polygon as ShapelyPolygon
from shapely.ops import nearest_points
from typing import List

from gefest.core.geometry.geometry import Geometry
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon


class Geometry2D(Geometry):
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

    def is_contain_point(self, poly: 'Polygon', point: Point):
        geom_poly_allowed = self._poly_to_geom(poly)
        geom_pt = GeomPoint(point.x, point.y)
        return geom_poly_allowed.contains(geom_pt)

    def nearest_point(self, point: Point, poly: Polygon) -> Point:
        geom_poly = self._poly_to_geom(poly)
        geom_point = GeomPoint(point.x, point.y)
        _, nearest_correct_position = nearest_points(geom_point, geom_poly)
        return Point(nearest_correct_position.x, nearest_correct_position.y)

    def nearest_points(self, poly_1: Polygon, poly_2: Polygon) -> List[Point]:
        geom_poly_1 = self._poly_to_geom(poly_1)
        geom_poly_2 = self._poly_to_geom(poly_2)

        _, nearest_correct_position = nearest_points(geom_poly_1, geom_poly_2)
        return [Point(pos.x, pos.y) for pos in nearest_correct_position]

    def get_convex(self, poly: 'Polygon') -> Polygon:
        geom_poly = self._poly_to_geom(poly)
        geom_convex = geom_poly.buffer(1)

        convex_points = []
        if isinstance(geom_convex, ShapelyMultiPolygon):
            geom_convex = geom_convex[0]
        if isinstance(geom_convex, ShapelyPolygon) and len(poly.points) > 2:
            for convex_pt in [(x, y) for x, y in zip(geom_convex.exterior.coords.xy[0],
                                                     geom_convex.exterior.coords.xy[1])]:
                convex_points.append(Point(*convex_pt))
        return Polygon(poly.id, convex_points)

    def intersects(self, poly_1: 'Polygon', poly_2: 'Polygon') -> bool:
        geom_poly_1 = self._poly_to_geom(poly_1)
        geom_poly_2 = self._poly_to_geom(poly_2)
        return geom_poly_1.intersects(geom_poly_2)

    def _poly_to_geom(self, poly: Polygon):
        return GeomPolygon([self._pt_to_geom(pt) for pt in poly.points])

    def _pt_to_geom(self, pt: Point):
        return GeomPoint(pt.x, pt.y)

    def distance(self, pt_1: Point, pt_2: Point) -> float:
        geom_pt_1 = self._pt_to_geom(pt_1)
        geom_pt_2 = self._pt_to_geom(pt_1)

        distance = geom_pt_1.distance(geom_pt_2)

        return distance
