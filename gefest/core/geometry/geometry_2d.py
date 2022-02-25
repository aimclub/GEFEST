from typing import List, Union
from uuid import uuid4

import bezier
import numpy as np
from shapely import affinity
from shapely.geometry import Point as GeomPoint, Polygon as GeomPolygon
from shapely.geometry.multipolygon import MultiPolygon as ShapelyMultiPolygon
from shapely.geometry.polygon import Polygon as ShapelyPolygon
from shapely.ops import nearest_points

from gefest.core.geometry.geometry import Geometry
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon


class Geometry2D(Geometry):
    def resize_poly(self, poly: Polygon, x_scale: float, y_scale: float):
        geom_polygon = self._poly_to_geom(poly)

        rescaled_geom_polygon = affinity.scale(geom_polygon,
                                               x_scale, y_scale)

        rescaled_points = [Point(x, y) for x, y in
                           zip(list(rescaled_geom_polygon.exterior.xy[0]),
                               list(rescaled_geom_polygon.exterior.xy[1]))]

        rescaled_poly = Polygon(polygon_id=poly.id,
                                points=rescaled_points)

        return rescaled_poly

    def rotate_poly(self, poly: Polygon, angle: float):
        geom_polygon = self._poly_to_geom(poly)

        rotated_geom_polygon = affinity.rotate(geom_polygon, angle, 'center')

        rotated_points = set((x, y) for x, y in
                             zip(list(rotated_geom_polygon.exterior.xy[0]),
                                 list(rotated_geom_polygon.exterior.xy[1])))
        rotated_poly = Polygon(polygon_id=poly.id,
                               points=[Point(*points) for points in rotated_points])

        return rotated_poly

    def get_square(self, polygon: 'Polygon'):
        if len(polygon.points) <= 1:
            return 0

        geom_polygon = self._poly_to_geom(polygon)

        return geom_polygon.area

    def is_contain_point(self, poly: 'Polygon', point: Point):
        geom_poly_allowed = self._poly_to_geom(poly)
        geom_pt = GeomPoint(point.x, point.y)
        return geom_poly_allowed.contains(geom_pt)

    def nearest_point(self, nearest_obj: Union[Point, Polygon], poly: Polygon) -> Point:
        geom_poly = self._poly_to_geom(poly)
        if isinstance(nearest_obj, Point):
            geom_nearest_obj = GeomPoint(nearest_obj.x, nearest_obj.y)
        elif isinstance(nearest_obj, Polygon):
            geom_nearest_obj = self._poly_to_geom(nearest_obj)
        _, nearest_correct_position = nearest_points(geom_nearest_obj, geom_poly)
        return Point(nearest_correct_position.x, nearest_correct_position.y)

    def _bezier_transform(self, poly: 'GeomPolygon') -> Polygon:
        convex_poly = self._poly_to_geom(poly).convex_hull
        points = convex_poly.boundary.xy

        x = points[0]
        y = points[1]
        z = np.asfortranarray([x, y])

        bezier_curve = bezier.Curve.from_nodes(z)
        number_of_points = len(poly.points)
        bezier_params = np.linspace(0, 1, number_of_points)

        transform_poly = Polygon(polygon_id=str(uuid4()),
                                 points=[(Point(bezier_curve.evaluate(param)[0][0], bezier_curve.evaluate(param)[1][0]))
                                         for param in bezier_params])

        transform_geom = self._poly_to_geom(transform_poly)

        return transform_geom

    def get_convex(self, poly: 'Polygon', *args, method='bezier') -> Polygon:
        method_variants = {'bezier': self._bezier_transform(poly)}

        if method in method_variants.keys():
            geom_convex = method_variants[method]
            convex_points = []
            if isinstance(geom_convex, ShapelyMultiPolygon):
                geom_convex = geom_convex[0]
            if isinstance(geom_convex, ShapelyPolygon) and len(poly.points) > 2:
                for convex_pt in [(x, y) for x, y in zip(geom_convex.exterior.coords.xy[0],
                                                         geom_convex.exterior.coords.xy[1])]:
                    convex_points.append(Point(*convex_pt))
            return Polygon(poly.id, convex_points)
        else:
            raise KeyError(f'Unknown method: "{method}", use one of followed: {list(method_variants.keys())}')

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
        geom_pt_2 = self._pt_to_geom(pt_2)

        distance = geom_pt_1.distance(geom_pt_2)

        return distance


def create_circle(struct: 'Structure') -> 'Structure':
    geom = Geometry2D()
    poly = struct.polygons[0]

    num_points = len(poly.points)
    radius = geom.get_length(struct.polygons[0]) / (2 * np.pi)

    x = [pt.x for pt in poly.points]
    y = [pt.y for pt in poly.points]

    center_x = round((max(x) + min(x)) / 2, 2)
    center_y = round((max(y) + min(y)) / 2, 2)

    theta = np.linspace(0, 2 * np.pi, num_points)
    a = radius * np.cos(theta) + center_x + 2.2 * radius
    b = radius * np.sin(theta) + center_y

    struct = [Polygon(polygon_id=str(uuid4()),
                      points=[(Point(x, y)) for x, y in zip(a, b)])]

    return struct
