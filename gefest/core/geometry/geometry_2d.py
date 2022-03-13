from typing import List
from uuid import uuid4

import bezier
import numpy as np
from shapely import affinity
from shapely.geometry import Point as GeomPoint, Polygon as GeomPolygon, LineString, MultiLineString
from shapely.ops import nearest_points

from gefest.core.geometry.geometry import Geometry
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon


class Geometry2D(Geometry):
    """
    Overriding the geometry base class for 2D structures.
    The input receives information about the closeness of the polygon
    """

    def __init__(self,
                 is_closed=True):
        self.is_closed = is_closed

    def get_coords(self, poly):
        # Transformation from shapely coords to GEFEST points for further analysis
        if isinstance(poly, GeomPolygon):
            # Converting  shapely.Polygon to shapely.LineString translation for correct conversion
            poly = LineString(poly.exterior.coords)
        if self.is_closed or len(poly.coords.xy[0]) < 3:
            points = [Point(x, y) for x, y in
                      zip(list(poly.coords.xy[0]),
                          list(poly.coords.xy[1]))]
        else:
            # For open polygons, the last point is ignored
            points = [Point(x, y) for x, y in
                      zip(list(poly.coords.xy[0][:-1]),
                          list(poly.coords.xy[1][:-1]))]

        return points

    def resize_poly(self, poly: Polygon, x_scale: float, y_scale: float):
        """
        Function for rescaling polygons along each axis.
        Scaling occurs relative to the center of mass of the polygon
        """

        geom_polygon = self._poly_to_geom(poly)  # Transformation to shapely structure

        rescaled_geom_polygon = affinity.scale(geom_polygon,
                                               x_scale, y_scale)  # Scaling along each axis

        rescaled_points = self.get_coords(rescaled_geom_polygon)

        rescaled_poly = Polygon(polygon_id=poly.id,
                                points=rescaled_points)  # Back transformation to GEFEST polygon

        return rescaled_poly

    def rotate_poly(self, poly: Polygon, angle: float):
        """
        Rotating polygon relative to the center of mass by a given angle
        """

        geom_polygon = self._poly_to_geom(poly)  # Transformation to shapely structure

        rotated_geom_polygon = affinity.rotate(geom_polygon, angle, 'center')  # Rotating the entire polygon

        rotated_points = self.get_coords(rotated_geom_polygon)
        rotated_poly = Polygon(polygon_id=poly.id,
                               points=rotated_points)  # Back transformation to GEFEST polygon

        return rotated_poly

    def get_square(self, polygon: 'Polygon'):
        # Area of two-dimension polygon using area method from shapely

        if len(polygon.points) <= 1:
            return 0

        # Transformation to shapely.polygon, cause LineString does not have an area method
        geom_polygon = GeomPolygon([self._pt_to_geom(pt) for pt in polygon.points])

        return geom_polygon.area

    def is_contain_point(self, poly: 'Polygon', point: Point):
        # Checking if a point is inside a polygon
        geom_poly_allowed = GeomPolygon([self._pt_to_geom(pt) for pt in poly.points])
        geom_pt = GeomPoint(point.x, point.y)
        return geom_poly_allowed.contains(geom_pt)

    def nearest_point(self, point: Point, poly: Polygon) -> Point:
        # Calculating closest point between input point and polygon.
        geom_poly = self._poly_to_geom(poly)
        geom_point = GeomPoint(point.x, point.y)
        _, nearest_correct_position = nearest_points(geom_point, geom_poly)  # One point as output
        return Point(nearest_correct_position.x, nearest_correct_position.y)

    def nearest_points(self, poly_1: Polygon, poly_2: Polygon) -> List[Point]:
        # Calculating closest point between two polygons
        geom_poly_1 = self._poly_to_geom(poly_1)
        geom_poly_2 = self._poly_to_geom(poly_2)

        _, nearest_correct_position = nearest_points(geom_poly_1, geom_poly_2)  # Set of points as output
        return [Point(pos.x, pos.y) for pos in nearest_correct_position]

    def bezier_transform(self, poly: 'Polygon') -> Polygon:
        """
        Function for bezier transformation over the polygon.
        The polygon is transformed into a convex spherical figure without self-intersections.
        Such transformation might be useful if you are working with round-shaped figures
        """

        poly = GeomPolygon([self._pt_to_geom(pt) for pt in poly.points])  # Transform to shapely Polygon
        points = LineString(poly.boundary).xy  # Getting points of polygon

        x = points[0]
        y = points[1]
        z = np.asfortranarray([x, y])  # Create a sequence of control points needed to define a bezier curve

        bezier_curve = bezier.Curve.from_nodes(z)  # Bezier curve on a sequence of control points
        number_of_points = len(poly.points)
        bezier_params = np.linspace(0, 1, number_of_points)  # Values to put as bezier parameters

        transform_poly = Polygon(polygon_id=str(uuid4()),
                                 points=[(Point(bezier_curve.evaluate(param)[0][0], bezier_curve.evaluate(param)[1][0]))
                                         for param in
                                         bezier_params])  # Bezier transformation as GEFEST polygon

        transform_geom = GeomPolygon([self._pt_to_geom(pt) for pt in transform_poly.points])
        return transform_geom

    def get_convex(self, poly: 'Polygon') -> Polygon:
        # Obtaining a convex polygon to avoid intersections
        if len(poly.points) < 3:
            return poly
        geom_poly = self._poly_to_geom(poly).convex_hull
        points = self.get_coords(geom_poly)
        polygon = Polygon(polygon_id='tmp', points=points)

        return polygon

    def get_centroid(self, poly: 'Polygon'):
        # Getting a point that is the center of mass of the polygon
        points = [pt for pt in poly.points]
        if len(points) < 3:
            points.append(points[0])
        geom_poly = GeomPolygon([self._pt_to_geom(pt) for pt in points])
        geom_point = geom_poly.centroid
        point = Point(geom_point.x, geom_point.y)
        return point

    def intersects(self, structure: 'Structure') -> bool:
        """
        Function to check for any intersection in structure of polygons
        Whole structure appears like shapely MultiLineString for which uses method is simple
        """
        polygons = structure.polygons
        multi_geom = MultiLineString([self._poly_to_geom(poly) for poly in polygons])
        return multi_geom.is_simple

    def intersects_poly(self, poly_1: 'Polygon', poly_2: 'Polygon') -> bool:
        # Intersection between two polygons
        geom_poly_1 = self._poly_to_geom(poly_1)
        geom_poly_2 = self._poly_to_geom(poly_2)
        return geom_poly_1.intersects(geom_poly_2)

    def _poly_to_geom(self, poly: Polygon):
        # Transformation GEFEST polygon to shapely LineString
        return LineString([self._pt_to_geom(pt) for pt in poly.points])

    def _pt_to_geom(self, pt: Point):
        # Transformation GEFEST point to shapely Point
        return GeomPoint(pt.x, pt.y)

    def min_distance(self, poly_1: 'Polygon', poly_2: 'Polygon') -> float:
        # Smallest distance between two polygons
        geom_poly_1 = self._poly_to_geom(poly_1)
        geom_poly_2 = self._poly_to_geom(poly_2)

        distance = geom_poly_1.distance(geom_poly_2)

        return distance

    def centroid_distance(self, point: 'Point', poly: 'Polygon'):
        # Distance from point to polygon
        geom_point = self._pt_to_geom(point)
        geom_poly = self._poly_to_geom(poly)
        d = geom_point.distance(geom_poly)

        return d


# Function to create a circle, needed for one of the synthetic examples
def create_circle(struct: 'Structure') -> 'Structure':
    geom = Geometry2D(is_closed=False)
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
