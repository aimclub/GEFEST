from typing import List
from uuid import uuid4

import numpy as np
from shapely import affinity
from shapely.geometry import Point as GeomPoint, Polygon as GeomPolygon, LineString, MultiLineString
from shapely.ops import nearest_points

from gefest.core.geometry.geometry import Geometry
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon


class Geometry2D(Geometry):
    """Overriding the geometry base class for 2D structures.
    The input receives information about the closeness of the polygon
    Args:
        is_closed: ``True`` if the :obj:`Polygon` must have close borders
            (first Point is equal to the last one), otherwise ``False``. Default value is ``True``
    """

    def __init__(self,
                 is_closed=True):
        self.is_closed = is_closed

    def get_coords(self, poly) -> List[Point]:
        """The function for getting points
        Args:
            poly: :obj:`Polygon` for processing
        Returns:
            all :obj:`Point` that :obj:`poly`contains
        """

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

    def resize_poly(self, poly: Polygon, x_scale: float, y_scale: float) -> Polygon:
        """The function for rescaling polygons along each axis.
        Scaling occurs relative to the center of mass of the polygon
        Args:
            poly: :obj:`Polygon` for processing
            x_scale: scale value for **x** axis
            y_scale: scale value for **y** axis
        Returns:
            scaled :obj:`poly` by ``(x,y)`` axes
        """
        geom_polygon = self._poly_to_geom(poly)  # Transformation to shapely structure

        rescaled_geom_polygon = affinity.scale(geom_polygon,
                                               x_scale, y_scale)  # Scaling along each axis

        rescaled_points = self.get_coords(rescaled_geom_polygon)

        rescaled_poly = Polygon(polygon_id=poly.id,
                                points=rescaled_points)  # Back transformation to GEFEST polygon

        return rescaled_poly

    def rotate_poly(self, poly: Polygon, angle: float) -> Polygon:
        """Rotating polygon relative to the center of mass by a given angle
        Args:
            poly: :obj:`Polygon` for processing
            angle: value of degree rotation
        Returns:
            rotated :obj:`poly`
        """

        geom_polygon = self._poly_to_geom(poly)  # Transformation to shapely structure

        rotated_geom_polygon = affinity.rotate(geom_polygon, angle, 'center')  # Rotating the entire polygon

        rotated_points = self.get_coords(rotated_geom_polygon)
        rotated_poly = Polygon(polygon_id=poly.id,
                               points=rotated_points)  # Back transformation to GEFEST polygon

        return rotated_poly

    def get_square(self, polygon: 'Polygon') -> float:
        """Recieving value of the area
        Args:
            polygon: :obj:`Polygon` for processing
        Returns:
            value of the :obj:`polygon` area
        """

        if len(polygon.points) <= 1:
            return 0

        # Transformation to shapely.polygon, cause LineString does not have an area method
        geom_polygon = GeomPolygon([self._pt_to_geom(pt) for pt in polygon.points])

        return geom_polygon.area

    def is_contain_point(self, poly: 'Polygon', point: Point) -> bool:
        """Checking if a point is inside a polygon
        Args:
            poly: :obj:`Polygon` that explore
            point: :obj:`Point` for checking presence inside the :obj:`Polygon`
        Returns:
            ``True`` if :obj:`point` is into :obj:`poly`, otherwise ``False``
        """
        geom_poly_allowed = GeomPolygon([self._pt_to_geom(pt) for pt in poly.points])
        geom_pt = GeomPoint(point.x, point.y)

        return geom_poly_allowed.contains(geom_pt)

    def nearest_point(self, point: Point, poly: Polygon) -> Point:
        """Calculating closest point between input point and polygon.
        Args:
            point: the :obj:`Point` that explore
            poly: the :obj:`Polygon` that explore
        Returns:
            returns the nearest :obj:`Point` from ``point`` among all points in the ``poly``
        """
        geom_poly = self._poly_to_geom(poly)
        geom_point = GeomPoint(point.x, point.y)
        _, nearest_correct_position = nearest_points(geom_point, geom_poly)  # One point as output
        return Point(nearest_correct_position.x, nearest_correct_position.y)

    def nearest_points(self, poly_1: Polygon, poly_2: Polygon) -> List[Point]:
        """Calculating closest point between two polygons
        Args:
            poly_1: the first :obj:`Polygon` that explore
            poly_2: the second :obj:`Polygon` that explore
        Returns:
            the couple of :obj:`Point` where the first one from :obj:`poly_1` and the second one from :obj:`poly_2`
        """
        geom_poly_1 = self._poly_to_geom(poly_1)
        geom_poly_2 = self._poly_to_geom(poly_2)

        _, nearest_correct_position = nearest_points(geom_poly_1, geom_poly_2)  # Set of points as output
        return nearest_correct_position

    def get_convex(self, poly: 'Polygon') -> Polygon:
        """Obtaining a convex polygon to avoid intersections
        Args:
            poly: :obj:`Polygon` for processing
        Returns:
            convex :obj:`Polygon`
        """
        if len(poly.points) < 3:
            return poly
        geom_poly = self._poly_to_geom(poly).convex_hull
        points = self.get_coords(geom_poly)
        polygon = Polygon(polygon_id='tmp', points=points)

        return polygon

    def get_centroid(self, poly: 'Polygon') -> Point:
        """Getting a point that is the center of mass of the polygon
        Args:
            poly: the :obj:`Polygon` that explore
        Returns:
            central :obj:`Point` of :obj:`poly`
        """
        points = [pt for pt in poly.points]
        if len(points) < 3:
            points.append(points[0])
        geom_poly = GeomPolygon([self._pt_to_geom(pt) for pt in points])
        geom_point = geom_poly.centroid
        point = Point(geom_point.x, geom_point.y)
        return point

    def intersects(self, structure: 'Structure') -> bool:
        """Function to check for any intersection in structure of polygons
        Whole structure appears like shapely MultiLineString for which uses method is simple
        Args:
            structure: the :obj:`Structure` that explore
        Returns:
            ``True`` if any :obj:`Polygon` in :obj:`structure` intersects with another one,
               otherwise - ``False``
        """
        polygons = structure.polygons
        multi_geom = MultiLineString([self._poly_to_geom(poly) for poly in polygons])
        return multi_geom.is_simple

    def contains(self, poly1: 'Polygon', poly2: 'Polygon') -> bool:
        geom_polygon1 = self._poly_to_geom(poly1)
        geom_polygon2 = GeomPolygon([self._pt_to_geom(pt) for pt in poly2.points])

        is_contain = geom_polygon2.contains(geom_polygon1)
        return is_contain

    def intersects_poly(self, poly_1: 'Polygon', poly_2: 'Polygon') -> bool:
        """Intersection between two polygons
        Args:
            poly_1: the first :obj:`Polygon` that explore
            poly_2: the second :obj:`Polygon` that explore
        Returns:
            ``True`` if the :obj:`poly_1` intersects with :obj:`poly_2`, otherwise - ``False``
        """
        geom_poly_1 = self._poly_to_geom(poly_1)
        geom_poly_2 = self._poly_to_geom(poly_2)
        return geom_poly_1.intersects(geom_poly_2)

    def _poly_to_geom(self, poly: Polygon) -> LineString:
        # Transformation GEFEST polygon to shapely LineString
        return LineString([self._pt_to_geom(pt) for pt in poly.points])

    def _pt_to_geom(self, pt: Point) -> GeomPoint:
        # Transformation GEFEST point to shapely Point
        return GeomPoint(pt.x, pt.y)

    def min_distance(self, obj_1, obj_2) -> float:
        """Smallest distance between two objects
        Args:
            obj_1: the first :obj:`obj_1` that explore
            obj_2: the second :obj:`obj_2` that explore
        Returns:
            value of distance between the nearest points of the explored objects
        """

        if isinstance(obj_1, Polygon):
            obj_1 = self._poly_to_geom(obj_1)
        elif isinstance(obj_1, Point):
            obj_1 = self._pt_to_geom(obj_1)
        if isinstance(obj_2, Polygon):
            obj_2 = self._poly_to_geom(obj_2)
        elif isinstance(obj_2, Point):
            obj_2 = self._pt_to_geom(obj_2)
        distance = obj_1.distance(obj_2)

        return distance

    def centroid_distance(self, point: 'Point', poly: 'Polygon') -> Point:
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

    struct = Polygon(polygon_id=str(uuid4()),
                     points=[(Point(x, y)) for x, y in zip(a, b)])

    return struct
