from typing import Union
from uuid import uuid4

import numpy as np
from golem.utilities.data_structures import ensure_wrapped_in_sequence
from loguru import logger
from shapely import affinity, get_parts
from shapely.affinity import scale
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
)
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box, mapping
from shapely.ops import nearest_points, split

from gefest.core.geometry import Point, Polygon, Structure

from .geometry import Geometry


class Geometry2D(Geometry):
    """Overriding the geometry base class for 2D structures.

    The input receives information about the closeness of the polygon.

    Args:
        is_closed: ``True`` if the :obj:`Polygon` must have close borders
            (first Point is equal to the last one), otherwise ``False``.
            Default value is ``True``
    """

    is_closed: bool = True
    is_convex: bool = True

    def get_length(self, polygon: Polygon):
        """Returns polygon perimeter."""
        if len(polygon.points) <= 2:
            return 0

        geom_polygon = LineString([ShapelyPoint(pt.x, pt.y) for pt in polygon])

        return geom_polygon.length

    def shapely_to_gefest(self, geom_in):
        """Converts any shapely object to GEFEST polygon."""
        if isinstance(geom_in, ShapelyPolygon):
            return Polygon(self.get_coords(geom_in))
        # add other shapely objects

    def get_coords(self, poly: Union[ShapelyPolygon, LineString]) -> list[Point]:
        """The function for getting points.

        Args:
            poly: :obj:`Polygon` for processing

        Returns:
            all :obj:`Point` that :obj:`poly`contains
        """
        if isinstance(poly, ShapelyPolygon):
            poly = poly.exterior

        points = [
            Point(x, y)
            for x, y in zip(
                list(poly.coords.xy[0]),
                list(poly.coords.xy[1]),
            )
        ]

        return points

    def get_prohibited_geom(
        self,
        prohibited_area: Structure,
        buffer_size: float = 0.001,
    ) -> GeometryCollection:
        """Generates Shapely GeometryCollection from pohibited structure."""
        geom_collection = []
        for poly in prohibited_area.polygons:
            if poly[0] == poly[-1]:
                geom_collection.append(self._poly_to_shapely_poly(poly).buffer(buffer_size))
            else:
                geom_collection.append(self._poly_to_shapely_line(poly).buffer(buffer_size))

        return GeometryCollection(geom_collection)

    def resize_poly(
        self,
        poly: Polygon,
        x_scale: float,
        y_scale: float,
    ) -> Polygon:
        """The function for rescaling polygons along each axis.

        Scaling occurs relative to the center of mass of the polygon.

        Args:
            poly: :obj:`Polygon` for processing
            x_scale: scale value for **x** axis
            y_scale: scale value for **y** axis

        Returns:
            scaled :obj:`poly` by ``(x,y)`` axes
        """
        geom_polygon = self._poly_to_shapely_line(poly)

        rescaled_geom_polygon = affinity.scale(
            geom_polygon,
            x_scale,
            y_scale,
        )

        rescaled_points = self.get_coords(rescaled_geom_polygon)

        rescaled_poly = Polygon(
            polygon_id=poly.id_,
            points=rescaled_points,
        )

        return rescaled_poly

    @logger.catch
    def get_angle(
        self,
        vector1: tuple[Point, Point],
        vector2: tuple[Point, Point],
    ) -> float:
        """Finds angle betwen two bectors."""
        v1 = np.array([vector1[1].x - vector1[0].x, vector1[1].y - vector1[0].y])
        v2 = np.array([vector2[1].x - vector2[0].x, vector2[1].y - vector2[0].y])
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

    def rotate_point(
        self,
        point: Point,
        origin: Point,
        angle: float,
    ) -> Polygon:
        """Rotates polygon by given angle."""
        rotated = affinity.rotate(
            ShapelyPoint(point.x, point.y),
            angle,
            ShapelyPoint(origin.x, origin.y),
        )
        return Point(rotated.x, rotated.y)

    def rotate_poly(
        self,
        poly: Polygon,
        angle: float,
    ) -> Polygon:
        """Rotating polygon relative to the center of mass by a given angle.

        Args:
            poly: :obj:`Polygon` for processing.
            angle: value of degree rotation.

        Returns:
            rotated :obj:`poly`.
        """
        geom_polygon = self._poly_to_shapely_line(poly)

        rotated_geom_polygon = affinity.rotate(
            geom_polygon,
            angle,
            'center',
        )

        rotated_points = self.get_coords(rotated_geom_polygon)
        rotated_poly = Polygon(
            polygon_i=poly.id_,
            points=rotated_points,
        )

        return rotated_poly

    def get_square(self, polygon: Polygon) -> float:
        """Recieving value of the area.

        Args:
            polygon: :obj:`Polygon` for processing.

        Returns:
            value of the :obj:`polygon` area.
        """
        if len(polygon.points) <= 2:
            return 0

        geom_polygon = ShapelyPolygon([self._pt_to_shapely_pt(pt) for pt in polygon])

        return geom_polygon.area

    def is_contain_point(self, poly: Polygon, point: Point) -> bool:
        """Checking if a point is inside a polygon.

        Args:
            poly: :obj:`Polygon` that explore
            point: :obj:`Point` for checking presence inside the :obj:`Polygon`.

        Returns:
            ``True`` if :obj:`point` is into :obj:`poly`, otherwise ``False``.
        """
        geom_poly_allowed = ShapelyPolygon([self._pt_to_shapely_pt(pt) for pt in poly])
        geom_pt = ShapelyPoint(point.x, point.y)

        return geom_poly_allowed.contains(geom_pt)

    def nearest_point(self, point: Point, poly: Polygon) -> Point:
        """Calculating closest point between input point and polygon.

        Args:
            point: the :obj:`Point` that explore
            poly: the :obj:`Polygon` that explore

        Returns:
            nearest_correct_position :obj:`Point` from ``point`` among all points in the ``poly``
        """
        geom_poly = self._poly_to_shapely_line(poly)
        geom_point = ShapelyPoint(point.x, point.y)
        _, nearest_correct_position = nearest_points(geom_point, geom_poly)
        return Point(nearest_correct_position.x, nearest_correct_position.y)

    def nearest_points(self, poly_1: Polygon, poly_2: Polygon) -> list[Point]:
        """Calculating closest point between two polygons.

        Args:
            poly_1: the first :obj:`Polygon` that explore
            poly_2: the second :obj:`Polygon` that explore

        Returns:
            the couple of :obj:`Point` where the first one from :obj:`poly_1`
            and the second one from :obj:`poly_2`
        """
        geom_poly_1 = self._poly_to_shapely_line(poly_1)
        geom_poly_2 = self._poly_to_shapely_line(poly_2)

        _, nearest_correct_position = nearest_points(
            geom_poly_1,
            geom_poly_2,
        )
        return nearest_correct_position

    def get_convex(self, poly: Polygon) -> Polygon:
        """Obtaining a convex polygon to avoid intersections.

        Args:
            poly: :obj:`Polygon` for processing

        Returns:
            convex :obj:`Polygon`
        """
        if len(poly.points) < 3:
            return poly

        geom_poly = self._poly_to_shapely_line(poly).convex_hull
        points = self.get_coords(geom_poly)
        polygon = Polygon(polygon_id='tmp', points=points)

        return polygon

    def intersection_line_line(self, points1, points2, scale1, scale2):
        """Returns point of two lines intersection."""
        a = scale(LineString([(p.x, p.y) for p in points1]), scale1, scale1)
        b = scale(LineString([(p.x, p.y) for p in points2]), scale2, scale2)
        intersection_point = a.intersection(b)
        if not intersection_point.is_empty:
            if isinstance(intersection_point, LineString):
                intersection_point = intersection_point.coords[0]
                intersection_point = Point(intersection_point[0], intersection_point[1])
            else:
                intersection_point = Point(intersection_point.x, intersection_point.y)
        else:
            intersection_point = None

        return intersection_point

    def intersection_poly_line(self, figure: Polygon, points: list[Point], scale_factor):
        """Returns points where line intersects polygon."""
        if self.is_closed:
            figure = self._poly_to_shapely_poly(figure)
        else:
            figure = self._poly_to_shapely_line(figure)

        minx, miny, maxx, maxy = figure.bounds
        line = LineString([(p.x, p.y) for p in points])
        line = scale(line, scale_factor)
        bounding_box = box(minx * 2, miny * 2, maxx * 2, maxy * 2)
        a = ShapelyPoint(line.boundary.bounds[:2])
        b = ShapelyPoint(line.boundary.bounds[2:])
        if a.x == b.x:  # vertical line
            extended_line = LineString([(a.x, miny), (a.x, maxy)])
        elif a.y == b.y:  # horizonthal line
            extended_line = LineString([(minx, a.y), (maxx, a.y)])
        else:
            # linear equation: y = k*x + m
            k = (b.y - a.y) / (b.x - a.x)
            m = a.y - k * a.x
            y0 = k * minx + m
            y1 = k * maxx + m
            x0 = (miny - m) / k
            x1 = (maxy - m) / k
            points_on_boundary_lines = [
                ShapelyPoint(minx, y0),
                ShapelyPoint(maxx, y1),
                ShapelyPoint(x0, miny),
                ShapelyPoint(x1, maxy),
            ]
            points_sorted_by_distance = sorted(points_on_boundary_lines, key=bounding_box.distance)
            extended_line = LineString(points_sorted_by_distance[:2])

        if extended_line.intersects(figure):
            interaction = extended_line.intersection(figure)
            if isinstance(interaction, ShapelyPoint):
                interaction = [Point(interaction.x, interaction.y)]
            elif isinstance(interaction, MultiPoint):
                interaction = [Point(p[0], p[1]) for p in interaction.geoms]
            else:
                interaction = [Point(p[0], p[1]) for p in interaction.coords]
        else:
            return None

    def simplify(self, poly: Polygon, tolerance: float) -> Polygon:
        """Simplifies polyon."""
        inp = poly
        if len(poly) < 3:
            return poly

        if self._poly_to_shapely_poly(poly).is_simple:

            poly = self._poly_to_shapely_poly(inp)
            compressed = poly.buffer(-tolerance, join_style='mitre')
            if not compressed.is_empty:
                poly = compressed.buffer(tolerance * 1.05, join_style='mitre')

            simplified = poly.simplify(tolerance)

            if isinstance(simplified, MultiPolygon):
                simplified = max(simplified.geoms, key=lambda p: p.area)

            if simplified.is_empty:
                poly = self._poly_to_shapely_poly(inp)
                compressed = poly.buffer(-tolerance, join_style='mitre')
                decompressed = compressed.buffer(tolerance * 1.1, join_style='mitre')
                intersected = decompressed.intersection(poly)
                simplified = intersected.simplify(tolerance)
                if isinstance(simplified, MultiPolygon):
                    simplified = max(simplified.geoms, lambda p: p.area)

            out = Polygon([Point(p[0], p[1]) for p in simplified.exterior.coords])
        else:
            simplified = self._poly_to_shapely_line(poly).convex_hull.simplify(tolerance)
            if simplified.is_empty:
                raise ValueError('Empty polygon produced')

            out = Polygon([Point(p[0], p[1]) for p in simplified.exterior.coords])

        return out

    def is_simple(self, poly: Polygon) -> bool:
        """Checks if poly is simple."""
        return self._poly_to_shapely_poly(poly).is_simple

    def get_random_point_in_shapey_geom(self, fig):
        """Returns random point from polygon of arbitrary shape shapely geometry."""
        if fig.is_empty:
            raise ValueError('Unable to pick a point from an empty polygon.')

        if isinstance(fig, MultiPolygon):
            bds = []
            for bound in list(fig.geoms):
                bds.extend(list(bound.exterior.coords))

            minx, miny, maxx, maxy = LineString(bds).bounds

        else:
            minx, miny, maxx, maxy = fig.bounds

        x = np.random.uniform(minx, maxx, 1)
        y = np.random.uniform(miny, maxy, 1)
        while not fig.contains(ShapelyPoint(x, y)):
            x = np.random.uniform(minx, maxx, 1)
            y = np.random.uniform(miny, maxy, 1)

        return Point(x, y)

    def get_random_point_in_poly(self, poly) -> Union[Point, None]:
        """Returns random point from polygon of arbitrary shape."""
        minx, miny, maxx, maxy = poly.bounds
        if any(b != b for b in poly.bounds):
            raise ValueError('Unable to pick a point from empty an polygon.')

        #  also can be used polar cords generator within circumscribed circle
        x = np.random.uniform(minx, maxx, 1)
        y = np.random.uniform(miny, maxy, 1)
        point = None
        for _ in range(100):
            x = np.random.uniform(minx, maxx, 1)
            y = np.random.uniform(miny, maxy, 1)
            if poly.contains(ShapelyPoint(x, y)):
                point = Point(x, y)
                break

        return point

    def get_centroid(self, poly: Polygon) -> Point:
        """Getting a point that is the center of mass of the polygon.

        Args:
            poly: the :obj:`Polygon` that explore

        Returns:
            central :obj:`Point` of :obj:`poly`
        """
        points = poly.points
        if len(points) < 3:
            points.append(points[0])

        geom_poly = ShapelyPolygon([self._pt_to_shapely_pt(pt) for pt in points])
        geom_point = geom_poly.centroid
        point = Point(geom_point.x, geom_point.y)
        return point

    def intersects(self, structure: Structure) -> bool:
        """Function to check for any intersection in structure of polygons.

        Whole structure appears like shapely MultiLineString for which uses method is simple.

        Args:
            structure: the :obj:`Structure` that explore

        Returns:
            ``True`` if any :obj:`Polygon` in :obj:`structure` intersects with another one,
               otherwise - ``False``
        """
        multi_geom = MultiLineString(
            [self._poly_to_shapely_line(poly) for poly in structure.polygons],
        )
        return multi_geom.is_simple

    def contains(self, poly1: Polygon, poly2: Polygon) -> bool:
        """Checks if poly2 contains poly1."""
        geom_polygon1 = self._poly_to_shapely_line(poly1)
        geom_polygon2 = ShapelyPolygon([self._pt_to_shapely_pt(pt) for pt in poly2])

        is_contain = geom_polygon2.contains(geom_polygon1)
        return is_contain

    def difference_polys(self, base_poly: Polygon, diff_polys: list[Polygon]):
        """Returns area of base_poly difference with diff_polys polygons."""
        base_poly = self._poly_to_shapely_poly(base_poly)
        diff_polys = ensure_wrapped_in_sequence(diff_polys)
        diff_polys = [self._poly_to_shapely_poly for poly in diff_polys]

        for poly in diff_polys:
            base_poly = base_poly.difference(poly)

        if base_poly.is_empty:
            assert 'Empty difference'

        return Polygon([Point(p[0], p[1]) for p in base_poly.exterior.coords])

    def intersection_polys(self, base_poly: Polygon, diff_polys: list[Polygon]):
        """Returns area of base_poly intersection with diff_polys polygons."""
        base_poly = self._poly_to_shapely_poly(base_poly).convex_hull
        if isinstance(diff_polys, Polygon):
            diff_polys = [diff_polys]

        diff_polys = [self._poly_to_shapely_poly(poly).convex_hull for poly in diff_polys]

        for poly in diff_polys:
            base_poly = base_poly.intersection(poly)

        if base_poly.is_empty:
            assert 'Empty intersection'

        return Polygon([Point(p[0], p[1]) for p in base_poly.exterior.coords])

    def intersects_poly(self, poly_1: Polygon, poly_2: Polygon) -> bool:
        """Intersection between two polygons.

        Args:
            poly_1: the first :obj:`Polygon` that explore
            poly_2: the second :obj:`Polygon` that explore
        Returns:
            ``True`` if the :obj:`poly_1` intersects with :obj:`poly_2`,
            otherwise - ``False``
        """
        geom_poly_1 = self._poly_to_shapely_line(poly_1)
        geom_poly_2 = self._poly_to_shapely_line(poly_2)
        return geom_poly_1.intersects(geom_poly_2)

    def _poly_to_shapely_line(self, poly: Polygon) -> LineString:
        """Transform GEFEST Polygon to shapely non cycled  LineString.

        Args:
            poly: Polygon
        Returns:
            LineString
        """
        return LineString([self._pt_to_shapely_pt(pt) for pt in poly.points])

    def _poly_to_shapely_poly(self, poly: Polygon) -> ShapelyPolygon:
        """Transform GEFEST Polygon to shapely Polygon.

        Args:
            poly: Polygon
        Returns:
            ShapelyPolygon
        """
        return ShapelyPolygon([(pt.x, pt.y) for pt in poly.points])

    def _pt_to_shapely_pt(self, pt: Point) -> ShapelyPoint:
        """Transform GEFEST Polygon to shapely Polygon.

        Args:
            poly: Point

        Returns:
            ShapelyPoint
        """
        return ShapelyPoint(pt.x, pt.y)

    def split_polygon(self, poly, line: tuple[Point, Point], scale_factor=1000) -> list:
        """Splits polygon by line.

        Returns:
            list: Produced parts.
        """
        poly = ShapelyPolygon([(p.x, p.y) for p in poly])
        line = LineString(
            [
                (line[0].x, line[0].y),
                (line[1].x, line[1].y),
            ],
        )
        line = scale(
            line,
            scale_factor,
            scale_factor,
        )
        parts = get_parts(split(poly, line)).tolist()
        parts = [mapping(p)['coordinates'][0][:-1] for p in parts]
        return parts

    def min_distance(self, obj_1, obj_2) -> float:
        """Finds smallest distance between two objects.

        Args:
            obj_1: the first :obj:`obj_1` that explore
            obj_2: the second :obj:`obj_2` that explore

        Returns:
            value of distance between the nearest points of the explored objects
        """
        if isinstance(obj_1, Polygon):
            obj_1 = self._poly_to_shapely_line(obj_1)
        elif isinstance(obj_1, Point):
            obj_1 = self._pt_to_shapely_pt(obj_1)

        if isinstance(obj_2, Polygon):
            obj_2 = self._poly_to_shapely_line(obj_2)
        elif isinstance(obj_2, Point):
            obj_2 = self._pt_to_shapely_pt(obj_2)

        distance = obj_1.distance(obj_2)

        return distance

    def centroid_distance(self, point: Point, poly: Polygon) -> Point:
        """Finds distance from point to polygon."""
        geom_point = self._pt_to_shapely_pt(point)
        geom_poly = self._poly_to_shapely_line(poly)
        dist = geom_point.distance(geom_poly)

        return dist


def create_circle(struct: Structure) -> Structure:
    """Creates circle."""
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

    struct = Polygon(
        polygon_id=str(uuid4()),
        points=[(Point(x, y)) for x, y in zip(a, b)],
    )

    return struct
