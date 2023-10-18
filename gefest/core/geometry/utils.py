import json
from dataclasses import dataclass
from functools import partial
from random import randint
from typing import Optional

import numpy as np
from loguru import logger
from shapely.affinity import scale

from gefest.core.geometry import Point, Polygon, Structure
from gefest.core.geometry.geometry_2d import Geometry2D

from .domain import Domain


def random_polar(origin: Point, radius_scale: float) -> Point:
    theta = np.random.random() * 2 * np.pi
    r = np.random.random() * radius_scale
    return Point((r * np.cos(theta)) + origin.x, (r * np.sin(theta)) + origin.y)


def get_random_structure(domain: Domain, **kwargs) -> Structure:
    # Creating structure with random number of polygons

    structure = Structure(polygons=())
    num_pols = randint(domain.min_poly_num, domain.max_poly_num)

    for _ in range(num_pols):
        polygon = get_random_poly(parent_structure=structure, domain=domain)
        if polygon is not None and len(polygon.points) > 1:
            structure.append(polygon)
        else:
            continue
    return structure


def get_random_poly(parent_structure: Optional[Structure], domain: Domain) -> Optional[Polygon]:
    geometry = domain.geometry
    try:
        """
        Function for creation random polygon.
        The main idea is to create centroids along with a neighborhood to locate the polygon.
        Neighborhood sizes range from small to large.
        The main condition for creating a neighborhood is the absence of other polygons in it.
        After setting the neighborhood, polygons are created around the centroid inside the given neighborhood.
        This approach is less demanding on postprocessing than random creation
        """

        # Centroid with it neighborhood called occupied area
        occupied_area = create_area(domain, parent_structure, geometry)
        if occupied_area is None:
            # If it was not possible to find the occupied area then returns None
            return None
        else:
            centroid = occupied_area[0]
            sigma = occupied_area[1]  # Size of neighborhood
            # The polygon is created relative to the centroid
            # and the size of the neighborhood
            polygon = create_poly(centroid, sigma, domain, geometry)
    except Exception as ex:
        print(ex)
        import traceback

        print(traceback.format_exc())
        return None

    return polygon


def get_random_point(polygon: Polygon, structure: Structure, domain: Domain) -> Optional[Point]:
    # Creating a point to fill the polygon

    centroid = domain.geometry.get_centroid(polygon)
    sigma = distance(centroid, structure, domain.geometry) / 3
    point = create_polygon_point(centroid, sigma)
    max_attempts = 20  # Number of attempts to create in bound point
    while not in_bound(point, domain):
        point = create_polygon_point(centroid, sigma)
        max_attempts -= 1
        if max_attempts == 0:
            return None
    return point


from polygenerator import (
    random_convex_polygon,
    random_polygon,
    random_star_shaped_polygon,
)


def create_poly(centroid: Point, sigma: int, domain: Domain, geometry: Geometry2D) -> Polygon:

    num_points = randint(
        domain.min_points_num,
        domain.max_points_num,
    )
    if domain.geometry.is_convex:  # convex closed/unclosed
        generator = random_convex_polygon

    else:  # non_convex, unclosed
        generator = np.random.choice(
            [
                random_convex_polygon,
                random_star_shaped_polygon,
                random_polygon,
            ],
        )
    new_poly = generator(num_points)
    if not geometry.is_closed:
        start = np.random.choice(range(num_points), 1)[0]
        # slice_length = np.random.choice(range(3, len(poly_points)), 1)[0]
        # logger.info(slice_length)
        new_poly = [new_poly[((start + i) % len(new_poly))] for i in range(num_points)]

    scale_factor = 2 * (sigma / (1 ** 0.5))
    c, s = centroid, scale_factor
    new_poly = Polygon(
        [
            Point(
                ((p[0] - 0.5) * s) + c.x,
                ((p[1] - 0.5) * s) + c.y,
            )
            for p in new_poly
        ],
    )
    if domain.geometry.is_closed:
        new_poly.points.append(new_poly[0])
    return new_poly


# def create_poly(centroid: Point, sigma: int, domain: Domain, geometry: Geometry2D) -> Polygon:
#     # Creating polygon in the neighborhood of the centroid
#     # sigma defines neighborhood
#     num_points = randint(
#         domain.min_points_num,
#         domain.max_points_num,
#     )  # Number of points in a polygon
#     points = []
#     cntr = 0
#     while len(points) < num_points:
#         cntr += 1

#         point = create_polygon_point(centroid, sigma)
#         while not in_bound(point, domain):
#             point = create_polygon_point(centroid, sigma)
#         points.append(point)
#         ind = len(points) - 1
#         if ind > 0:
#             if (
#                 np.linalg.norm(
#                     np.array(points[ind].coords[:2]) - np.array(points[ind - 1].coords[:2]), ord=1
#                 )
#                 < domain.dist_between_points
#             ):
#                 del points[ind]
#         if len(points) == num_points:
#             if (
#                 np.linalg.norm(
#                     np.array(points[-1].coords[:2]) - np.array(points[0].coords[:2]), ord=1
#                 )
#                 < domain.dist_between_points
#             ):
#                 del points[-1]
#         if len(points) == num_points:
#             if domain.geometry.is_closed:
#                 points.append(points[0])
#             poly = geometry.get_convex(Polygon(points=points))
#             points = poly.points
#             if cntr > 5000 and len(points) > 4:
#                 break

#     # logger.info(f'Create poly finish, {cntr} iterations.')
#     return poly


def get_sigma_max(poly, init_max):
    sigma_max = init_max
    left_bound = 0
    right_bound = init_max
    for _ in range(100):
        if poly.buffer(-sigma_max, 1).is_empty:
            right_bound = sigma_max
            sigma_max -= (right_bound - left_bound) / 2
        else:
            left_bound = sigma_max
            sigma_max += (right_bound - left_bound) / 2
        if sigma_max < 0.02:
            break
    return sigma_max


def create_area(domain: Domain, structure: Structure, geometry: Geometry2D) -> (Point, float):
    geom = domain.geometry
    area = geom._poly_to_shapely_poly(domain.bound_poly).buffer(-(domain.min_dist_from_boundary), 1)
    prohibs = geom.get_prohibited_geom(
        domain.prohibited_area,
        buffer_size=domain.dist_between_polygons,
    )
    for g in prohibs.geoms:
        area = area.difference(
            g.buffer(
                domain.min_dist_from_boundary,
            ),
        )
    for poly in structure.polygons:
        area = area.difference(
            geom._poly_to_shapely_poly(poly).convex_hull.buffer(domain.dist_between_polygons, 1),
        ).intersection(area)

    sigma_max = 0.95 * get_sigma_max(area, (min(domain.max_x, domain.max_y) / 2) * 1.01)
    sigma_min = max(domain.max_x - domain.min_x, domain.max_y - domain.min_y) * 0.05

    sigma = np.random.uniform(sigma_min, sigma_max)
    centroid = geom.get_random_point_in_shapey_geom(area.buffer(-sigma, 1))

    return centroid, sigma * 0.99


def create_random_point(domain: Domain) -> Point:
    point = Point(
        np.random.uniform(low=domain.min_x, high=domain.max_x),
        np.random.uniform(low=domain.min_y, high=domain.max_y),
    )
    while not in_bound(point, domain):
        point = Point(
            np.random.uniform(low=domain.min_x, high=domain.max_x),
            np.random.uniform(low=domain.min_y, high=domain.max_y),
        )

    return point


def create_polygon_point(centroid: Point, sigma: int) -> Point:
    # Creating polygon point inside the neighborhood defined by the centroid
    point = Point(
        np.random.normal(centroid.x, sigma, 1)[0],
        np.random.normal(centroid.y, sigma, 1)[0],
    )

    return point


def in_bound(point: Point, domain: Domain) -> bool:
    return domain.geometry.is_contain_point(domain.allowed_area, point)


def distance(point: Point, structure: Structure, geometry: Geometry2D) -> float:
    polygons = structure.polygons
    distances = []
    for poly in polygons:
        d = geometry.centroid_distance(point, poly)
        distances.append(d)

    return min(distances)


from shapely.affinity import scale
from shapely.geometry import LineString, MultiLineString
from shapely.geometry import Point as GeomPoint


def get_selfintersection_safe_point(
    poly: Polygon,
    domain: Domain,
    point_left_idx: int,
    point_right_idx: int,
) -> Polygon:
    'for non convex geometry'
    geom = domain.geometry
    if len(poly) <= 5:
        logger.info('special_case')
    if geom.is_closed:
        parts = [
            *poly.points[(point_right_idx) % len(poly) : :],
            *poly.points[0 : (point_left_idx + 1) % len(poly)],
        ]
        border = [geom._poly_to_shapely_line(Polygon(parts))]
    else:
        border = [
            geom._poly_to_shapely_line(Polygon(part))
            for part in [
                poly.points[(point_right_idx) % len(poly) :],
                poly.points[0 : (point_left_idx + 1) % len(poly)],
            ]
            if len(part) != 1
        ]
    # return border
    border = MultiLineString(
        [scale(line, xfact=0.99, yfact=0.99) for line in border if not line.is_empty],
    )
    l = poly[point_left_idx % len(poly)]
    r = poly[point_right_idx % len(poly)]

    origin = Point((l.x + r.x) / 2, (l.y + l.y) / 2)
    scalefactor = (
        (LineString(((l.x, l.y), (r.x, r.y))).length / 2)
        if len(poly) > 2
        else geom.get_length(poly) * 1.5
    )  # 1.5 ??
    points = []
    p_area = geom.get_prohibited_geom(
        domain.prohibited_area,
        buffer_size=domain.dist_between_polygons,
    )
    for _ in range(200):
        point = random_polar(origin, scalefactor)
        points.append(point)
        new_segment = scale(LineString(((l.x, l.y), (point.x, point.y), (r.x, r.y))), 0.99, 0.99)

        if all(
            (
                not new_segment.intersects(border),
                not new_segment.intersects(geom._poly_to_shapely_line(domain.allowed_area)),
                not any([g.intersects(new_segment) for g in p_area.geoms]),
            ),
        ):
            break
    else:
        point = None

    return point, border
