import json
from dataclasses import dataclass
from random import randint
from typing import Optional

import numpy as np

from gefest.core.geometry import Point, Polygon, Structure
from gefest.core.geometry.geometry_2d import Geometry2D

from .domain import Domain


def get_random_structure(domain: Domain) -> Structure:
    # Creating structure with random number of polygons

    structure = Structure(polygons=[])

    num_pols = randint(domain.min_poly_num, domain.max_poly_num)

    for _ in range(num_pols):
        polygon = get_random_poly(parent_structure=structure, domain=domain)
        if polygon is not None and len(polygon.points) > 1:
            structure.polygons.append(polygon)
        else:
            continue

    for poly in structure:
        print(poly[0], poly[-1])

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


def create_poly(centroid: Point, sigma: int, domain: Domain, geometry: Geometry2D) -> Polygon:
    # Creating polygon in the neighborhood of the centroid
    # sigma defines neighborhood

    num_points = randint(
        domain.min_points_num, domain.max_points_num,
    )  # Number of points in a polygon
    points = []
    for _ in range(num_points):
        point = create_polygon_point(centroid, sigma)  # point in polygon
        while not in_bound(point, domain):  # checking if a point is in domain
            point = create_polygon_point(centroid, sigma)
        points.append(point)
    if domain.geometry.is_closed:
        points.append(points[0])

    poly = geometry.get_convex(Polygon(points=points))  # avoid self intersection in polygon

    return poly


def create_area(domain: Domain, structure: Structure, geometry: Geometry2D) -> (Point, float):
    n_poly = len(structure.polygons)  # Number of already existing polygons
    area_size = np.random.randint(low=3, high=15)  # Neighborhood compression ratio
    sigma = (
        max(domain.max_x - domain.min_x, domain.max_y - domain.min_y) / area_size
    )  # Neighborhood size
    if n_poly == 0:
        # In the absence of polygons, the centroid can be located anywhere
        centroid = create_random_point(domain)
    else:
        """
        This procedure allows to find a centroid in the neighborhood
        of which there are no other polygons.
        The minimum distance must be less than 2.5 * sigma.
        """
        centroid = create_random_point(domain)
        min_dist = distance(
            centroid, structure, geometry,
        )  # Distance to the nearest polygon in the structure
        max_attempts = 20
        while min_dist < 2.5 * sigma:
            area_size = np.random.randint(low=3, high=15)
            sigma = max(domain.max_x - domain.min_x, domain.max_y - domain.min_y) / area_size
            centroid = create_random_point(domain)
            min_dist = distance(centroid, structure, geometry)
            if max_attempts == 0:
                return None
            max_attempts -= 1

    return centroid, sigma


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
        np.random.normal(centroid.x, sigma, 1)[0], np.random.normal(centroid.y, sigma, 1)[0],
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
