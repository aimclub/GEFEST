import json
import copy
import random
from dataclasses import dataclass
from random import randint
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from gefest.core.structure.domain import Domain
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon


@dataclass
class Structure:
    """The geometrical object made up of :obj:`Polygon` objects

    Args:
        polygons (list): list of :obj:`Polygon` objects which form a combined set of polygons,
            needed for joint processing capability of polygons

    Attributes:
        polygons (list): returns the :obj:`list` of :obj:`Polygon` objects


    Returns:
        Structure: ``Structure(List[Polygon])``

    """
    polygons: List[Polygon]

    def __str__(self):
        out_str = ''
        for i, pol in enumerate(self.polygons):
            out_str += f'\r\n Polygon {i}, size {len(pol.points)}: \r\n'
            for j, pt in enumerate(pol.points):
                out_str += f'Point {j}: x={round(pt.x, 2)}, y={round(pt.y, 2)}; '
        return out_str

    def __repr__(self):
        return json.dumps(self, default=vars)

    @property
    def text_id(self) -> str:
        '''returns information about ``Polygons`` and ``Points``
        included in ``Structure``
        '''

        out_str = ''
        for i, pol in enumerate(self.polygons):
            out_str += f'P{i}={len(pol.points)}:'
            for j, pt in enumerate(pol.points):
                out_str += f'(x={round(pt.x, 2)}, y={round(pt.y, 2)}); '
        return out_str

    @property
    def total_points(self) -> list:
        '''returns list with lengths (number of ``Point``)
        of every ``Polygon`` included
        '''

        return [len(p.points) for p in self.polygons]

    def plot(self, structure, domain=None, title=None):
        '''returns visualization with drawn ``Strucrure``
        '''

        x = [point._x for point in structure.polygons[0].points]
        y = [point._y for point in structure.polygons[0].points]
        plt.plot(x, y)
        plt.title(title)
        plt.show()


def get_random_structure(domain: 'Domain') -> Structure:
    # Creating structure with random number of polygons

    structure = Structure(polygons=[])

    num_pols = randint(domain.min_poly_num, domain.max_poly_num)

    for _ in range(num_pols):
        polygon = get_random_poly(parent_structure=structure,
                                  domain=domain)
        if polygon is not None and len(polygon.points) > 1:
            structure.polygons.append(polygon)
        else:
            continue

    return structure


def get_random_poly(parent_structure: Optional[Structure],
                    domain: Domain) -> Optional[Polygon]:
    geometry = domain.geometry
    try:
        """
        Function for create random polygon.
        The main idea is to create centroids along with a neighborhood to locate the polygon.
        Neighborhood sizes range from small to large.
        The main condition for creating a neighborhood is the absence of other polygons in it.
        After setting the neighborhood, polygons are created around the centroid inside the given neighborhood.
        This approach is less demanding on postprocessing than random creation
        """

        # Centroid with it neighborhood called occupied area
        occupied_area = create_area(domain,
                                    parent_structure,
                                    geometry)
        if occupied_area is None:
            # If it was not possible to find the occupied area then returns None
            return None
        else:
            centroid, sigma = occupied_area  # Size of neighborhood
            # The polygon is created relative to the centroid
            # and the size of the neighborhood
            polygon = create_poly(centroid,
                                  sigma,
                                  domain,
                                  geometry)
    except Exception as ex:
        print(ex)
        import traceback
        print(traceback.format_exc())
        return None

    return polygon


def get_random_point(polygon: 'Polygon',
                     structure: 'Structure',
                     domain: 'Domain'):
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


def create_poly(centroid: 'Point',
                sigma: int,
                domain: 'Domain',
                geometry: 'Geometry'):
    # Creating polygon in the neighborhood of the centroid
    # sigma defines neighborhood

    num_points = randint(domain.min_points_num, domain.max_points_num)  # Number of points in a polygon
    points = []
    for _ in range(num_points):
        point = create_polygon_point(centroid, sigma)  # point in polygon
        while not in_bound(point, domain):  # checking if a point is in domain
            point = create_polygon_point(centroid, sigma)
        points.append(point)
    if domain.is_closed:
        points.append(points[0])

    poly = geometry.get_convex(Polygon('tmp', points=points))  # avoid self intersection in polygon

    return poly


def create_area(domain: 'Domain',
                structure: 'Structure',
                geometry: 'Geometry'):
    n_poly = len(structure.polygons)  # Number of already existing polygons
    area_size = np.random.randint(low=3, high=15)  # Neighborhood compression ratio
    sigma = max(domain.max_x - domain.min_x, domain.max_y - domain.min_y) / area_size  # Neighborhood size
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
        min_dist = distance(centroid, structure, geometry)  # Distance to the nearest polygon in the structure
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


def create_random_point(domain: 'Domain'):
    point = Point(np.random.uniform(low=domain.min_x, high=domain.max_x),
                  np.random.uniform(low=domain.min_y, high=domain.max_y))

    return point


def create_polygon_point(centroid: 'Point',
                         sigma: int):
    # Creating polygon point inside the neighborhood defined by the centroid
    point = Point(np.random.normal(centroid.x, sigma, 1)[0],
                  np.random.normal(centroid.y, sigma, 1)[0])

    return point


def in_bound(point: 'Point',
             domain: 'Domain'):
    if point.x < domain.min_x or point.x > domain.max_x:
        return False
    if point.y < domain.min_y or point.y > domain.max_y:
        return False
    return True


def distance(point: 'Point',
             structure: 'Structure',
             geometry: 'Geometry'):
    polygons = structure.polygons
    distances = []
    for poly in polygons:
        d = geometry.centroid_distance(point, poly)
        distances.append(d)

    return min(distances)


def shuffle_structures(structure_1: Structure, structure_2: Structure):
    """Shuffling polygons between structures in random way.
    Every Structure has one polygon at least"""

    s1 = copy.deepcopy(structure_1)
    s2 = copy.deepcopy(structure_2)

    all_polygons = s1.polygons
    all_polygons.extend(s2.polygons)

    choosen_1 = random.choice(all_polygons)
    s1.polygons = [choosen_1]
    all_polygons.remove(choosen_1)

    choosen_2 = random.choice(all_polygons)
    s2.polygons = [choosen_2]
    all_polygons.remove(choosen_2)

    #  Distribution Polygons between Structures if their number > 2
    max_iter = 50
    while all([all_polygons, max_iter > 0]):
        choosen_structure = random.choice([s1, s2])
        choosen_poly = random.choice(all_polygons)

        temp_poly_list = choosen_structure.polygons
        temp_poly_list.extend([choosen_poly])

        choosen_structure.polygons = temp_poly_list
        all_polygons.remove(choosen_poly)
        max_iter -= 1

    return s1, s2
