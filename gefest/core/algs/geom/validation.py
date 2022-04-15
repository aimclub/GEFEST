from itertools import permutations

from shapely.geometry import Point as GeomPoint, LineString

from gefest.core.structure.domain import Domain
from gefest.core.structure.polygon import Polygon
from gefest.core.geometry.geometry import Geometry
from gefest.core.structure.structure import Structure

min_dist_from_boundary = 0.01

"""
Here are defined general constraints on polygons.
We define intersection between polygons in structure, self-intersection in polygons,
out of bound for points in polygon, closeness between polygons and unclosed (for closed polygons).
"""


def intersection(structure: 'Structure', geometry: 'Geometry'):
    polygons = structure.polygons
    if len(polygons) < 2:
        return False

    for poly_1, poly_2 in permutations(polygons, 2):
        if geometry.intersects_poly(poly_1, poly_2):
            return True
    return False


def out_of_bound(structure: 'Structure', domain: 'Domain'):
    domain_poly = domain.bound_poly
    for poly in structure.polygons:
        if not domain.geometry.is_contain_poly(poly, domain_poly):
            return True
    return False

def too_close(structure: 'Structure', domain: Domain):
    polygons = structure.polygons
    for poly_1, poly_2 in permutations(polygons, 2):
        distance = _pairwise_dist(poly_1, poly_2, domain)
        if distance < domain.min_dist:
            return True
    return False


def _pairwise_dist(poly_1: Polygon, poly_2: Polygon, domain: Domain):
    if len(poly_1.points) == 0 or len(poly_2.points) == 0:
        return 9999

    return domain.geometry.min_distance(poly_1, poly_2)


# The is simple method indicates that the figure is self-intersecting
def self_intersection(structure: 'Structure'):
    intersected = not any([LineString([GeomPoint(pt.x, pt.y) for pt in poly.points]).is_simple
                           for poly in structure.polygons])
    return bool(intersected)


# Checks for equality of the first and last points
def unclosed_poly(structure: 'Structure', domain: 'Domain'):
    if domain.geometry.is_closed:
        return bool(any([poly.points[0] != poly.points[-1] for poly in structure.polygons]))
    return False
