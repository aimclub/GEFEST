from shapely.geometry import Point as GeomPoint, LineString

from gefest.core.structure.domain import Domain
from gefest.core.structure.polygon import Polygon
from gefest.core.structure.structure import Structure

min_dist_from_boundary = 0.01

"""
Here are defined general constraints on polygons.
We define intersection between polygons in structure, self-intersection in polygons,
out of bound for points in polygon, closeness between polygons and unclosed (for closed polygons).
"""


def intersection(structure: 'Structure',
                 domain: 'Domain'):
    try:
        for poly1 in domain.prohibited_area.polygons:
            for poly2 in structure.polygons:
                if domain.geometry.intersects_poly(poly1, poly2):
                    return True
        return False

    except AttributeError:
        return False


def is_contain(structure: 'Structure',
               domain: 'Domain'):
    is_contains = []

    try:
        for poly_area in domain.prohibited_area.polygons:
            if poly_area.id == 'prohibited_area':
                for poly in structure.polygons:
                    is_contains.append(domain.geometry.contains(poly, poly_area))

        return any(is_contains)

    except AttributeError:
        return False


def out_of_bound(structure: 'Structure', domain):
    domain_poly = domain.bound_poly
    for poly in structure.polygons:
        if domain.geometry.intersects_poly(poly, domain_poly):
            return True

    return False


def too_close(structure: 'Structure', domain: Domain):
    polygons = structure.polygons
    num_poly = len(polygons)

    for i, poly in enumerate(polygons):
        for j in range(i + 1, num_poly):
            distance = _pairwise_dist(poly, polygons[j], domain)
            if distance < domain.min_dist:
                return True

    return False


def _pairwise_dist(poly_1: Polygon, poly_2: Polygon, domain: Domain):
    if poly_1 is poly_2 or len(poly_1.points) == 0 or len(poly_2.points) == 0:
        return 9999

    return domain.geometry.min_distance(poly_1, poly_2)


# The is_simple method indicates that the figure is self-intersecting
def self_intersection(structure: 'Structure'):
    intersected = not any([LineString([GeomPoint(pt.x, pt.y) for pt in poly.points]).is_simple
                           for poly in structure.polygons])
    return intersected


# Checks for equality of the first and last points
def unclosed_poly(structure: 'Structure', domain: 'Domain'):
    if domain.is_closed:
        return int(any([poly.points[0] != poly.points[-1] for poly in structure.polygons]))
    else:
        return False
