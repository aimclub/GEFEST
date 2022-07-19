"""
Here are defined general constraints on polygons.
We define intersection between polygons in structure, self-intersection in polygons,
out of bound for points in polygon, closeness between polygons and unclosed (for closed polygons).
"""

from itertools import permutations

from shapely.geometry import Point as GeomPoint, LineString

from gefest.core.structure.domain import Domain
from gefest.core.structure.polygon import Polygon
from gefest.core.geometry.geometry import Geometry
from gefest.core.structure.structure import Structure

min_dist_from_boundary = 0.01


def intersection(structure: 'Structure', geometry: 'Geometry') -> bool:
    """The method for checking intersection between Polygons that put to Structure

    Args:
        structure (Structure): the :obj:`Structure` that explore
        geometry (Geometry): way of processing geometrical objects,
           :obj:`class Geometry2D` for processing 2D objects

    Returns:
        bool: ``True`` if at least one of the polygons in given :obj:`structure` intersects another,
        otherwise - ``False``
    """

    polygons = structure.polygons
    if len(polygons) < 2:
        return False

    for poly_1, poly_2 in permutations(polygons, 2):
        if geometry.intersects_poly(poly_1, poly_2):
            return True

    return False


def out_of_bound(structure: 'Structure', domain: 'Domain') -> bool:
    """The method for checking every polygon in the given :obj:`Structure`
    on crossing borders of :obj:`Domain`

    Args:
        structure (Structure): the :obj:`Structure` that explore
        domain (Domain): the :obj:`Domain` that determinates the main
            parameters, this method requires ``allowed_area`` from :obj:`Domain`

    Returns:
        bool: ``True`` if at least one of the polygons in given :obj:`structure` crossing borders
        of allowed area, otherwise - ``False``
    """
    domain_poly = domain.bound_poly
    for poly in structure.polygons:
        if not domain.geometry.is_contain_poly(poly, domain_poly):
            return True

    return False


def too_close(structure: 'Structure', domain: Domain) -> bool:
    """Checking for too close location between every :obj:`Polygon` in the
    given :obj:`Structure`

    Args:
        structure (Structure): the :obj:`Structure` that explore
        domain (Domain): the :obj:`Domain` that determinates the main
            parameters, this method requires ``min_dist`` from :obj:`Domain`

    Returns:
        bool: ``True`` if at least one distance between any polygons is less than value of minimal
        distance set by :obj:`Domain`, otherwise - ``False``
    """

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


def self_intersection(structure: 'Structure') -> bool:
    """The method indicates that any :obj:`Polygon` in the :obj:`Structure`
    is self-intersected

    Args:
        structure (Structure): the :obj:`Structure` that explore

    Returns:
        bool: ``True`` if at least one of the polygons in the :obj:`Structure` is
        self-intersected, otherwise - ``False``
    """

    intersected = not any([LineString([GeomPoint(pt.x, pt.y) for pt in poly.points]).is_simple
                           for poly in structure.polygons])

    return intersected


def unclosed_poly(structure: 'Structure', domain: 'Domain') -> bool:
    """Checking for equality of the first and last points

    Args:
        structure (Structure): the :obj:`Structure` that explore
        domain (Domain): the :obj:`Domain` that determinates the main
            parameters, this method requires ``is_closed`` from :obj:`Domain`

    Returns:
        bool: ``True`` if patameter ``is_closed`` from :obj:`Domain` is ``True`` and at least
        one of the polygons has unequality between the first one and the last point,
        otherwise - ``False``
    """

    if domain.is_closed:
        return any([poly.points[0] != poly.points[-1] for poly in structure.polygons])
    else:
        return False
