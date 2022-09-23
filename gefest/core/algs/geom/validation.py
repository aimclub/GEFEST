"""
Here are defined general constraints on polygons by validation rules.
Validation is a checking on valid and unvalid objects for further processing.
"""

from shapely.geometry import Point as GeomPoint, Polygon as GeomPolygon
from shapely.validation import explain_validity
from itertools import permutations

from gefest.core.structure.domain import Domain
from gefest.core.structure.polygon import Polygon

MIN_DIST = 15

min_dist_from_boundary = 1


def intersection(structure: 'Structure', domain: 'Domain') -> bool:
    """The method for checking intersection between Polygons in the Structure
    Args:
        structure: the :obj:`Structure` that explore
        domain: the :obj:`class Domain`
    Returns:
        ``True`` if at least one of the polygons in given :obj:`structure` intersects another,
        otherwise - ``False``
    """

    polygons = structure.polygons
    if len(polygons) < 2:
        return False

    for poly_1, poly_2 in permutations(polygons, 2):
        if domain.geometry.intersects_poly(poly_1, poly_2):
            return True

    return False


def out_of_bound(structure: 'Structure', domain=None) -> bool:
    """The method for checking every polygon in the given :obj:`Structure`
        on crossing borders of :obj:`Domain`

        Args:
            structure: the :obj:`Structure` that explore
            domain: the :obj:`Domain` that determinates the main
                parameters, this method requires ``allowed_area`` from :obj:`Domain`
        Returns:
        ``True`` if at least one of the polygons in given :obj:`structure` crossing borders
        of allowed area, otherwise - ``False``
    """
    geom_poly_allowed = GeomPolygon([GeomPoint(pt[0], pt[1]) for pt in domain.allowed_area])

    for poly in structure.polygons:
        for pt in poly.points:
            geom_pt = GeomPoint(pt.x, pt.y)
            if not geom_poly_allowed.contains(geom_pt) and not \
                    geom_poly_allowed.distance(geom_pt) < min_dist_from_boundary:
                return True

    return False


def too_close(structure: 'Structure', domain: Domain) -> bool:
    """Checking for too close location between every :obj:`Polygon` in the
    given :obj:`Structure`
    Args:
        structure: the :obj:`Structure` that explore
        domain: the :obj:`Domain` that determinates the main
            parameters, this method requires ``min_dist`` from :obj:`Domain`
    Returns:
        ``True`` if at least one distance between any polygons is less than value of minimal
        distance set by :obj:`Domain`, otherwise - ``False``
    """
    is_too_close = any(
        [any([_pairwise_dist(poly_1, poly_2, domain) < domain.min_dist for
              poly_2 in structure.polygons]) for poly_1
         in structure.polygons])
    return is_too_close


def _pairwise_dist(poly_1: Polygon, poly_2: Polygon, domain: Domain):
    """
    ::TODO:: find the answer for the question: why return 0 gives infinite computation
    """
    if poly_1 is poly_2 or len(poly_1.points) == 0 or len(poly_2.points) == 0:
        return 9999

    # nearest_pts = domain.geometry.nearest_points(poly_1, poly_2) ??? why return only 1 point
    return domain.geometry.min_distance(poly_1, poly_2)


def self_intersection(structure: 'Structure'):
    """The method indicates that any :obj:`Polygon` in the :obj:`Structure`
    is self-intersected
    Args:
        structure: the :obj:`Structure` that explore
    Returns:
        ``True`` if at least one of the polygons in the :obj:`Structure` is
        self-intersected, otherwise - ``False``
    """
    return any([len(poly.points) > 2 and
                _forbidden_validity(explain_validity(GeomPolygon([GeomPoint(pt.x, pt.y) for pt in poly.points])))
                for poly in structure.polygons])


def unclosed_poly(structure: 'Structure', domain: 'Domain') -> bool:
    """Checking for equality of the first and the last points
    Args:
        structure: the :obj:`Structure` that explore
        domain: the :obj:`Domain` that determinates the main
            parameters, this method requires ``is_closed`` from :obj:`Domain`
    Returns:
        ``True`` if patameter ``is_closed`` from :obj:`Domain` is ``True`` and at least
        one of the polygons has unequality between the first one and the last point,
        otherwise - ``False``
    """

    if domain.is_closed:
        return any([poly.points[0] != poly.points[-1] for poly in structure.polygons])
    else:
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


def _forbidden_validity(validity):
    return validity != 'Valid Geometry' and 'Ring Self-intersection' not in validity
