from copy import deepcopy
from itertools import combinations

import numpy as np

from gefest.core.geometry import Polygon, Structure
from gefest.core.geometry.domain import Domain


def distance_between_points(structure: 'Structure', domain: 'Domain') -> bool:
    """The method indicates that any :obj:`Point` in each :obj:`Polygon` of :obj:`Structure`
    is placed in correct distance by previous point
    Args:
        structure: the :obj:`Structure` that explore
    Returns:
        ``True`` if any side of poly have incorrect lenght, otherwise - ``False``
    """
    lenght = domain.dist_between_points
    p1 = structure
    check = []
    for i in [[p.coords()[:2] for p in poly.points] for poly in p1.polygons]:
        for ind, pnt in enumerate(i[1:]):
            check.append(np.norm(np.array(pnt) - np.array(i[ind]), ord=1) < lenght)
    if any(check):
        print('Намутировал плохой полигон!, distance_between_points')
    return any(check)

def postprocess(
    structure: Structure,
    rule_fix_pairs: dict,
    domain: Domain,
) -> Structure:
    if structure is None:
        return None
    if any(
        [(not poly or len(poly) == 0 or any([not pt for pt in poly])) for poly in structure],
    ):
        print('Wrong structure - problems with points')
        return None

    corrected_structure = deepcopy(structure)
    for i, poly in enumerate(corrected_structure.polygons):
        local_structure = Structure([poly])
        for rule, fix in rule_fix_pairs.values():
            if not rule(local_structure, domain):
                corrected_structure.polygons[i] = fix(poly, domain)

    cts = [rule(structure, domain) for rule, _ in rule_fix_pairs.values()]
    if not any(cts):
        return corrected_structure
    return None


def correct_unclosed_poly(poly: Polygon, domain: Domain) -> Polygon:
    if domain.geometry.is_closed:
        point_to_add = poly.points[0]
        poly.points.append(point_to_add)
        correct_poly = poly
        return correct_poly
    return poly


def correct_wrong_point(poly: Polygon, domain: Domain) -> Polygon:
    point_moved = False
    for p_id, point in enumerate(poly.points):
        if domain.fixed_points:
            if point in domain.fixed_points:
                continue
        point.x = max(point.x, domain.min_x + domain.len_x * 0.05)
        point.y = max(point.y, domain.min_y + domain.len_y * 0.05)
        point.x = min(point.x, domain.max_x + domain.len_x * 0.05)
        point.y = min(point.y, domain.max_y + domain.len_y * 0.05)
        if point not in domain:
            new_point = domain.geometry.nearest_point(point, domain.bound_poly)
            poly.points[p_id] = new_point
            point_moved = True

    if point_moved:
        poly = domain.geometry.resize_poly(poly=poly, x_scale=0.8, y_scale=0.8)

    return poly


def correct_self_intersection(poly: Polygon, domain: Domain) -> Polygon:
    # Change self-intersected poly to convex
    convex_poly = domain.geometry.get_convex(poly)
    return convex_poly


def too_close(structure: Structure, domain: Domain) -> bool:
    """Checks for too close location between every :obj:`Polygon` in the
        given :obj:`Structure`
    Args:
        structure: the :obj:`Structure` that explore
        domain: the :obj:`Domain` that determinates the main
            parameters, this method requires ``min_dist`` from :obj:`Domain`
    Returns:
        ``True`` if at least one distance between any polygons is less than value of minimal
        distance set by :obj:`Domain`, otherwise - ``False``
    """

    pairs = tuple(combinations(structure.polygons, 2))
    is_too_close = [False] * len(pairs)

    for idx, poly_1, poly_2 in enumerate(pairs):
        is_too_close[idx] = _pairwise_dist(poly_1, poly_2, domain) < domain.min_dist

    return any(is_too_close)


def _pairwise_dist(
    poly_1: Polygon,
    poly_2: Polygon,
    domain: Domain,
) -> float:
    """
    ::TODO:: find the answer for the question: why return 0 gives infinite computation
    """
    if poly_1 is poly_2 or len(poly_1) == 0 or len(poly_2) == 0:
        return float('inf')

    # nearest_pts = domain.geometry.nearest_points(poly_1, poly_2) ??? why return only 1 point
    return domain.geometry.min_distance(poly_1, poly_2)
