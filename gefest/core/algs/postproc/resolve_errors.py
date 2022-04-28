from collections import Counter
from copy import deepcopy
from itertools import permutations

import numpy as np

from gefest.core.algs.geom.validation import out_of_bound, self_intersection, too_close, unclosed_poly
from gefest.core.opt.constraints import check_constraints
from gefest.core.structure.domain import Domain
from gefest.core.structure.point import Point
from gefest.core.structure.polygon import Polygon
from gefest.core.structure.structure import Structure

"""
Defines methods to correct wrong structures (not satisfying the constraints)
Function postprocess makes structures that satisfy the constraints given in validation
"""


def postprocess(structure: Structure, domain: Domain):
    corrected_structure = deepcopy(structure)

    # If you set fixed polygons in the domain, they will be added to the structure
    for fixed_poly in domain.fixed_points:
        corrected_structure.polygons.insert(0, deepcopy(fixed_poly))

    # Fixing proximity between polygons
    while too_close(corrected_structure, domain):
        corrected_structure = _correct_closeness(corrected_structure, domain)

    # Fixing each polygon in structure
    for i, poly in enumerate(corrected_structure.polygons):
        local_structure = Structure([poly])
        if self_intersection(local_structure):
            corrected_structure.polygons[i] = _correct_self_intersection(poly, domain)
        elif out_of_bound(local_structure, domain):
            corrected_structure.polygons[i] = _correct_wrong_point(poly, domain)
        elif unclosed_poly(local_structure, domain) and domain.geometry.is_closed:
            corrected_structure.polygons[i] = _correct_unclosed_poly(poly)

    return corrected_structure


def iterative_postprocess(new_structure, default_structure, domain, max_attempts=3):
    constraints = False
    while not constraints:
        new_structure = postprocess(new_structure, domain)
        constraints = check_constraints(structure=new_structure, domain=domain)
        max_attempts -= 1
        if max_attempts == 0:
            # If the number of attempts is over,
            # the transformation is considered unsuccessful
            # and one of the structures is returned
            return default_structure
    return new_structure


def _correct_low_points(poly: 'Polygon',
                        domain: 'Domain'):
    new_point = Point(np.random.uniform(domain.min_x, domain.max_x),
                      np.random.uniform(domain.min_y, domain.max_y))
    poly.points.append(new_point)

    return poly


def _correct_unclosed_poly(poly: Polygon) -> Polygon:
    # Simple fix for open polygons by adding first point to end
    point_to_add = poly.points[0]
    poly.points.append(point_to_add)
    correct_poly = poly
    return correct_poly


def _correct_wrong_point(poly: Polygon, domain: Domain):
    point_moved = False
    for p_id, point in enumerate(poly.points):
        # Correcting each point out of bounds
        if point in domain.fixed_points:
            continue
        point.x = max(point.x, domain.min_x + domain.len_x * 0.05)
        point.y = max(point.y, domain.min_y + domain.len_y * 0.05)
        point.x = min(point.x, domain.max_x + domain.len_x * 0.05)
        point.y = min(point.y, domain.max_y + domain.len_y * 0.05)
        if not domain.contains(point):
            # if the point is not in the domain, we look for the nearest one inside
            new_point = domain.geometry.nearest_point(point, domain.bound_poly)
            poly.points[p_id] = new_point
            point_moved = True

    if point_moved:
        poly = domain.geometry.resize_poly(poly=poly, x_scale=0.8, y_scale=0.8)

    return poly


def _correct_self_intersection(poly: Polygon, domain: Domain):
    # Change self-intersected poly to convex
    convex_poly = domain.geometry.get_convex(poly)
    return convex_poly


def _correct_closeness(structure: Structure, domain: Domain):
    """
    For polygons that are closer than the specified threshold,
    one of them will be removed
    """
    polygons = structure.polygons
    matching = {}
    for poly_1, poly_2 in permutations(polygons, 2):
        distance = _pairwise_dist(poly_1, poly_2, domain)
        if distance < domain.min_dist and poly_2.id != 'fixed':
            matching[poly_1.id] = poly_2.id

    to_delete = Counter(matching.values()).most_common(1)[0][0]
    corrected_structure = Structure([poly for poly in structure.polygons if poly.id != to_delete])

    return corrected_structure


def _pairwise_dist(poly_1: Polygon, poly_2: Polygon, domain: Domain):
    if len(poly_1.points) == 0 or len(poly_2.points) == 0:
        return 9999

    return domain.geometry.min_distance(poly_1, poly_2)
