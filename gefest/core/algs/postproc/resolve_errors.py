import numpy as np
from copy import deepcopy

from gefest.core.algs.geom.validation import out_of_bound, self_intersection, too_close, unclosed_poly, intersection
from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Polygon, Structure, Point

"""
Defines methods to correct wrong structures (not satisfying the constraints)
Function postprocess makes structures that satisfy the constraints given in validation
"""


def postprocess(structure: Structure, domain: Domain):
    corrected_structure = deepcopy(structure)

    # Fixing each polygon in structure
    for i, poly in enumerate(corrected_structure.polygons):
        local_structure = Structure([poly])
        if unclosed_poly(local_structure, domain) and domain.is_closed:
            corrected_structure.polygons[i] = _correct_unclosed_poly(poly)
        if self_intersection(local_structure):
            corrected_structure.polygons[i] = _correct_self_intersection(poly, domain)
        if out_of_bound(local_structure, domain):
            corrected_structure.polygons[i] = _correct_wrong_point(poly, domain)

    #  Fixing proximity between polygons
    if too_close(structure, domain):
        corrected_structure = _correct_closeness(corrected_structure, domain)

    # If you set fixed polygons in the domain, here they are added to the structure
    if len(corrected_structure.polygons) > 0:
        for fixed in domain.fixed_points:
            if not (fixed.points == [p.points for p in corrected_structure.polygons]):
                corrected_structure.polygons.append(deepcopy(fixed))

    return corrected_structure


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
    num_poly = len(polygons)
    to_delete = []

    for i in range(num_poly - 1):
        for j in range(i + 1, num_poly):
            distance = _pairwise_dist(polygons[i], polygons[j], domain)
            if distance < domain.min_dist:
                if polygons[i].id != 'fixed':
                    to_delete.append(i)  # Collecting polygon indices for deletion

    to_delete_poly = [structure.polygons[i] for i in np.unique(to_delete)]
    corrected_structure = Structure(polygons=[poly for poly in structure.polygons if poly not in to_delete_poly])
    return corrected_structure


def _pairwise_dist(poly_1: Polygon, poly_2: Polygon, domain: Domain):
    if poly_1 is poly_2 or len(poly_1.points) == 0 or len(poly_2.points) == 0:
        return 9999

    return domain.geometry.min_distance(poly_1, poly_2)
