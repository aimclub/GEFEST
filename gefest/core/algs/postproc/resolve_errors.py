from copy import deepcopy

from gefest.core.algs.geom.validation import out_of_bound, self_intersection, too_close
from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Polygon, Structure


def postprocess(structure: Structure, domain: Domain):
    corrected_structure = deepcopy(structure)

    for i, poly in enumerate(corrected_structure.polygons):
        local_structure = Structure([poly])
        if self_intersection(local_structure):
            corrected_structure.polygons[i] = _correct_self_intersection(poly, domain)
        if out_of_bound(local_structure, domain):
            corrected_structure.polygons[i] = _correct_wrong_point(poly, domain)

    if too_close(structure, domain):
        corrected_structure = _correct_closeness(corrected_structure, domain)

    if len(corrected_structure.polygons) > 0:
        for fixed in domain.fixed_points:
            if fixed not in corrected_structure.polygons[0].points:
                corrected_structure.polygons[0].points.append(deepcopy(fixed))
    return corrected_structure


def _correct_wrong_point(poly: Polygon, domain: Domain):
    point_moved = False
    for p_id, point in enumerate(poly.points):
        if point in domain.fixed_points:
            continue
        point.x = max(point.x, domain.min_x + domain.len_x * 0.05)
        point.y = max(point.y, domain.min_y + domain.len_y * 0.05)
        point.x = min(point.x, domain.max_x + domain.len_x * 0.05)
        point.y = min(point.y, domain.max_y + domain.len_y * 0.05)
        if not domain.geometry.is_contain_point(domain.bound_poly, point):
            new_point = domain.geometry.nearest_point(point, domain.bound_poly)
            poly.points[p_id] = new_point
            point_moved = True

    if point_moved:
        poly = domain.geometry.resize_poly(poly=poly, x_scale=0.8, y_scale=0.8)

    return poly


def _correct_self_intersection(poly: Polygon, domain: Domain):
    # change self-intersected poly to convex
    convex_poly = domain.geometry.get_convex(poly)
    return convex_poly


def _correct_closeness(structure: Structure, domain: Domain):
    # shrink all polygons
    for poly_1 in structure.polygons:
        for poly_2 in structure.polygons:
            if poly_1 is not poly_2 and domain.geometry.intersects(poly_1, poly_2):
                structure.polygons.remove(poly_1)
    return structure
