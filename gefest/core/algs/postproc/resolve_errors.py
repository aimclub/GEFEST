from copy import deepcopy

from shapely.geometry.multipolygon import MultiPolygon as ShapelyMultiPolygon
from shapely.geometry.polygon import Polygon as ShapelyPolygon
from shapely.ops import nearest_points

from gefest.core.algs.geom.validation import out_of_bound, self_intersection, too_close
from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Polygon, Structure


def postprocess(structure: Structure, domain: Domain):
    corrected_structure = deepcopy(structure)

    for i, poly in enumerate(corrected_structure.polygons):
        local_structure = Structure([poly])
        if self_intersection(local_structure):
            corrected_structure.polygons[i] = _correct_self_intersection(poly)
        if out_of_bound(local_structure, domain):
            corrected_structure.polygons[i] = _correct_wrong_point(poly, domain)

    if too_close(structure, domain):
        corrected_structure = _correct_closeness(corrected_structure)

    if len(corrected_structure.polygons) > 0:
        for fixed in domain.fixed_points:
            if fixed not in corrected_structure.polygons[0].points:
                corrected_structure.polygons[0].points.append(deepcopy(fixed))
    return corrected_structure


def _correct_wrong_point(poly: Polygon, domain: Domain):
    point_moved = False
    for point in poly.points:
        if point in domain.fixed_points:
            continue
        point.x = max(point.x, domain.min_x + domain.len_x * 0.05)
        point.y = max(point.y, domain.min_y + domain.len_y * 0.05)
        point.x = min(point.x, domain.max_x + domain.len_x * 0.05)
        point.y = min(point.y, domain.max_y + domain.len_y * 0.05)
        if not domain.contains(point):

            point = domain.geometry.nearest_points(point, domain.bound_poly)
            #     nearest_points(point.as_geom(),
            #                    domain.as_geom())
            # point.x = nearest_correct_position.x
            # point.y = nearest_correct_position.y
            point_moved = True

        if point_moved:
            poly = domain.geometry.resize_poly(poly=poly, x_scale=0.8, y_scale=0.8)
    return poly


def _correct_self_intersection(poly: Polygon):
    # change self-intersected poly to convex
    convex = poly.as_geom().buffer(1)
    if isinstance(convex, ShapelyMultiPolygon):
        convex = convex[0]
    if isinstance(convex, ShapelyPolygon) and len(poly.points) > 2:
        poly.points = []
        for convex_pt in [(x, y) for x, y in zip(convex.exterior.coords.xy[0],
                                                 convex.exterior.coords.xy[1])]:
            poly.points.append(PolygonPoint(convex_pt[0], convex_pt[1]))
    return poly


def _correct_closeness(structure: Structure):
    # shrink all polygons
    for poly1 in structure.polygons:
        for poly2 in structure.polygons:
            if poly1 is not poly2 and poly1.as_geom().intersects(poly2.as_geom()):
                structure.polygons.remove(poly1)
    return structure
