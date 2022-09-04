from shapely.geometry.point import Point
from typing import Optional

from gefest.core.structure.structure import Structure
from gefest.core.structure.structure import Polygon
from gefest.core.structure.structure import Point as G_Point


def create_prohibited(targets: Optional[list[list]] = None, fixed_points: Optional[list[list]] = None,
                      fixed_area: Optional[list[list]] = None) -> Structure:
    """
    Creation of fixed, prohibited structures. Polygons cannot cross them

    :param targets: (Optional[list[list]]), fixed targets inside domain
    :param fixed_points: (Optional[list[list]]), fixed lines inside domain
    :param fixed_area: (Optional[list[list]]), fixed areas inside domain
    :return: Structure, structure of all prohibited polygons (targets, lines, areas)
    """
    prohibited_area = []
    if targets is not None:
        target_polygons = [list(Point(target).buffer(20).exterior.coords) for target in targets]
        target_points = [[G_Point(p[0], p[1]) for p in target] for target in target_polygons]
        poly_targets = [Polygon(polygon_id='prohibited_target', points=points) for points in target_points]
        prohibited_area += poly_targets

    if fixed_points is not None:
        fix_points = [[G_Point(p[0], p[1]) for p in fixed] for fixed in fixed_points]
        poly_fixed = [Polygon(polygon_id='prohibited_poly', points=points) for points in fix_points]
        prohibited_area += poly_fixed

    if fixed_area is not None:
        fix_area = [[G_Point(p[0], p[1]) for p in fixed] for fixed in fixed_area]
        poly_area = [Polygon(polygon_id='prohibited_area', points=points) for points in fix_area]
        prohibited_area += poly_area

    struct = Structure(prohibited_area)

    return struct
