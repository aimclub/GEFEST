from typing import Optional

from shapely.geometry.point import Point

from gefest.core.geometry import Point as G_Point
from gefest.core.geometry import Polygon, PolyID, Structure


def create_prohibited(
    points_radius,
    targets: Optional[list[list]] = None,
    fixed_points: Optional[list[list]] = None,
    fixed_area: Optional[list[list]] = None,
) -> Structure:
    """Creation of fixed, prohibited structures. Polygons cannot cross them.

    Args:
        points_radius (_type_): _description_
        targets (Optional[list[list]], optional): Fixed targets inside domain. Defaults to None.
        fixed_points (Optional[list[list]], optional): Fixed lines inside domain. Defaults to None.
        fixed_area (Optional[list[list]], optional): Fixed areas inside domain. Defaults to None.

    Returns:
        Structure: Compilation of imputs as GEFEST Structure.

    """
    prohibited_area = []
    if targets is not None:
        target_polygons = [
            list(Point(target).buffer(points_radius).exterior.coords) for target in targets
        ]
        target_points = [[G_Point(p[0], p[1]) for p in target] for target in target_polygons]
        poly_targets = [Polygon(id_=PolyID.PROH_TARG, points=points) for points in target_points]
        prohibited_area += poly_targets

    if fixed_points is not None:
        fix_points = [[G_Point(p[0], p[1]) for p in fixed] for fixed in fixed_points]
        poly_fixed = [Polygon(id_=PolyID.PROH_POLY, points=points) for points in fix_points]
        prohibited_area += poly_fixed

    if fixed_area is not None:
        fix_area = [[G_Point(p[0], p[1]) for p in fixed] for fixed in fixed_area]
        poly_area = [Polygon(id_=PolyID.PROH_AREA, points=points) for points in fix_area]
        prohibited_area += poly_area

    struct = Structure(polygons=prohibited_area)

    return struct
