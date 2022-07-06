from typing import List

from gefest.core.structure.point import Point


class Polygon:
    """The geometrical object made up of ``Point`` objects

    Args:
        polygon_id (str): name of Polygon, used as separator between polygons
        in the postprocessing layer; must be ``"fixed"`` where created a polygon
        that can not be deleted from working area due postprocessing
        points (list): list of ``Point`` objects which form borders of Polygon

    Returns:
        ``Polygon(List[Point])``
    """

    def __init__(self, polygon_id: str, points: List[Point]):
        self.id = polygon_id
        self.points = points
