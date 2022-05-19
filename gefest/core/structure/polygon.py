from typing import List, Optional
from uuid import uuid4

from gefest.core.structure.point import Point


class Polygon:
    """The geometrical object made up of :obj:`Point` objects

    Args:
        polygon_id: name of :obj:`Polygon`, used as separator between polygons
            in the postprocessing layer; must be ``"fixed"`` where created a :obj:`Polygon`
            that can not be deleted from working area due postprocessing
        points: :obj:`list` of :obj:`Point` objects which form borders of :obj:`Polygon`

    Attributes:
        polygon_id: returns the name of :obj:`Polygon`
            that can not be deleted from working area due postprocessing
        points: returns :obj:`list` of :obj:`Point` which form borders of :obj:`Polygon`

    Examples:
        >>> from gefest.core.structure.point import Point
        >>> from gefest.core.structure.polygon import Polygon
        >>> points_triagle = [Point(0,0), Point(3,3), Point(3,0)]
        >>> triangle = Polygon('triangle', points=points_triagle)
        >>> triangle.points
        [Point(_x=0, _y=0, _z=0.0),
        Point(_x=3, _y=3, _z=0.0),
        Point(_x=3, _y=0, _z=0.0)]

    Returns:
        Polygon: ``Polygon(List[Point])``
    """

    def __init__(self, polygon_id: str = str(uuid4()), points: Optional[List[Point]] = None):
        if points is None:
            points = []
        self.id = polygon_id
        self.points = points
