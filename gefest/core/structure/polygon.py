from typing import List

from gefest.core.structure.point import Point


class Polygon:
    def __init__(self, polygon_id: str, points: List[Point]):
        self.id = polygon_id
        self.points = points
