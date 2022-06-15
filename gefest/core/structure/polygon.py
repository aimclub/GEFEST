from typing import List
from uuid import uuid4

from gefest.core.structure.point import Point


class Polygon:
    def __init__(self, polygon_id: str = str(uuid4()), points=None):
        if points is None:
            points = []
        self.id = polygon_id
        self.points = points
