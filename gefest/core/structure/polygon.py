from dataclasses import field
from typing import List
from uuid import uuid4

from gefest.core.structure.point import Point


class Polygon:
    def __init__(self, polygon_id: str = None, points: List[Point] = None):
        self.id = polygon_id if polygon_id else uuid4()
        self.points = points if points else []
