from uuid import UUID, uuid4

from pydantic import Field
from pydantic.dataclasses import dataclass

from .point import Point


@dataclass
class Polygon:
    points: list[Point] = Field(default_factory=list)
    _id: UUID = Field(default_factory=uuid4)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, key) -> Point:
        return self.points[key]

    def __setitem__(self, key: int, value: Point):
        self.points[key] = value
