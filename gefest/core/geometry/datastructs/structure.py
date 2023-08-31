from uuid import UUID, uuid4
from typing import Union
from pydantic import Field
from pydantic.dataclasses import dataclass

from .polygon import Polygon
from .point import Point


@dataclass
class Structure:
    polygons: list[Polygon] = Field(default_factory=list)
    fitness: list[float] = Field(default_factory=list)
    _id: UUID = Field(default_factory=uuid4)

    def __len__(self):
        return len(self.polygons)

    def __getitem__(self, key):
        return self.polygons[key]

    def __setitem__(self, key, value):
        self.polygons[key] = value

    def __contains__(self, item: Union[Point, Polygon]):
        if isinstance(item, Polygon):
            return item in self.polygons
        if isinstance(item, Point):
            return any(item in poly for poly in self.polygons)
