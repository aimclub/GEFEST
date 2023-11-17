from typing import Union
from uuid import UUID, uuid4

from pydantic import Field
from pydantic.dataclasses import dataclass

from .point import Point
from .polygon import Polygon


@dataclass
class Structure:
    """Structure dataclass."""

    polygons: tuple[Polygon, ...] = Field(default_factory=tuple)
    fitness: list[float] = Field(default_factory=list)
    extra_characteristics: dict = Field(default_factory=dict)
    id_: UUID = Field(default_factory=uuid4)

    def __len__(self):
        return len(self.polygons)

    def __setattr__(self, name, value):
        if name in ['polygons']:
            if self.polygons != tuple(value):
                self.fitness = []

        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        if not isinstance(value, Polygon):
            raise ValueError()

        polygons = list(self.polygons)
        if polygons[key] != value:
            polygons[key] = value
            self.polygons = tuple(polygons)

    def __getitem__(self, key):
        return self.polygons[key]

    def __contains__(self, item: Union[Point, Polygon]):
        if isinstance(item, Polygon):
            return item in self.polygons

        if isinstance(item, Point):
            return any(item in poly for poly in self.polygons)

    def append(self, value):
        """Adds polygon to structure."""
        polygons = list(self.polygons)
        polygons.append(value)
        self.polygons = tuple(polygons)

    def remove(self, value):
        """Removes polygon from structure."""
        polygons = list(self.polygons)
        polygons.remove(value)
        self.polygons = tuple(polygons)
