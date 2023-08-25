from uuid import UUID, uuid4

from pydantic import Field
from pydantic.dataclasses import dataclass

from .polygon import Polygon

# @dataclass
# class Individual:
#     genotype: Structure
#     fitness: list[float] = Field(default_factory=list)
#     _id: UUID = Field(default_factory=uuid4)


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
