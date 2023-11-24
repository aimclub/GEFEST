from enum import Enum
from typing import Optional, Union
from uuid import UUID, uuid4

from pydantic import Field
from pydantic.dataclasses import dataclass

from .point import Point


class PolyID(Enum):
    """Enumeration of special polygons ids."""

    TEMP = 'tmp'
    CONSTR = 'constraint'
    FIXED_AREA = 'fixed_area'
    FIXED_POLY = 'fixed_poly'
    PROH_AREA = 'prohibited_area'
    PROH_TARG = 'prohibited_target'
    PROH_POLY = 'prohibited_poly'


@dataclass
class Polygon:
    """Polygon dataclass."""

    points: list[Point] = Field(default_factory=list)
    id_: Optional[Union[UUID, PolyID]] = Field(default_factory=uuid4)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, key) -> Point:
        if isinstance(key, slice):
            indices = range(*key.indices(len(self.points)))
            return Polygon([self.points[i] for i in indices])

        return self.points[key]

    def __setitem__(self, key: int, value: Point):
        self.points[key] = value

    def __contains__(self, item):
        return item in self.points
