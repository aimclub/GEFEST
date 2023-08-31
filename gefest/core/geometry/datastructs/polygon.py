from uuid import UUID, uuid4
from enum import Enum
from typing import Optional, Union

from pydantic import Field
from pydantic.dataclasses import dataclass

from .point import Point


class PolyID(Enum):
    TEMP = "tmp"
    CONSTR = "constraint"
    FIXED_AREA = "fixed_area"
    FIXED_POLY = "fixed_poly"
    PROH_AREA = "prohibited_area"
    PROH_TARG = "prohibited_target"
    PROH_POLY = "prohibited_poly"


@dataclass
class Polygon:
    points: list[Point] = Field(default_factory=list)
    id_: Optional[Union[UUID, PolyID]] = Field(default_factory=uuid4)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, key) -> Point:
        return self.points[key]

    def __setitem__(self, key: int, value: Point):
        self.points[key] = value

    def __contains__(self, item):
        return item in self.points
