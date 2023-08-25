from pydantic import Field, computed_field
from pydantic.dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float
    # z: Optional[float] = 0

    @computed_field
    def coords(self) -> list[float]:
        return [self.x, self.y]

    # list(map(
    #             partial(getattr,self),
    #             self.__dataclass_fields__.keys()
    # ))


@dataclass
class Polygon:
    points: list[Point] = Field(default_factory=list)
    # _id: uuid.UUID = Field(default_factory=uuid.uuid4)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value


@dataclass
class Structure:
    polygons: list[Polygon] = Field(default_factory=list)

    def __len__(self):
        return len(self.polygons)

    def __getitem__(self, key):
        return self.polygons[key]

    def __setitem__(self, key, value):
        self.polygons[key] = value
