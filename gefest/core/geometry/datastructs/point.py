from pydantic import computed_field
from pydantic.dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float

    @computed_field
    def coords(self) -> list[float]:
        return [self.x, self.y]
