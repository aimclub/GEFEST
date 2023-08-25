from pydantic import computed_field
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
