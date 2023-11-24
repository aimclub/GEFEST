from pydantic import computed_field
from pydantic.dataclasses import dataclass


@dataclass
class Point:
    """2D point dataclass."""

    x: float
    y: float

    @computed_field(repr=False)
    def coords(self) -> list[float]:
        """List coordinates representation."""
        return [self.x, self.y]
