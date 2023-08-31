from typing import Iterable, Optional, Union

from pydantic import Field, computed_field, field_validator, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Annotated

from gefest.core import Geometry2D
from gefest.core.geometry import Point, Polygon, Structure


@dataclass
class Domain:
    allowed_area: Union[Polygon, list[list[float]]]
    name: str = "main"
    min_poly_num: int = 2
    max_poly_num: int = 4
    min_points_num: int = 20
    max_points_num: int = 50
    prohibited_area: Optional[Structure] = None
    fixed_points: Optional[Polygon] = Field(default_factory=list)
    geometry: Optional[object] = Geometry2D(is_closed=True)

    def __contains__(self, point: Point):
        """Checking :obj:`Domain` contains :obj:`point`
        Args:
            point: checked point
        Returns:
            ``True`` if given :obj:`Point` locates in the allowed area borders,
           otherwise returns ``False``
        """
        return self.geometry.is_contain_point(self.allowed_area, point)

    @model_validator(mode="after")
    def min_max_bounds(self):
        if self.min_poly_num > self.max_poly_num:
            raise ValueError("Invalid points number interval.")
        if self.min_points_num > self.max_points_num:
            raise ValueError("Invalid points number interval.")

    @field_validator("allowed_area")
    def parse_allowed_area(cls, data: Union[Polygon, list[list[float]]]):
        if data is None or len(data) <= 2:
            raise ValueError("Not enough points for allowed_area.")
        return Polygon([Point(*coords) for coords in data])

    @computed_field
    def min_x(self) -> int:
        return min(p.x for p in self.allowed_area)

    @computed_field
    def max_x(self) -> int:
        return max(p.x for p in self.allowed_area)

    @computed_field
    def min_y(self) -> int:
        return min(p.y for p in self.allowed_area)

    @computed_field
    def max_y(self) -> int:
        return max(p.y for p in self.allowed_area)

    @computed_field
    def len_x(self) -> int:
        return abs(self.max_x - self.min_x)

    @computed_field
    def len_y(self) -> int:
        return abs(self.max_y - self.min_y)

    @computed_field
    def bound_poly(self) -> Polygon:
        return self.allowed_area

        bnd_points = [Point(*pt_coords) for pt_coords in self.allowed_area]
        return Polygon(polygon_id=f"bnd_{self.name}", points=bnd_points)
