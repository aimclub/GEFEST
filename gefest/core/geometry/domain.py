from typing import Optional, Union

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from gefest.core.geometry import Point, Polygon, Structure
from gefest.core.geometry.geometry_2d import Geometry, Geometry2D


class Domain(BaseModel):
    allowed_area: Union[Polygon, list[list[float]]]
    name: str = 'main'
    min_poly_num: int = 2
    max_poly_num: int = 4
    min_points_num: int = 20
    max_points_num: int = 50
    polygon_side: float = 0.0001
    min_dist_from_boundary: float = 0.0001
    prohibited_area: Optional[Structure] = Field(default=Structure([]))
    fixed_points: Optional[Union[Polygon, list[list[float]]]] = Field(default_factory=list)
    geometry_is_convex: bool = True
    geometry_is_closed: bool = True
    geometry: Optional[Union[Geometry, str]] = '2D'

    def __contains__(self, point: Point):
        """Checking :obj:`Domain` contains :obj:`point`
        Args:
            point: checked point
        Returns:
            ``True`` if given :obj:`Point` locates in the allowed area borders,
           otherwise returns ``False``
        """
        return self.geometry.is_contain_point(self.allowed_area, point)

    @model_validator(mode='after')
    def create_classes_instances(self):
        if self.min_poly_num > self.max_poly_num:
            raise ValueError('Invalid points number interval.')
        if self.min_points_num > self.max_points_num:
            raise ValueError('Invalid points number interval.')
        if self.geometry == '2D':
            self.geometry = Geometry2D(
                is_closed=self.geometry_is_closed,
                is_convex=self.geometry_is_convex,
            )
        return self

    @field_validator('fixed_points')
    def validate_allowed_area(cls, data: Union[Polygon, list[list[float]]]):
        if isinstance(data, Polygon):
            return data
        return Polygon([Point(*coords) for coords in data])

    @field_validator('allowed_area')
    def validate_allowed_area(cls, data: Union[Polygon, list[list[float]]]):
        if data is None or len(data) <= 2:
            raise ValueError('Not enough points for allowed_area.')
        return Polygon([Point(*coords) for coords in data])

    @computed_field
    def dist_between_polygons(self) -> float:
        return max(self.max_x - self.min_x, self.max_y - self.min_y) / 35

    @computed_field
    def dist_between_points(self) -> float:
        return self.dist_between_polygons * 15 * self.polygon_side

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
