from typing import Optional, Union

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from gefest.core.geometry import Point, Polygon, Structure
from gefest.core.geometry.geometry_2d import Geometry, Geometry2D
from gefest.core.utils.functions import parse_structs


class Domain(BaseModel):
    """Domain configuration dataclass."""

    allowed_area: Union[Polygon, list[list[float]]]
    name: str = 'main'
    min_poly_num: int = 2
    max_poly_num: int = 4
    min_points_num: int = 20
    max_points_num: int = 50
    polygon_side: float = 0.0001
    min_dist_from_boundary: float = 0.0001
    prohibited_area: Optional[Union[Structure, str]] = Field(default=Structure())
    fixed_points: Optional[Union[Polygon, list[list[float]]]] = Field(default_factory=list)
    geometry_is_convex: bool = True
    geometry_is_closed: bool = True
    geometry: Optional[Union[Geometry, str]] = '2D'

    def __contains__(self, point: Point):
        """Checking :obj:`Domain` contains :obj:`point`.

        Args:
            point: checked point

        Returns:
            ``True`` if given :obj:`Point` locates in the allowed area borders,
           otherwise returns ``False``
        """
        return self.geometry.is_contain_point(self.allowed_area, point)

    @model_validator(mode='after')
    def _post_init_validation(self):
        if self.min_poly_num > self.max_poly_num:
            raise ValueError('Invalid polygons number interval.')

        if self.min_points_num > self.max_points_num:
            raise ValueError('Invalid points number interval.')

        if self.min_poly_num < 1 or self.max_poly_num < 1:
            raise ValueError('Number of polygons must be positive value.')

        min_points_in_poly = 1 + int(self.geometry_is_closed)
        if self.min_points_num <= min_points_in_poly or self.max_points_num <= min_points_in_poly:
            raise ValueError('Number of points must be >2/>1 for closed/non closed geometies.')

        if self.geometry == '2D':
            self.geometry = Geometry2D(
                is_closed=self.geometry_is_closed,
                is_convex=self.geometry_is_convex,
            )

        return self

    @field_validator('min_poly_num')
    @classmethod
    def validate_min_poly_num(cls, data: int):
        """Validates min number of polygons."""
        if data < 1:
            raise ValueError('Min number of polygons must be positive value.')

        return data

    @field_validator('max_poly_num')
    @classmethod
    def validate_max_poly_num(cls, data: int):
        """Validates max number of polygons."""
        if data < 1:
            raise ValueError('Max number of polygons must be positive value.')

        return data

    @field_validator('min_points_num')
    @classmethod
    def validate_min_points_num(cls, data: int):
        """Validates min number of points."""
        if data < 1:
            raise ValueError('Max number of polygons must be positive value.')

        return data

    @field_validator('fixed_points')
    @classmethod
    def validate_fixed_points(cls, data: Union[Polygon, list[tuple[float, float]]]):
        """Validates max number of points."""
        if isinstance(data, Polygon):
            return data

        return Polygon([Point(*coords) for coords in data])

    @field_validator('prohibited_area')
    @classmethod
    def validate_prohibited_area(cls, data: Optional[Union[Structure, str]]):
        """Validates prohibit area format."""
        if isinstance(data, Structure):
            return data

        if isinstance(data, str):
            structs_from_file = parse_structs(data)
            num_records = len(structs_from_file)
            if num_records != 1:
                raise ValueError(f'{num_records} structures found in {data} file, expected 1.')
            else:
                return structs_from_file[0]

        raise TypeError(f'Invalid argument {data}.')

    @field_validator('allowed_area')
    @classmethod
    def validate_allowed_area(cls, data: Union[Polygon, list[list[float]]]):
        """Validates allowed area area format."""
        if data is None or len(data) <= 2:
            raise ValueError('Not enough points for allowed_area.')

        return Polygon([Point(*coords) for coords in data])

    @computed_field
    def dist_between_polygons(self) -> float:
        """Min distance between polygons instructure."""
        return max(self.max_x - self.min_x, self.max_y - self.min_y) / 35

    @computed_field
    def dist_between_points(self) -> float:
        """Min dstance between neighbours points in polygon."""
        return self.dist_between_polygons * 15 * self.polygon_side

    @computed_field
    def min_x(self) -> int:
        """Min x domain coord."""
        return min(p.x for p in self.allowed_area)

    @computed_field
    def max_x(self) -> int:
        """Max x domain coord."""
        return max(p.x for p in self.allowed_area)

    @computed_field
    def min_y(self) -> int:
        """Min y domain coord."""
        return min(p.y for p in self.allowed_area)

    @computed_field
    def max_y(self) -> int:
        """Max y domain coord."""
        return max(p.y for p in self.allowed_area)

    @computed_field
    def len_x(self) -> int:
        """Len of x domain side."""
        return abs(self.max_x - self.min_x)

    @computed_field
    def len_y(self) -> int:
        """Len of y domain side."""
        return abs(self.max_y - self.min_y)

    @computed_field
    def bound_poly(self) -> Polygon:
        """Allowed area bound. Deprecated."""
        return self.allowed_area
