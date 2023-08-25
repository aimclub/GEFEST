from uuid import UUID, uuid4

from pydantic import Field
from pydantic.dataclasses import dataclass

from gefest.core.geometry import Structure


@dataclass
class Individual:
    genotype: Structure
    fitness: list[float] = Field(default_factory=list)
    _id: UUID = Field(default_factory=uuid4)
