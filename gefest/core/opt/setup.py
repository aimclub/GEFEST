from collections import Callable
from dataclasses import dataclass
from typing import List, Union

from gefest.core.structure.domain import Domain


@dataclass
class Setup:
    domain: Union[Domain, List[Domain]]
    simulator: Callable = None