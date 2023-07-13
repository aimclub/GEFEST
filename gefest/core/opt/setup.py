from collections.abc import Callable
from dataclasses import dataclass
from typing import List, Union

from gefest.core.structure.domain import Domain


@dataclass
class Setup:
    """The wrapper object that union :obj:`Domain` and a simulator of physical process

    Args:
        domain (Domain): the :obj:`Domain` that use for optimization process
        simulator (Callable): the simulator of physical process

    Returns:
        Setup: prepared settings for optimization
    """

    domain: Union[Domain, List[Domain]]
    simulator: Callable = None
