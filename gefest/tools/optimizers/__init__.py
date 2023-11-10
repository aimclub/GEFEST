from enum import Enum

from .GA.GA import BaseGA
from .golem_optimizer.standard import StandardOptimizer
from .golem_optimizer.surrogate import SurrogateOptimizer
from .SPEA2.SPEA2 import SPEA2


class OptimizerTypes(Enum):
    """Enumeration of all optimizers."""

    gefest_ga = BaseGA
    gefest_spea2 = SPEA2
    golem_optimizer = StandardOptimizer
    golem_surrogate = SurrogateOptimizer
