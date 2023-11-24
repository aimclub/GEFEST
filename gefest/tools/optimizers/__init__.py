from enum import Enum

from .GA.GA import BaseGA
from .golem_optimizer.standard import StandardOptimizer
from .golem_optimizer.surrogate import SurrogateOptimizer


class OptimizerTypes(Enum):
    """Enumeration of all optimizers."""

    gefest_ga = BaseGA
    golem_optimizer = StandardOptimizer
    golem_surrogate = SurrogateOptimizer
