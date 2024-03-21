from enum import Enum

from .GA.GA import BaseGA
from .GA.GA_history import BaseGA as BaseGA_h
from .golem_optimizer.standard import StandardOptimizer
from .golem_optimizer.surrogate import SurrogateOptimizer


class OptimizerTypes(Enum):
    """Enumeration of all optimizers."""

    gefest_ga = BaseGA
    gefest_ga_h = BaseGA_h
    golem_optimizer = StandardOptimizer
    golem_surrogate = SurrogateOptimizer
