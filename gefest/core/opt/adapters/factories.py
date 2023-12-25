from golem.core.adapter.adapter import BaseOptimizationAdapter
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.random_graph_factory import RandomGraphFactory

from gefest.tools.samplers.sampler import Sampler


class StructureFactory(RandomGraphFactory):
    """GOLEM RandomGraphFactory version of GEFEST sampler."""

    def __init__(
        self,
        sampler: Sampler,
        adapter: BaseOptimizationAdapter,
    ) -> None:
        self.sampler = sampler
        self.adapter = adapter

    def __call__(self, *args, **kwargs) -> OptGraph:
        """Generates ranom GOLEM graph."""
        samples = self.sampler(1)
        return self.adapter.adapt(samples[0])
