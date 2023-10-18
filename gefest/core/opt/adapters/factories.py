from golem.core.adapter.adapter import BaseOptimizationAdapter
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.random_graph_factory import RandomGraphFactory

from gefest.tools.samplers.sampler import Sampler


class StructureFactory(RandomGraphFactory):
    """Simple GEFEST sampler wrap for GOLEM RandomGraphFactory compatibility."""

    def __init__(
        self,
        sampler: Sampler,
        adapter: BaseOptimizationAdapter,
    ) -> None:
        self.sampler = sampler
        self.adapter = adapter

    def __call__(self, *args, **kwargs) -> OptGraph:
        samples = self.sampler(1)
        return self.adapter(samples[0])
