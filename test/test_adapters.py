from pathlib import Path

from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams

from gefest.core.configs.utils import load_config
from gefest.core.geometry.datastructs.structure import Structure
from gefest.core.opt.adapters.configuration_mapping import (
    map_into_gpa,
    map_into_graph_generation_params,
    map_into_graph_requirements,
)
from gefest.core.opt.adapters.factories import StructureFactory

filepath = Path(__file__)
test_config = load_config(str(filepath.parent) + '/test_config.py')


def test_mapping_into_graph_requirements():
    """Test OptimizationParams translation into GraphRequirements."""
    graph_requirements = map_into_graph_requirements(test_config)
    assert isinstance(graph_requirements, GraphRequirements)


def test_mapping_into_graph_generation_params():
    """Test OptimizationParams translation into GraphGenerationParams."""
    graph_generation_params = map_into_graph_generation_params(test_config)
    assert isinstance(graph_generation_params, GraphGenerationParams)


def test_mapping_into_gpa():
    """Test OptimizationParams translation into GPAlgorithmParameters."""
    gpa = map_into_gpa(test_config)
    assert isinstance(gpa, GPAlgorithmParameters)


def test_structure_factory():
    """Test sampler for GOLEM."""
    factory = StructureFactory(test_config.sampler, test_config.golem_adapter)
    assert isinstance(factory, StructureFactory)
    random_graph = factory()
    assert isinstance(random_graph, OptGraph)
    assert isinstance(test_config.golem_adapter.restore(random_graph), Structure)
