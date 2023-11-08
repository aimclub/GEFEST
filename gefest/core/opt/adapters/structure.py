from golem.core.adapter.adapter import BaseOptimizationAdapter
from golem.core.optimisers.graph import OptGraph, OptNode
from loguru import logger

from gefest.core.geometry import Point, Polygon, Structure
from gefest.core.geometry.domain import Domain


class StructureAdapter(BaseOptimizationAdapter):
    """Adapter for GOLEM."""

    def __init__(self, domain: Domain) -> None:
        super().__init__(base_graph_class=Structure)
        self.domain = domain

    def _point_to_node(self, point) -> OptNode:
        if type(point) is OptNode:
            self._log.warn('Unexpected: OptNode found in adapter instead' 'Point.')
        else:
            content = {
                'name': f'pt_{point.x}_{point.y}',
                'params': {},
            }
            node = OptNode(content=content)
            node.content['params'] = {
                'x': point.x,
                'y': point.y,
            }
            return node

    def _adapt(self, adaptee: Structure) -> OptGraph:
        """Convert Structure into OptGraph."""
        nodes = []
        for polygon in adaptee.polygons:
            if polygon[-1] == polygon[0]:
                polygon = polygon[:-1]

            prev_node = None
            for point_id in range(len(polygon.points)):
                node = self._point_to_node(polygon.points[point_id])
                if prev_node:
                    node.nodes_from = [prev_node]

                prev_node = node
                nodes.append(node)

        graph = OptGraph(nodes)
        return graph

    def _restore(self, opt_graph: OptGraph) -> Structure:
        """Convert OptGraph into Structure."""
        structure = []
        poly = Polygon()
        first_node = opt_graph.nodes[0]
        if not len(first_node.nodes_from):
            poly.points.append(
                Point(
                    first_node.content['params']['x'],
                    first_node.content['params']['y'],
                ),
            )
        else:
            logger.critical('Unexpected nodes order. First node is not root.')

        for node in opt_graph.nodes[1:]:
            if not len(node.nodes_from):
                if self.domain.geometry.is_closed:
                    poly.points.append(poly[0])

                structure.append(poly)
                poly = Polygon()

            poly.points.append(
                Point(
                    node.content['params']['x'],
                    node.content['params']['y'],
                ),
            )

        if poly not in structure:
            if self.domain.geometry.is_closed:
                poly.points.append(poly[0])

            structure.append(poly)

        return Structure(structure)
