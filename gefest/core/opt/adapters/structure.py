from typing import Any, Dict, Optional

from golem.core.adapter.adapter import BaseOptimizationAdapter
from golem.core.optimisers.graph import OptGraph, OptNode

from gefest.core.geometry import Point, Polygon, Structure


class StructureAdapter(BaseOptimizationAdapter):
    def __init__(self):
        """
        Optimization adapter for Pipeline class
        """
        super().__init__(base_graph_class=Structure)

    def _point_to_node(self, point):
        # Prepare content for nodes
        if isinstance(point, OptNode):
            self._log.warn('Unexpected: OptNode found in adapter instead' 'Point.')
        else:
            content = {'name': f'pt_{point.x}_{point.y}', 'params': {}}
            node = OptNode(content=content)
            node.content['params'] = {'x': point.x, 'y': point.y}
            return node

    def _adapt(self, adaptee: Structure):
        """Convert Structure class into OptGraph class"""
        nodes = []
        for polygon in adaptee.polygons:
            prev_node = None
            if polygon.points[0] == polygon.points[-1]:
                polygon.points = polygon.points[:-1]
            for point_id in range(len(polygon.points)):
                node = self._point_to_node(polygon.points[point_id])
                if prev_node:
                    node.nodes_from = [prev_node]
                prev_node = node
                nodes.append(node)

        graph = OptGraph(nodes)
        return graph

    def _restore(
        self,
        opt_graph: OptGraph,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Structure:
        """Convert OptGraph class into Structure class"""
        structure = []
        poly = Polygon()
        for node in opt_graph.nodes:
            if node.nodes_from is None:
                # next polygon started
                structure.append(poly)
                poly = Polygon()
            poly.points.append(
                Point(
                    node.content['params']['x'],
                    node.content['params']['y'],
                ),
            )
        poly.points.append(poly[0])
        if poly not in structure:
            # add last poly
            structure.append(poly)

        return Structure(structure)
