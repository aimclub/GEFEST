class StructureAdapter(BaseOptimizationAdapter):
    def __init__(self, log=None):
        """
        Optimization adapter for Pipeline class
        """
        super().__init__(base_graph_class=Structure, base_node_class=Point, log=log)

    def _point_to_node(self, node):
        # Prepare content for nodes
        if type(node) == OptNode:
            self._log.warn('Unexpected: OptNode found in adapter instead'
                           'Point.')
        else:
            content = {'name': f'pt_{node.x}_{node.y}',
                       'params': {
                           x: node.x,
                           y: node.y
                       }}

            node = OptNode(content=content)
            return node

    def adapt(self, adaptee: Structure):
        """ Convert Structure class into OptGraph class """
        nodes = []
        for polygon in adaptee.polygons:
            prev_node = None
            for point_id in range(len(polygon.points)):
                node = _point_to_node(polygon.points[point_id])
                if prev_node:
                    node.nodes_from = [prev_node]
                prev_node = node
                nodes.append(node)

        graph = OptGraph(nodes)
        return graph

    def restore(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> 'Pipeline':
        """ Convert OptGraph class into Structure class """
        structure = Structure()
        poly = None
        for node in opt_graph.nodes:
            if node.nodes_from is None:
                # next polygon started
                structure.polygons.append(poly)
                poly = Polygon()
            poly.points.append(Point(node.params['x'],
                                     node.params['y']))
            structure.polygons.append(prev_poly)
        if poly not in structure.polygons:
            # add last poly
            structure.polygons.append(poly)

        return structure
