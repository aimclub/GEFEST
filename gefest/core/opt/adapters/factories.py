import random
from math import cos, pi, sin

from golem.core.optimisers.graph import OptNode
from golem.core.optimisers.opt_node_factory import OptNodeFactory

from gefest.core.geometry.domain import Domain


class SimpleGefestOptNodeFactory(OptNodeFactory):
    def __init__(
        self,
        domain: Domain,
    ):
        self.domain = Domain

        # self._generation_radius = 1
        # if domain_allowed_area is not None:
        #     dx = max(domain_allowed_area, key = lambda point: point[0])[0] \
        #          - min(domain_allowed_area, key = lambda point: point[0])[0]
        #     dy = max(domain_allowed_area, key = lambda point: point[1])[1] \
        #          - min(domain_allowed_area, key = lambda point: point[1])[1]
        #     self._generation_radius = min(dx, dy) * 0.05

    def exchange_node(self, node: OptNode, **kwargs) -> OptNode:
        return self.get_node(node)

    def get_parent_node(self, node: OptNode, **kwargs) -> OptNode:
        return self.get_node(node=node)

    def get_node(self, **kwargs) -> OptNode:
        if 'node' in kwargs.keys():
            theta = random() * 2 * pi
            r = random()
            px = (r * cos(theta) * self._generation_radius) + kwargs['node'].content['params']['x']
            py = (r * sin(theta) * self._generation_radius) + kwargs['node'].content['params']['y']
        else:
            px = int(random() * self._generation_radius)
            py = int(random() * self._generation_radius)

        return OptNode(
            content={
                'name': f'pt_{px}_{py}',
                'params': {'x': px, 'y': py},
            },
        )
