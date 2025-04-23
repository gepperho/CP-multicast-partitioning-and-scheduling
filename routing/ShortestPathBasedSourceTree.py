from typing import Dict, List

import MulticastTree
from routing import Routing


def create_source_tree(source: int, destinations: List[int], network: Dict[int, List[int]]) \
        -> List[MulticastTree.Tree]:
    multicast_tree = MulticastTree.Tree()
    for destination in destinations:
        shortest_path = Routing.get_dijkstra_shortest_path(source, destination, network, {})
        multicast_tree.integrate(shortest_path)

    return [multicast_tree]
