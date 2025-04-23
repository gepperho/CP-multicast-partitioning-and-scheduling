import copy
from typing import Dict, List, Tuple
import math

import MulticastTree
from routing import Routing


def compute_routes_for_flow(source: int, destinations: List[int], network: Dict[int, List[int]],
                            number_of_candidates: int) \
        -> List[MulticastTree.Tree]:
    # collect different paths
    edge_weights: Dict[Tuple[int, int], int] = {}
    candidate_trees: List[MulticastTree.Tree] = []
    edge_weight_default_weight = 100
    edge_weight_increase = 200
    edge_weight_decrease = -1 * math.inf

    if len(destinations) == 0:
        return []

    duplicate_counter = 0
    # TODO move magic number elsewhere
    while len(candidate_trees) < number_of_candidates and duplicate_counter < 10:

        temp_edge_weights = copy.deepcopy(edge_weights)

        multicast_tree = MulticastTree.Tree()
        ring_shifted_destinations = destinations[(duplicate_counter + len(candidate_trees)) % len(destinations):] + \
                                    destinations[:(duplicate_counter + len(candidate_trees)) % len(destinations)]
        for destination in ring_shifted_destinations:
            shortest_path = Routing.get_dijkstra_shortest_path(source, destination, network, temp_edge_weights)
            Routing.modify_edge_weights(shortest_path, temp_edge_weights, edge_weight_default_weight,
                                        edge_weight_decrease)
            multicast_tree.integrate(shortest_path)

        if candidate_trees.__contains__(multicast_tree):
            duplicate_counter += 1
        else:
            candidate_trees.append(multicast_tree)

            Routing.modify_edge_weights(multicast_tree.get_links(), edge_weights, edge_weight_default_weight,
                                        edge_weight_increase)

    return candidate_trees
