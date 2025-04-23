import heapq
from typing import Dict, List, Tuple

import MulticastTree
from routing import Routing


def compute_routes_for_flow(source: int, destinations: List[int], network: Dict[int, List[int]],
                            number_of_candidates: int) \
        -> List[MulticastTree.Tree]:
    # collect different paths
    candidate_trees: List[MulticastTree.Tree] = []

    if len(destinations) == 0:
        return []

    duplicate_counter = 0
    # TODO move magic number elsewhere
    while len(candidate_trees) < number_of_candidates and duplicate_counter < 10:
        current_tree = MulticastTree.Tree()

        # TODO make sure the first path is always different
        initial_path = Routing.get_dijkstra_shortest_path(source,
                                                          destinations[(len(candidate_trees) + duplicate_counter)
                                                                       % len(destinations)],
                                                          network, {})
        current_tree.integrate(initial_path)

        for destination in destinations:
            next_path = compute_best_intermediate_path(current_tree, destination, network)
            current_tree.integrate(next_path)

        if candidate_trees.__contains__(current_tree):
            duplicate_counter += 1
        else:
            candidate_trees.append(current_tree)

    return candidate_trees


def compute_best_intermediate_path(current_tree: MulticastTree.Tree, destination: int, network: Dict[int, List[int]]) \
        -> List[Tuple[int, int]]:
    all_intermediate_paths = [Routing.get_dijkstra_shortest_path(intermediate_node, destination, network, {}) for
                              intermediate_node in current_tree.get_intermediate_nodes()]

    def path_scoring(path: List[Tuple[int, int]]) -> float:
        length = len(path)
        if length == 0:
            return 0
        return length + current_tree.get_hop_distance(path[0][0]) / (2 * current_tree.get_depth())

    # use hop distance as tie-breaker
    return min(all_intermediate_paths, key=lambda path: path_scoring(path))
