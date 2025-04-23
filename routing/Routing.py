import sys
from enum import Enum
from queue import PriorityQueue
from typing import Dict, List, Set, Tuple

import routing.MulticastDijkstraOverlap as MulticastDijkstraOverlap
import routing.ShortestPathBasedSourceTree as ShortestPathBasedSourceTree
import routing.IntermediateNodeDistance as IntermediateNodeDistance
import MulticastTree
from MyStructs import TimeStep


class RoutingAlgorithm(Enum):
    DIJKSTRA_OVERLAP = 1
    INTERMEDIATE_NODE_DISTANCE = 2
    SHORTEST_PATH_BASED_SOURCE_TREE = 3

    def to_string(self):
        if self == RoutingAlgorithm.DIJKSTRA_OVERLAP:
            return "ModifiedDijkstraOverlap"
        elif self == RoutingAlgorithm.INTERMEDIATE_NODE_DISTANCE:
            return "IntermediateNodeDistance"
        elif self == RoutingAlgorithm.SHORTEST_PATH_BASED_SOURCE_TREE:
            return "ShortestPathBasedSourceTree"
        else:
            return "unknown"


def modify_edge_weights(path: List[Tuple[int, int]],
                        edge_weights: Dict[Tuple[int, int], int],
                        default_weight: int,
                        increase):
    # modify weight
    for link in path:
        weight = default_weight
        if link in edge_weights:
            weight = edge_weights[link]
        weight += increase
        if weight < 0:
            weight = 0
        edge_weights[link] = int(weight)


def get_dijkstra_shortest_path(source: int, destination: int,
                               network: Dict[int, List[int]],
                               increased_edge_weights: Dict[Tuple[int, int], int]) \
        -> List[Tuple[int, int]]:
    link_from_predecessor: Dict[int, Tuple[int, int]] = {}

    # holds distance, node tuples. This ordering, since it is ordered by the first entry first
    frontier_queue = PriorityQueue()

    frontier_distances: Dict[int, int] = {}
    for key in network.keys():
        frontier_distances[key] = sys.maxsize - 1000000

    frontier_distances[source] = 0
    frontier_queue.put((0, source))

    checked_nodes: Set[int] = set()
    # expand nodes
    while not frontier_queue.empty():
        current_distance, current_node = frontier_queue.get()

        if frontier_distances[destination] < current_distance:
            # useless path
            continue

        # this filters already expanded nodes, since we add duplicates in emplace
        if not checked_nodes.__contains__(current_node):

            for next_hop in network[current_node]:
                out_link = (current_node, next_hop)
                # get edge weight
                edge_weight = 1
                if out_link in increased_edge_weights:
                    edge_weight = increased_edge_weights[out_link]

                distance_to_next_hop = frontier_distances[current_node] + edge_weight

                if distance_to_next_hop < frontier_distances[next_hop]:
                    frontier_distances[next_hop] = distance_to_next_hop

                    link_from_predecessor[next_hop] = out_link
                    # update frontier_queue, note that this can add duplicates with lower frontier_distances
                    frontier_queue.put((distance_to_next_hop, next_hop))

            checked_nodes.add(current_node)

    # extract path
    path: List[(int, int)] = []
    current_node = destination

    while current_node is not source:
        current_link = link_from_predecessor[current_node]
        path.append(current_link)
        predecessor = current_link[0]
        current_node = predecessor

    path.reverse()
    return path


def create_multicast_destination_mapping(scenario) -> Dict[int, List[int]]:
    mapping = {}
    if len(scenario.multicast_group_info) == 0:
        return mapping
    for network_element in range(scenario.multicast_group_info["multicast_first_possible_address"]):
        mapping[network_element] = [network_element]
    for multicast_address_element in scenario.multicast_group_info['multicast_addresses']:
        address_key: int = int(multicast_address_element['multicast'])
        addresses = multicast_address_element['dst_addresses']
        mapping[address_key] = addresses

    return mapping


def compute_all_pair_shortest_path(network: Dict[int, List[int]], destinations: List[int]) \
        -> Dict[Tuple[int, int], int]:
    all_pair_shortest_paths: Dict[Tuple[int, int], int] = {}
    for source in destinations:
        for destination in destinations:
            if source < destination:
                distance = len(get_dijkstra_shortest_path(source=source,
                                                          destination=destination,
                                                          network=network,
                                                          increased_edge_weights={}))
                all_pair_shortest_paths[(source, destination)] = distance
                all_pair_shortest_paths[(destination, source)] = distance
    return all_pair_shortest_paths


class Router:
    network: Dict[int, List[int]]
    routing_algorithm: RoutingAlgorithm
    no_candidate_paths: int

    multicast_mapping: Dict[int, List[int]]

    def __init__(self, network: Dict[int, List[int]], routing_algorithm: RoutingAlgorithm, no_candidate_paths: int):
        self.network = network
        self.routing_algorithm = routing_algorithm
        self.no_candidate_paths = no_candidate_paths

    def compute_candidate_routes(self, time_step: TimeStep, multicast_mapping) -> Dict[int, List[MulticastTree.Tree]]:
        """
        calls the candidate route computation for the clusters in the given time step
        :param time_step: the time step to compute the routes for
        :param multicast_mapping: the mapping of multicast addresses to destinations
        :return:
        """
        routes: Dict[int, List[MulticastTree.Tree]] = {}
        temp_routes: List[MulticastTree.Tree]

        for cluster in time_step.addClusters:
            for flow in cluster.streams:

                if self.routing_algorithm == RoutingAlgorithm.DIJKSTRA_OVERLAP:
                    temp_routes = MulticastDijkstraOverlap.compute_routes_for_flow(source=flow.source,
                                                                                   destinations=multicast_mapping[
                                                                                       flow.destination],
                                                                                   network=self.network,
                                                                                   number_of_candidates=self.no_candidate_paths)
                elif self.routing_algorithm == RoutingAlgorithm.INTERMEDIATE_NODE_DISTANCE:
                    temp_routes = IntermediateNodeDistance.compute_routes_for_flow(source=flow.source,
                                                                                   destinations=multicast_mapping[
                                                                                       flow.destination],
                                                                                   network=self.network,
                                                                                   number_of_candidates=self.no_candidate_paths)
                elif self.routing_algorithm == RoutingAlgorithm.SHORTEST_PATH_BASED_SOURCE_TREE:
                    temp_routes = ShortestPathBasedSourceTree.create_source_tree(source=flow.source,
                                                                                 destinations=multicast_mapping[
                                                                                     flow.destination],
                                                                                 network=self.network)
                else:
                    exit("Unknown routing algorithm")
                routes[int(flow.stream_id)] = temp_routes

        return routes
