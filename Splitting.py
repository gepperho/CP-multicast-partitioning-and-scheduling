import copy
from typing import List, Tuple, Dict

from MyStructs import Stream, Cluster
from routing import Routing


def create_stream_from_group(current_stream: Stream, group: List[int], mapping: Dict[int, List[int]], counter: int):
    # copy the attributes
    new_stream = copy.deepcopy(current_stream)
    new_stream.stream_id = int(1e10 + (current_stream.stream_id * 100) + counter)
    new_stream.number_of_destinations = len(group)

    if new_stream.number_of_destinations > 1:
        multicast_address = max(mapping.keys()) + 1
        mapping[multicast_address] = group
        new_stream.destination = multicast_address
    else:
        new_stream.destination = group[0]

    return new_stream


def get_distance_to_nearest_group(distances, groups, current_node) -> Tuple[int, int]:
    def get_min_distance_of_group(group):
        return min([distances[(elem, current_node)] for elem in group])

    return min([(groups.index(group), get_min_distance_of_group(group)) for group in groups], key=lambda x: x[1])


def append_next_node(distances, groups, current_node, threshold):
    group_index, distance = get_distance_to_nearest_group(distances, groups, current_node)
    if distance <= threshold:
        groups[group_index].append(current_node)
    else:
        # form a new group
        groups.append([current_node])


def build_subgroups(distances, stream, destinations, threshold):
    groups = []

    # create first group
    init_element = max(destinations, key=lambda x: distances[(stream.source, x)] if x != stream.source else 0)
    groups.append([init_element])
    destinations.remove(init_element)

    # append nodes to the groups
    while len(destinations) > 0:
        next_destination = max(destinations, key=lambda x: get_distance_to_nearest_group(distances, groups, x))
        destinations.remove(next_destination)
        append_next_node(distances, groups, next_destination, threshold)

    return groups


def split_single_multicast_in_multiple_streams(stream: Stream, network_graph, threshold: int, mapping) -> List[Stream]:
    if stream.destination not in mapping or len(mapping[stream.destination]) == 1:
        return [stream]

    apsp_destinations = copy.deepcopy(mapping[stream.destination])
    apsp_destinations.append(stream.source)
    distances = Routing.compute_all_pair_shortest_path(network_graph, apsp_destinations)

    groups = build_subgroups(distances, stream, copy.deepcopy(mapping[stream.destination]), threshold)
    if len(groups) == 1:
        return [stream]

    streams = []
    counter = 0
    for group in groups:
        streams.append(create_stream_from_group(stream, group, mapping, counter))
        counter += 1

    return streams


def split_multicasts_in_multiple_streams(cluster: Cluster, network: Dict[int, List[int]], mapping,
                                         threshold: int) -> bool:
    modified_streams = []

    for stream in cluster.streams:
        new_streams = split_single_multicast_in_multiple_streams(stream, network, threshold, mapping)
        modified_streams.extend(new_streams)

    cluster.streams = modified_streams
    return True
