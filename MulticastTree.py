from typing import List, Tuple, Dict
from itertools import chain


class Tree:
    links: List[List[Tuple[int, int]]]
    hop_distances: Dict[int, int]

    def __init__(self):
        super().__init__()
        self.links = []
        self.hop_distances = {}

    def get_first_link(self) -> Tuple[int, int]:
        return self.links[0][0]

    def get_links(self) -> List[Tuple[int, int]]:
        return list(chain.from_iterable(self.links))

    def integrate(self, shortest_path: List[Tuple[int, int]]):
        # iterate in reverse
        for link in reversed(shortest_path):
            current_source = link[0]
            if current_source in self.hop_distances:
                # found first link
                index = shortest_path.index(link)
                self.append_sub_path(shortest_path[index:], self.hop_distances[current_source])
                return

        # add whole path
        self.append_sub_path(shortest_path, 0)

    def append_sub_path(self, sub_path: List[Tuple[int, int]], first_index: int):

        current_index = first_index
        for link in sub_path:
            if len(self.links) <= current_index:
                self.links.append([])
            self.links[current_index].append(link)
            current_index += 1
            # add destination to the hop_distances
            self.hop_distances[link[1]] = current_index

    def get_depth(self) -> int:
        return max(self.hop_distances.values())

    def get_links_by_hop(self) -> List[List[Tuple[int, int]]]:
        all_links = []
        for hop in range(1, self.get_depth() + 1):
            all_links.append(
                [temp_link for temp_link in self.get_links() if self.hop_distances[temp_link[1]] == hop]
            )
        return all_links

    def get_intermediate_nodes(self) -> List[int]:
        return list(self.hop_distances.keys())

    def get_hop_distance(self, node: int) -> int:
        return self.hop_distances[node]
