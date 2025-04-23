import json
from typing import Dict, List

from MyStructs import Scenario


def load_network(network_path: str) -> Dict[int, List[int]]:
    network = {}
    with open(network_path, 'r') as network_file:
        for raw_line in network_file.readlines():
            if raw_line.startswith('#') or raw_line.startswith('%'):
                continue

            line = str(raw_line).replace('\n', '').replace('\r', '')
            split = ' '
            if line.__contains__('\t'):
                split = '\t'
            s_v1, s_v2 = line.split(split, maxsplit=1)
            v1: int = int(s_v1)
            v2: int = int(s_v2)

            if not network.__contains__(v1):
                network[v1] = [v2]
            else:
                network[v1].append(v2)

            if not network.__contains__(v2):
                network[v2] = [v1]
            else:
                network[v2].append(v1)
    return network


def load_scenario(scenario_path: str) -> Scenario:
    with open(scenario_path) as scenario_file:
        return Scenario.from_dict(json.load(scenario_file))
