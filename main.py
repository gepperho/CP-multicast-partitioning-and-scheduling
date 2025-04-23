import argparse
import time
from typing import Dict, List

import IO
from routing import Routing
import Scheduler
from MyStructs import ResultStruct, Details

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--network", type=str, help="Path to the network graph file", required=True)
parser.add_argument("-s", "--scenario", type=str, help="Path to the flow scenario file", required=True)
parser.add_argument("-t", "--timelimit", type=int,
                    help="solver time limit in seconds. Use negative values for unlimited. Default: 120s", default=120)
parser.add_argument("--threads", type=int, help="Number of threads to be used at most", default=4)
parser.add_argument("-v", "--verbose", help="print a lot of debug outputs", action='store_true')
parser.add_argument("-raw", "--print-raw", help="print results in raw format", action='store_true')
parser.add_argument("-r", "--routing", type=int, help="Routing strategy for the candidate paths: "
                                                      "[dijkstra_overlap: 1, intermediate_node_distance: 2, "
                                                      "shortest_path_based_source_tree: 3]", default=2)
parser.add_argument("-nw", "--no-wait", help="Enables no-wait scheduling", action='store_true')
parser.add_argument("--multicast-splitting-threshold", type=int, default=0,
                    help="Split the multicasts in several smaller multicasts/unicasts. Parameter provides the threshold distance to split. Use 0 for no splits.")
parser.add_argument("-p", "--candidate-paths", type=int, help="Number of candidate paths to be computed.", default=1)
parser.add_argument("-o", "--optimize-traffic",
                    help="Set this flag to optimize the aggregated network traffic instead of the number of streams",
                    action='store_true')
parser.add_argument("--scheduling_output", type=str, help="Path to the scheduling output file", required=False)
parser.add_argument("--cplex-path", type=str, help="Path to the CPLEX solver", required=False)

args = parser.parse_args()

network: Dict[int, List[int]] = IO.load_network(args.network)
scenario = IO.load_scenario(args.scenario)
routing_algorithm = Routing.RoutingAlgorithm(args.routing)

unique_time_stamp = int(time.time() * 1000 * 1000)

navigator = Routing.Router(network=network, routing_algorithm=routing_algorithm,
                           no_candidate_paths=args.candidate_paths)

if args.verbose:
    print('network: ', network)
    print('scenario: ', scenario)

details = Details(
    timeout=args.timelimit if args.timelimit > 0 else None,
    verbose=args.verbose,
    threads=args.threads,
    hyper_cycle=int(scenario.config_info['hyper_cycle']),
    optimize_traffic=args.optimize_traffic,
    split_multicasts=args.multicast_splitting_threshold,
    no_wait=args.no_wait,
    scheduling_output=args.scheduling_output if args.scheduling_output is not None else None,
    cplex_path=args.cplex_path if args.cplex_path is not None else None,
)
scheduler = Scheduler.Scheduler(details, Routing.create_multicast_destination_mapping(scenario))
results: List[ResultStruct] = scheduler.schedule_scenario(scenario=scenario, router=navigator)

for result in results:
    if args.print_raw:
        static_output = '{network:}\t{scenario:}\t{splitting:}\t{routing:}\t{candidate_paths:}\t'.format(
            network=args.network,
            scenario=args.scenario,
            splitting=args.multicast_splitting_threshold,
            routing=routing_algorithm.to_string(),
            candidate_paths=args.candidate_paths)
        static_output += '{builder_dummy:}\t{max_configs_dummy:}\t{expansion_dummy:}\t'.format(
            builder_dummy='CP',
            max_configs_dummy=-1,
            expansion_dummy='CP')
        static_output += '{threads:}\t{unique_time_stamp:}\t'.format(
            threads=args.threads,
            unique_time_stamp=unique_time_stamp)
        static_output += '{strategy:}\t{planning_mode:}'.format(
            strategy='CP-NW' if details.no_wait else 'CP-Q',
            planning_mode='offensive')
        print('{time_step:}\t{static:}\t{dynamic}'.format(
            time_step=result.time_step,
            static=static_output,
            dynamic=result.print_dfsv2_style(details.timeout)))
    else:
        print('===============================')
        result.print()
        # if args.verbose:
        #     print('===============================')
        #     for name, var in result.solution_variables.items():
        #        if str(name).isdigit():
        #            continue
        #        print(var)
