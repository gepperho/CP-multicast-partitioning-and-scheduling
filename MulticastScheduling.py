from docplex.cp.model import *
import time

import Globals
from MyStructs import Stream, ResultStruct, Cluster, TimeStep, Details
from typing import Dict, List, Tuple, Any
from parse import *
import MulticastTree


class MulticastScheduling:

    def __init__(self, details: Details):
        super().__init__()
        self.details: Details = details

        self.streams: Dict[int, Stream] = {}
        self.clusters: Dict[int, Cluster] = {}

        self.var_admitted_clusters: Dict[int, Any] = {}
        self.var_admitted_streams: Dict[int, Any] = {}
        # stream -> [multicast trees]
        self.var_tree_usage: Dict[int, List[Any]] = {}
        # link -> stream -> [transmissions]
        self.var_link_usage: Dict[Tuple[int, int], Dict[int, List[Any]]] = {}
        self.var_traffic: Dict[int, Any] = {}
        self.last_result = None

        # frame isolation variables
        self.var_pcp: Dict[int, Any] = {}
        # Link -> stream -> queuing_interval
        self.var_queuing: Dict[Tuple[int, int], Dict[int, List[Any]]] = {}

    def has_var(self, var_name):
        in_admitted_clusters = var_name in self.var_admitted_clusters
        in_admitted_streams = var_name in self.var_admitted_streams
        in_traffic = var_name in self.var_traffic
        in_tree_usage = var_name in self.var_tree_usage
        in_link_usage = var_name in self.var_link_usage
        in_pcp = var_name in self.var_pcp
        in_queuing = var_name in self.var_queuing
        return in_admitted_clusters or in_admitted_streams or in_traffic or in_tree_usage or in_link_usage or in_pcp or in_queuing

    def schedule_time_step(self, time_step: TimeStep, routes, multicast_mapping) -> ResultStruct:
        start_step_time = time.time()
        if len(time_step.removeClusters) > 0:
            self.remove_clusters(time_step.removeClusters, routes)
        self.add_new_clusters(routes, time_step.addClusters)

        if self.details.verbose:
            print("create model...")
        model = self.create_model(routes, time_step.addClusters, multicast_mapping)

        if self.details.verbose:
            print("solve model...")

        if self.details.cplex_path is not None:
            model_result = model.solve(TimeLimit=self.details.timeout,
                                       Workers=self.details.threads,
                                       LogVerbosity='Terse' if self.details.verbose else 'Quiet',
                                       execfile=self.details.cplex_path)
        else:
            model_result = model.solve(TimeLimit=self.details.timeout,
                                       Workers=self.details.threads,
                                       LogVerbosity='Terse' if self.details.verbose else 'Quiet')

        total_step_time = time.time() - start_step_time
        self.last_result = model_result.solution

        if self.details.verbose:
            print("extract result...")

        result_struct = self.extract_result(model_result, time_step, total_step_time, multicast_mapping)

        if result_struct.flows_rejected > 0:
            if len(result_struct.solution_variables) > 0:
                # remove rejected
                clusters_to_remove = [cluster_id for cluster_id, var in self.var_admitted_clusters.items() if
                                      model_result.solution.var_solutions_dict[var.name].get_value() == 0]
            else:
                # all new ones were reject
                clusters_to_remove = [cluster.cluster_id for cluster in time_step.addClusters]
            self.remove_clusters(clusters_to_remove, routes)

        if self.details.verbose:
            # model_result.write()
            # print only parts of the result

            output = []
            for variable_name, variable in model_result.solution.var_solutions_dict.items():
                if type(variable_name) is int:
                    continue
                if "link_usage_stream_" in str(variable_name):
                    information = parse("link_usage_stream_{:d}_link_{:d}-{:d}_frame_{:d}", variable_name)
                    if (information[1] == 0 and information[2] == 1) or information[1] == 3 or information[1] == 4 or \
                            (information[1] == 0 and information[2] == 5):
                        output.append(f'{variable}')
                    continue
                if "queuing_link_" in str(variable_name) and "0-1" not in str(variable_name):
                    continue
                output.append(f'{variable}')
            for line in sorted(output):
                print(line)

        if self.details.scheduling_output:
            self.write_schedule_output_to_file(model_result, routes)

        return result_struct

    def extract_result(self, model_result, time_step, total_step_time, multicast_destination_mapping) -> ResultStruct:

        traffic_total = sum(
            [stream.get_traffic() * len(multicast_destination_mapping[stream.destination]) for stream in
             self.streams.values()])
        if (model_result is not None and model_result.solution is not None
                and model_result.solution.var_solutions_dict is not None):
            # valid result, extract meta data

            solution_variables = model_result.solution.var_solutions_dict

            def extract_id(var_name: str):
                return int(str(var_name).rsplit('_', maxsplit=1)[1])

            no_clusters_admitted = sum([solution_variables[var.name].get_value()
                                        for var in self.var_admitted_clusters.values()])

            rejected_clusters = [extract_id(var.name) for var in self.var_admitted_clusters.values() if
                                 solution_variables[var.name].get_value() == 0]
            no_clusters_rejected = len(rejected_clusters)

            no_streams_admitted = sum([solution_variables[var.name].get_value()
                                       for var in self.var_admitted_streams.values()])
            rejected_streams = [extract_id(var.name) for var in self.var_admitted_streams.values() if
                                solution_variables[var.name].get_value() == 0]

            no_streams_rejected = len(rejected_streams)

            traffic_admitted = sum([solution_variables[var.name].get_value()
                                    for var in self.var_traffic.values()])

            no_destinations_admitted = sum(
                [len(multicast_destination_mapping[stream.destination]) for stream in self.streams.values() if
                 solution_variables[
                     self.var_admitted_streams[stream.stream_id].name].get_value() == 1])
            no_destinations_rejected = sum(
                [len(multicast_destination_mapping[stream.destination]) for stream in self.streams.values() if
                 solution_variables[
                     self.var_admitted_streams[stream.stream_id].name].get_value() == 0])

            no_frames_admitted = sum(
                [len(multicast_destination_mapping[stream.destination]) * int(
                    self.details.hyper_cycle / stream.period)
                 for stream in self.streams.values() if
                 solution_variables[
                     self.var_admitted_streams[stream.stream_id].name].get_value() == 1])
            no_frames_rejected = sum(
                [len(multicast_destination_mapping[stream.destination]) * int(
                    self.details.hyper_cycle / stream.period)
                 for stream in self.streams.values() if
                 solution_variables[
                     self.var_admitted_streams[stream.stream_id].name].get_value() == 0])

        else:
            # invalid result, nothing is admitted
            no_clusters_admitted = 0
            no_clusters_rejected = len(self.var_admitted_clusters)
            no_streams_admitted = 0
            no_streams_rejected = len(self.var_admitted_streams)

            rejected_clusters = [cluster.cluster_id for cluster in time_step.addClusters]
            rejected_streams = [stream.stream_id for cluster in time_step.addClusters for stream in cluster.streams]

            traffic_admitted = 0
            no_destinations_admitted = 0
            no_destinations_rejected = sum([len(multicast_destination_mapping[stream.destination]) for stream
                                            in self.streams.values()])
            no_frames_admitted = 0
            no_frames_rejected = sum([len(multicast_destination_mapping[stream.destination]) * int(
                self.details.hyper_cycle / stream.period)
                                      for stream in self.streams.values()])
            solution_variables = {}

        return ResultStruct(time_step=time_step.time,
                            total_step_time=total_step_time,
                            solving_time=model_result.solver_infos['SolveTime'],
                            new_flows_requested=sum([len(cluster.streams) for cluster in time_step.addClusters]),
                            new_clusters_requested=len(time_step.addClusters),
                            clusters_admitted=no_clusters_admitted,
                            clusters_rejected=no_clusters_rejected,
                            clusters_total=no_clusters_admitted + no_clusters_rejected,
                            flows_admitted=no_streams_admitted,
                            flows_rejected=no_streams_rejected,
                            flows_total=no_streams_admitted + no_streams_rejected,
                            traffic_admitted=traffic_admitted,
                            traffic_rejected=traffic_total - traffic_admitted,
                            traffic_total=traffic_total,
                            frames_admitted=no_frames_admitted,
                            frames_rejected=no_frames_rejected,
                            frames_total=no_frames_admitted + no_frames_rejected,
                            destinations_admitted=no_destinations_admitted,
                            destinations_rejected=no_destinations_rejected,
                            destinations_total=no_destinations_admitted + no_destinations_rejected,
                            solution_variables=solution_variables,
                            rejected_clusters=rejected_clusters,
                            rejected_streams=rejected_streams)

    def remove_clusters(self, remove_clusters, routes):
        for cluster_id in remove_clusters:
            if cluster_id in self.var_admitted_clusters:
                del self.var_admitted_clusters[cluster_id]
            if cluster_id in self.clusters:
                self.remove_streams([stream.stream_id for stream in self.clusters[cluster_id].streams], routes)
                del self.clusters[cluster_id]

    def remove_streams(self, remove_streams, routes):
        for stream_id in remove_streams:
            if stream_id in self.var_pcp:
                del self.var_pcp[stream_id]
            if stream_id in self.var_tree_usage:
                del self.var_tree_usage[stream_id]
            if stream_id in self.var_traffic:
                del self.var_traffic[stream_id]
            for tree in routes[stream_id]:
                for link in tree.get_links():
                    if stream_id in self.var_link_usage[link]:
                        del self.var_link_usage[link][stream_id]
                    if link in self.var_queuing:
                        if stream_id in self.var_queuing[link]:
                            del self.var_queuing[link][stream_id]
            if stream_id in self.var_admitted_streams:
                del self.var_admitted_streams[stream_id]
            if stream_id in self.streams:
                del self.streams[stream_id]

    def add_new_clusters(self, routes, add_clusters: List[Cluster]):
        def create_and_insert_admitted_cluster_variable(cluster: Cluster):
            self.var_admitted_clusters[cluster.cluster_id] = expression.binary_var(
                name='admit_cluster_{}'.format(cluster.cluster_id))

        for current_cluster in add_clusters:
            create_and_insert_admitted_cluster_variable(current_cluster)
            self.clusters[current_cluster.cluster_id] = current_cluster
            self.add_new_streams(routes, current_cluster.streams)

    def add_new_streams(self, routes, add_streams: List[Stream]):
        def create_and_insert_admitted_streams_variable(stream: Stream):
            self.var_admitted_streams[stream.stream_id] = expression.binary_var(
                name='admit_stream_{}'.format(stream.stream_id))

        def create_and_insert_traffic_variable(stream: Stream):
            self.var_traffic[stream.stream_id] = expression.float_var(
                name='traffic_stream_{}'.format(stream.stream_id))

        def create_and_insert_tree_usage_variable(stream: Stream):
            self.var_tree_usage[stream.stream_id] = expression.binary_var_list(
                size=len(routes[stream.stream_id]),
                name='tree_usage_stream_{}'.format(stream.stream_id))

        def create_and_insert_link_usage_variables(stream: Stream, link: Tuple[int, int]):
            if link not in self.var_link_usage:
                self.var_link_usage[link] = {}
            if stream.stream_id not in self.var_link_usage[link]:
                temp_variables = []
                for frame in range(stream.get_number_of_frames(self.details.hyper_cycle)):
                    temp_variables.append(
                        expression.interval_var(start=(frame * stream.period, (frame + 1) * stream.period),
                                                end=(frame * stream.period, (frame + 1) * stream.period),
                                                length=stream.get_transmission_time(),
                                                optional=True,
                                                name='link_usage_stream_{}_link_{}-{}_frame_{}'.format(
                                                    stream.stream_id, link[0], link[1], frame))
                    )
                self.var_link_usage[link][stream.stream_id] = temp_variables

        def create_pcp_variable(stream: Stream):
            if not self.details.no_wait:
                self.var_pcp[stream.stream_id] = expression.integer_var(0, 7, name='pcp_{}'.format(stream.stream_id))

        def create_queuing_variables(stream: Stream, link: Tuple[int, int]):
            if self.details.no_wait:
                # no queuing needed
                return
            if link not in self.var_queuing:
                self.var_queuing[link] = {}
            if stream.stream_id not in self.var_queuing[link]:
                temp_variables = []
                for frame in range(stream.get_number_of_frames(self.details.hyper_cycle)):
                    temp_variables.append(
                        expression.interval_var(start=(frame * stream.period, (frame + 1) * stream.period),
                                                end=(frame * stream.period, (frame + 1) * stream.period),
                                                optional=True,
                                                name='queuing_link_{}-{}_stream_{}_frame_{}'.format(
                                                    link[0], link[1], stream.stream_id, frame))
                    )
                self.var_queuing[link][stream.stream_id] = temp_variables

        # start add_new_streams function
        for current_stream in add_streams:
            self.streams[current_stream.stream_id] = current_stream

            create_and_insert_admitted_streams_variable(current_stream)
            create_and_insert_traffic_variable(current_stream)
            create_and_insert_tree_usage_variable(current_stream)
            for current_tree in routes[current_stream.stream_id]:
                for current_link in current_tree.get_links():
                    create_and_insert_link_usage_variables(current_stream, current_link)
                    create_queuing_variables(current_stream, current_link)

            create_pcp_variable(current_stream)

    def create_model(self, routes, add_clusters, multicast_destination_mapping) -> CpoModel:

        def create_cluster_constraints(cluster: Cluster):
            # either admit all streams in a cluster or none
            for a in [self.var_admitted_streams[stream.stream_id] for stream in cluster.streams]:
                model.add_constraint(self.var_admitted_clusters[cluster.cluster_id] == a)

        def create_traffic_constraint(stream: Stream):
            # map traffic to the variables
            model.add_constraint(
                self.var_admitted_streams[stream.stream_id] * stream.get_traffic()
                * len(multicast_destination_mapping[stream.destination])  # number of destinations
                == self.var_traffic[stream.stream_id]
            )

        def create_tree_usage_constraint(stream: Stream):
            model.add_constraint(
                sum_of(self.var_tree_usage[stream.stream_id]) == self.var_admitted_streams[
                    stream.stream_id])

        def create_precedence_or_no_wait_constraints(stream: Stream):
            delay = Globals.PROCESSING_DELAY + Globals.PROPAGATION_DELAY
            for tree_index, current_tree in enumerate(routes[stream.stream_id]):
                links_by_hop = current_tree.get_links_by_hop()
                for hop in range(1, len(links_by_hop)):
                    predecessor_link = links_by_hop[hop - 1][0]

                    for next_link in links_by_hop[hop]:
                        # no-wait
                        for current_frame in range(stream.get_number_of_frames(self.details.hyper_cycle)):
                            if self.details.no_wait:
                                # enforce strict no-wait constraints
                                model.add_constraint(
                                    if_then(
                                        self.var_tree_usage[stream.stream_id][tree_index] == 1,
                                        # end_at_start not possible, since we need a boolean expression
                                        end_of(self.var_link_usage[predecessor_link][stream.stream_id][
                                                   current_frame]) + delay ==
                                        start_of(self.var_link_usage[next_link][stream.stream_id][current_frame])
                                    )
                                )
                            else:
                                # enforce precedence constraints
                                model.add_constraint(
                                    if_then(
                                        self.var_tree_usage[stream.stream_id][tree_index] == 1,
                                        # end_before_start not possible, since we need a boolean expression
                                        end_of(self.var_link_usage[predecessor_link][stream.stream_id][
                                                   current_frame]) + delay <=
                                        start_of(self.var_link_usage[next_link][stream.stream_id][current_frame])
                                    )
                                )

        def create_release_time_jitter_constraints(stream: Stream):
            for frame_no in range(1, stream.get_number_of_frames(self.details.hyper_cycle)):
                first_link = routes[stream.stream_id][0].get_first_link()
                model.add_constraint(
                    start_at_start(
                        self.var_link_usage[first_link][stream.stream_id][frame_no - 1],
                        self.var_link_usage[first_link][stream.stream_id][frame_no],
                        delay=stream.period
                    ))

        def create_tree_usage_to_link_usage_constraints():
            for stream_id, tree_var_list in self.var_tree_usage.items():
                for tree_index, tree_variable in enumerate(tree_var_list):
                    for link in routes[stream_id][tree_index].get_links():
                        for frame in range(self.streams[stream_id].get_number_of_frames(self.details.hyper_cycle)):
                            model.add_constraint(
                                if_then(tree_variable == 1,
                                        presence_of(self.var_link_usage[link][stream_id][frame])
                                        )
                            )

        def create_no_overlap_constraints():
            for current_link, variable_dict in self.var_link_usage.items():
                if len(variable_dict) == 0:
                    continue
                model.add_constraint(no_overlap(
                    [var for sublist in variable_dict.values() for var in sublist]))

        def create_admit_old_streams_constraints():
            add_streams = [s for c in add_clusters for s in c.streams]
            for old_stream in [s for s in self.streams.values() if s not in add_streams]:
                model.add_constraint(self.var_admitted_streams[old_stream.stream_id] == 1)

        def create_partial_solution():
            partial_solution = model.create_empty_solution()
            if self.last_result is not None:
                for var_name, var_solution in self.last_result.var_solutions_dict.items():
                    if self.has_var(var_name):
                        partial_solution.add_var_solution(var_solution)
            return partial_solution

        def create_frame_isolation_constraints():
            if self.details.no_wait:
                # no-wait does not need explicit frame isolation
                return
            # for each stream, each link set the queueing variable if the link is used
            for stream in self.streams.values():
                for tree_index, tree_var in enumerate(self.var_tree_usage[stream.stream_id]):
                    for link in routes[stream.stream_id][tree_index].get_links():
                        for frame in range(stream.get_number_of_frames(self.details.hyper_cycle)):
                            model.add_constraint(
                                if_then(tree_var == 1, presence_of(self.var_queuing[link][stream.stream_id][frame]))
                            )

            def get_stream_id_from_var_name(var_name):
                return int(parse('queuing_link_{:d}-{:d}_stream_{:d}_frame_{:d}', var_name)[2])

            def share_time_window(a: CpoIntervalVar, b: CpoIntervalVar):
                return not (a.end[1] <= b.start[0] or b.end[1] <= a.start[0])

            # for each link, no overlap between the queuing variables where the streams have the same PCP value
            for link, stream_dict in self.var_queuing.items():
                for var_a, var_b in itertools.combinations(
                        [var for var_list in stream_dict.values() for var in var_list], 2):
                    stream_id_a = get_stream_id_from_var_name(var_a.name)
                    stream_id_b = get_stream_id_from_var_name(var_b.name)
                    if stream_id_a != stream_id_b and share_time_window(var_a, var_b):
                        model.add_constraint(if_then(self.var_pcp[stream_id_a] == self.var_pcp[stream_id_b],
                                                     # poor no_overlap, since we need a boolean expression
                                                     overlap_length(var_a, var_b) == 0))

        # start actual model creation
        model = CpoModel(name=MulticastScheduling)

        # create the constraints always needed
        for current_stream in self.streams.values():
            create_traffic_constraint(current_stream)
            create_tree_usage_constraint(current_stream)

            # create the link usage constraints
            create_precedence_or_no_wait_constraints(current_stream)
            create_release_time_jitter_constraints(current_stream)

        for current_cluster in self.clusters.values():
            create_cluster_constraints(current_cluster)

        create_tree_usage_to_link_usage_constraints()
        create_no_overlap_constraints()

        if not self.details.no_wait:
            create_frame_isolation_constraints()

        if self.last_result is not None:
            # preload model with previous solution
            model.set_starting_point(create_partial_solution())

            # ensure the old streams need to be admitted
            create_admit_old_streams_constraints()

        if self.details.optimize_traffic:
            model.maximize(sum_of(self.var_traffic.values()))
        else:
            model.maximize(sum_of(self.var_admitted_clusters.values()))

        return model

    def write_schedule_output_to_file(self, result: CpoSolveResult, routes: Dict[int, List[MulticastTree.Tree]]):

        output = {'properties': {
            'hyper_cycle': self.details.hyper_cycle,
            'link speed mbps': 1000,
            'processing delay': 4,
            'propagation delay': 1
        }, 'streams': []}

        stream: Stream
        for stream in self.streams.values():
            if result[f'admit_stream_{stream.stream_id}'] == 0:
                continue

            def get_used_route():
                for i in range(len(routes[stream.stream_id])):
                    tree_used = result[f'tree_usage_stream_{stream.stream_id}_{i}']
                    if tree_used:
                        return routes[stream.stream_id][i]
                return None

            tree: MulticastTree.Tree = get_used_route()

            route = []
            for link_group in tree.get_links_by_hop():
                for link in link_group:
                    route.append([link[0], link[1]])

            first_link = tree.get_first_link()
            offset = result[
                f'link_usage_stream_{stream.stream_id}_link_({first_link[0]}, {first_link[1]})_frame_0'].start

            output['streams'].append({
                'destination': stream.destination,
                'frame size': stream.frame_size,
                'offset': offset,
                'period': stream.period,
                'route': route,
                'source': tree.get_first_link()[0],
                'stream id': stream.stream_id,
                'pcp': result[f'pcp_{stream.stream_id}'] if f'pcp_{stream.stream_id}' in result else 7
            })

        json.dump(output, open(self.details.scheduling_output, 'w'), indent=4)
