from dataclasses import dataclass
import math
from typing import Dict, Any, List


@dataclass
class Details:
    timeout: int
    threads: int
    verbose: bool
    hyper_cycle: int
    optimize_traffic: bool
    no_wait: bool
    split_multicasts: int
    scheduling_output: str = None
    cplex_path: str = None


@dataclass
class Stream:
    stream_id: int
    period: int
    frame_size: int
    source: int
    destination: int

    @classmethod
    def from_dict(cls, input_dict):
        stream_id = input_dict.get('flowID', 0)
        period = input_dict.get('period', 0)
        frame_size = input_dict.get('frame size', 0)
        if frame_size == 0:
            # backwards compatability with old input format
            frame_size = input_dict.get('package size', 0)
        source = input_dict.get('source', 0)
        destination = input_dict.get('destination', 0)
        return cls(stream_id, period, frame_size, source, destination)

    def get_transmission_time(self) -> int:
        return math.ceil(self.frame_size / 125)

    def get_traffic(self) -> float:
        return self.frame_size * 8 / self.period

    def get_number_of_frames(self, hyper_cycle: int) -> int:
        return int(hyper_cycle / self.period)


@dataclass
class Cluster:
    cluster_id: int
    streams: List[Stream]

    @classmethod
    def from_dict(cls, input_dict):
        cluster_id = int(input_dict['cluster_id'])
        streams = [Stream.from_dict(stream) for stream in input_dict['streams']]
        return cls(cluster_id, streams)


@dataclass
class ResultStruct:
    time_step: int
    total_step_time: float
    solving_time: float

    new_flows_requested: int
    new_clusters_requested: int

    clusters_admitted: int
    clusters_rejected: int
    clusters_total: int

    flows_admitted: int
    flows_rejected: int
    flows_total: int

    traffic_admitted: float
    traffic_rejected: float
    traffic_total: float

    frames_admitted: int
    frames_rejected: int
    frames_total: int

    destinations_admitted: int
    destinations_rejected: int
    destinations_total: int

    solution_variables: Dict[Any, Any]
    rejected_clusters: List[int]
    rejected_streams: List[int]

    def print(self):
        print("Time step: {}".format(self.time_step))
        print("Time [s] (total step, reported solving) ({}, {})".format(self.total_step_time, self.solving_time))
        print("Clusters (new/admit/reject/total) ({},{},{},{})".format(self.new_clusters_requested, self.clusters_admitted,
                                                                       self.clusters_rejected, self.clusters_total))
        print("Streams (new/admit/reject/total) ({},{},{},{})".format(self.new_flows_requested, self.flows_admitted,
                                                                      self.flows_rejected,
                                                                      self.flows_total))
        print("Ingress traffic [Mbit/s]: {}, to {} destinations, via {} frames".format(self.traffic_admitted,
                                                                                       self.destinations_admitted,
                                                                                       self.frames_admitted))

    def print_raw(self):
        step_part = self.time_step
        time_part = '{}\t{}'.format(self.total_step_time, self.solving_time)
        cluster_admission_part = '{}\t{}\t{}\t{}'.format(self.new_clusters_requested, self.clusters_admitted,
                                                         self.clusters_rejected,
                                                         self.clusters_total)
        flow_admission_part = '{}\t{}\t{}\t{}'.format(self.new_flows_requested, self.flows_admitted,
                                                      self.flows_rejected,
                                                      self.flows_total)
        traffic_part = '{}\t{}\t{}\t'.format(self.traffic_admitted, self.destinations_admitted, self.frames_admitted)
        print('{}\t{}\t{}\t{}\t{}'.format(step_part, time_part, cluster_admission_part, flow_admission_part,
                                          traffic_part))

    def print_dfsv2_style(self, timeout: int):
        cluster_part = f'{self.clusters_admitted}\t{self.clusters_rejected}\t{self.clusters_total}'
        streams_part = f'{0}\t{self.flows_admitted}\t{self.flows_rejected}\t{self.flows_total}'
        traffic_part = f'{self.traffic_admitted}\t{self.traffic_rejected}\t{self.traffic_total}'
        frames_part = f'{self.frames_admitted}\t{self.frames_rejected}\t{self.frames_total}'
        destinations_part = f'{self.destinations_admitted}\t{self.destinations_rejected}\t{self.destinations_total}'

        cg_part = '-1\t-1\t-1\t-1'
        time_part = '{solving_time:}\t{add_time:}\t{remove_dummy:}\t{collection_dummy:}\t{clean_up_dummy}'.format(
            solving_time=self.solving_time,
            add_time=self.total_step_time - self.solving_time,
            remove_dummy=0,
            collection_dummy=0,
            clean_up_dummy=0
        )
        # replaces the last cg prob value with the timeout
        cg_prob_part = f'-1\t-1\t-1\t-1\t{timeout}'

        return f'{cluster_part}\t{streams_part}\t{traffic_part}\t{frames_part}\t{destinations_part}\t{cg_part}\t{time_part}\t{cg_prob_part}'


@dataclass
class TimeStep:
    time: int
    addClusters: List[Cluster]
    removeClusters: List[int]

    @classmethod
    def from_dict(cls, input_dict):
        time = input_dict.get('time', 0)
        addClusters = [Cluster.from_dict(cluster) for cluster in input_dict.get('addClusters', [])]
        removeClusters = input_dict.get('removeClusters', [])
        return cls(time, addClusters, removeClusters)


@dataclass
class Scenario:
    time_steps: List[TimeStep]
    config_info: Dict[str, Any]
    multicast_group_info: Dict[str, Any]

    @classmethod
    def from_dict(cls, input_dict):
        time_steps = [TimeStep.from_dict(time_step) for time_step in input_dict.get('time_steps', [])]
        config_info = input_dict.get('config_info', {})
        multicast_group_info = input_dict.get('multicast_group_info', {})
        return cls(time_steps, config_info, multicast_group_info)