import unittest

import IO
from MyStructs import TimeStep
from Splitting import *
from routing import Routing


class TestSplitting(unittest.TestCase):

    @staticmethod
    def count_streams_in_time_step(step: TimeStep) -> int:
        return sum([len(cluster.streams) for cluster in step.addClusters])

    def test_unicast_only(self):
        network = IO.load_network("test_data/graph_random_6-1.txt")
        scenario = IO.load_scenario("test_data/multicast_splitting/simple_unicast_scenario.json")
        mapping = Routing.create_multicast_destination_mapping(scenario)
        multicast_split_threshold = 4

        step_0 = scenario.time_steps[0]
        self.assertEqual(len(step_0.addClusters), 9)
        self.assertEqual(TestSplitting.count_streams_in_time_step(step_0), 9)

        step_1 = scenario.time_steps[-1]
        self.assertEqual(len(step_1.addClusters), 2)
        self.assertEqual(TestSplitting.count_streams_in_time_step(step_1), 2)
        self.assertEqual(len(step_1.removeClusters), 4)

        for step in scenario.time_steps:
            for cluster in step.addClusters:
                split_multicasts_in_multiple_streams(cluster, network, mapping, multicast_split_threshold)

        self.assertEqual(len(step_0.addClusters), 9)
        self.assertEqual(TestSplitting.count_streams_in_time_step(step_0), 9)
        self.assertEqual(len(step_1.addClusters), 2)
        self.assertEqual(TestSplitting.count_streams_in_time_step(step_1), 2)
        self.assertEqual(len(step_1.removeClusters), 4)

    def test_simple_multicast_case(self):
        network = IO.load_network("test_data/multicast_splitting/graph_even_grid_5.txt")
        scenario = IO.load_scenario("test_data/multicast_splitting/simple_multicast_splitting_scenario.json")
        mapping = Routing.create_multicast_destination_mapping(scenario)
        multicast_split_threshold = 4

        '''
        step 0
        224000: split into 2 groups
        224001: no split
        224002: no split
        224003: split into 2 groups
        224004: no split
        224005: split into 4 groups
        224006: split into 2 groups
        224007: no split
        step 1
        224008: no split
        224009: split into 2 groups
        224010: split into 3 groups
        224011: split into 2 groups
        224012: split into 2 groups
        '''

        step_0 = scenario.time_steps[0]
        self.assertEqual(len(step_0.addClusters), 10)
        self.assertEqual(TestSplitting.count_streams_in_time_step(step_0), 10)

        step_1 = scenario.time_steps[-1]
        self.assertEqual(len(step_1.addClusters), 10)
        self.assertEqual(TestSplitting.count_streams_in_time_step(step_1), 10)
        self.assertEqual(len(step_1.removeClusters), 0)

        for step in scenario.time_steps:
            for cluster in step.addClusters:
                split_multicasts_in_multiple_streams(cluster, network, mapping, multicast_split_threshold)

        self.assertEqual(len(step_0.addClusters), 10)
        self.assertEqual(TestSplitting.count_streams_in_time_step(step_0), 16)
        self.assertEqual(len(step_1.addClusters), 10)
        self.assertEqual(TestSplitting.count_streams_in_time_step(step_1), 15)
        self.assertEqual(len(step_1.removeClusters), 0)
