import MulticastScheduling
import MulticastTree
import Splitting
import routing.Routing
from MyStructs import *


class Scheduler:

    def __init__(self, details: Details, mapping):
        super().__init__()
        self.solver = MulticastScheduling.MulticastScheduling(details=details)
        self.active_streams: List[Stream] = []
        self.routes: Dict[int, List[MulticastTree.Tree]] = {}
        self.multicast_mapping = mapping
        self.split_multicasts = details.split_multicasts

    def schedule_scenario(self, scenario, router: routing.Routing.Router) \
            -> List[ResultStruct]:
        results = []

        for time_step in scenario.time_steps:
            if self.split_multicasts:
                for cluster in time_step.addClusters:
                    Splitting.split_multicasts_in_multiple_streams(cluster=cluster, network=router.network,
                                                                   mapping=self.multicast_mapping,
                                                                   threshold=self.split_multicasts)

            new_routes = router.compute_candidate_routes(time_step=time_step, multicast_mapping=self.multicast_mapping)
            self.routes.update(new_routes)

            result = self.solver.schedule_time_step(time_step=time_step,
                                                    routes=self.routes,
                                                    multicast_mapping=self.multicast_mapping)
            for stream_id in result.rejected_streams:
                del self.routes[stream_id]

            results.append(result)

        return results
