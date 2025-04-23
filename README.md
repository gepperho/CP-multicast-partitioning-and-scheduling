# ConstraintProgramming-for-Multicast-Scheduling

This is a CP formulation for multicast scheduling, used in the evaluation of "Multicast-partitioning in Time-triggered Stream Planning for Time-Sensitive Networks.
When using the code, please cite the following paper:

TBA
```

```


## Requirements

- Python (tested with 3.10)
- CPLEX (tested with 22.10)

## Getting started

usage: 
```
main.py [-h] -n NETWORK -s SCENARIO [-t TIMELIMIT] [--threads THREADS] [-v] [-raw] [-r ROUTING] [-nw]
               [--multicast-splitting-threshold MULTICAST_SPLITTING_THRESHOLD] [-p CANDIDATE_PATHS] [-o]
               [--scheduling_output SCHEDULING_OUTPUT]
```

We provide small sample inputs in the `dummy_data` directory.

`python main.py -n dummy_data/graph_random_6-1.txt -s dummy_data/scenario_graph_random_6-1_mini.json -nw --multicast-splitting-threshold 3`


### Options

| Option                                                        | Description                                                                                                                           |
|---------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| -h, --help                                                    | show this help message and exit                                                                                                       |
| -n NETWORK, --network NETWORK                                 | Path to the network graph file                                                                                                        |
| -s SCENARIO, --scenario SCENARIO                              | Path to the flow scenario file                                                                                                        |
| -t TIMELIMIT, --timelimit TIMELIMIT                           | solver time limit in seconds. Use negative values for unlimited. Default: 120s                                                        |
| --threads THREADS                                             | Number of threads to be used at most                                                                                                  |
| -v, --verbose                                                 | print a lot of debug outputs                                                                                                          |
| -raw, --print-raw                                             | print results in raw format                                                                                                           |
| -r ROUTING, --routing ROUTING                                 | Routing strategy for the candidate paths: [dijkstra_overlap: 1, intermediate_node_distance: 2, shortest_path_based_source_tree: 3]    |
| -nw, --no-wait                                                | Enables no-wait scheduling                                                                                                            |
| --multicast-splitting-threshold MULTICAST_SPLITTING_THRESHOLD | Split the multicasts in several smaller multicasts/unicasts. Parameter provides the threshold distance to split. Use 0 for no splits. |
| -p CANDIDATE_PATHS, --candidate-paths CANDIDATE_PATHS         | Number of candidate paths to be computed.                                                                                             |
| -o, --optimize-traffic                                        | Set this flag to optimize the aggregated network traffic instead of the number of streams                                             |
| --scheduling_output SCHEDULING_OUTPUT                         | Path to the scheduling output file                                                                                                    |

Note that using more than one candidate path will increase the runtime significantly.
