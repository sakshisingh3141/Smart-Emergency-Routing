import torch
import numpy as np

def graph_to_pyg(G):
    node_map = {node: i for i, node in enumerate(G.nodes())}

    x = []
    edge_index = []

    for node in G.nodes():
        outgoing_edges = list(G.out_edges(node, keys=True, data=True))

        if len(outgoing_edges) == 0:
            # if no outgoing edges, keep safe default values
            avg_length = 0.0
            avg_travel_time = 0.0
            avg_dynamic_time = 0.0
            avg_risk = 0.0
            avg_speed = 0.0
        else:
            lengths = []
            travel_times = []
            dynamic_times = []
            risks = []
            speeds = []

            for u, v, key, data in outgoing_edges:
                lengths.append(float(data.get("length", 0.0)))
                travel_times.append(float(data.get("travel_time", 0.0)))
                dynamic_times.append(float(data.get("dynamic_time", data.get("travel_time", 0.0))))
                risks.append(float(data.get("risk", 0.0)))
                speeds.append(float(data.get("speed_kph", 0.0)))

            avg_length = np.mean(lengths)
            avg_travel_time = np.mean(travel_times)
            avg_dynamic_time = np.mean(dynamic_times)
            avg_risk = np.mean(risks)
            avg_speed = np.mean(speeds)

        degree = float(G.out_degree(node))

        # 6 node features
        features = [
            avg_length / 500.0,
            avg_travel_time / 120.0,
            avg_dynamic_time / 120.0,
            avg_risk,
            avg_speed / 100.0,
            degree / 10.0
        ]

        x.append(features)

    for u, v, key, data in G.edges(keys=True, data=True):
        i = node_map[u]
        j = node_map[v]

        edge_index.append([i, j])

        # add reverse edge also for better message passing
        edge_index.append([j, i])

    x = torch.tensor(x, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return x, edge_index, node_map