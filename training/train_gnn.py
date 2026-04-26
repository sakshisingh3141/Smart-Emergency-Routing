import torch
import torch.optim as optim
import numpy as np

from models.gnn_model import GNNModel
from gnn.gnn_utils import graph_to_pyg
from map_loader import load_map
from simulation.traffic_simulator import simulate_traffic

def build_node_targets(G, node_map):
    y = np.zeros((len(node_map), 1), dtype=np.float32)

    for node, idx in node_map.items():
        outgoing_edges = list(G.out_edges(node, keys=True, data=True))

        if len(outgoing_edges) == 0:
            y[idx] = 1.0
            continue

        congestion_values = []

        for u, v, key, data in outgoing_edges:
            base = float(data.get("travel_time", 1.0))
            dyn = float(data.get("dynamic_time", base))

            if base <= 0:
                congestion = 1.0
            else:
                congestion = dyn / base

            congestion_values.append(congestion)

        y[idx] = np.mean(congestion_values)

    return torch.tensor(y, dtype=torch.float32)

# load map
G_base = load_map()

model = GNNModel(input_dim=6, hidden_dim=32)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

for epoch in range(30):
    # create fresh simulated traffic graph
    G = simulate_traffic(G_base.copy())

    x, edge_index, node_map = graph_to_pyg(G)
    y = build_node_targets(G, node_map)

    pred = model(x, edge_index)

    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model_gnn.pth")
print("✅ GNN trained with real congestion targets")