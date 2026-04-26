import torch
from models.gnn_model import GNNModel
from gnn.gnn_utils import graph_to_pyg

model = GNNModel(input_dim=6, hidden_dim=32)
model.load_state_dict(torch.load("model_gnn.pth", map_location="cpu"))
model.eval()

def apply_gnn_weights(G):
    x, edge_index, node_map = graph_to_pyg(G)

    with torch.no_grad():
        preds = model(x, edge_index).cpu().numpy().flatten()

    print("Pred min:", preds.min(), "Pred max:", preds.max())

    for node, idx in node_map.items():
        node_congestion = max(0.7, min(3.0, float(preds[idx])))

        for u, v, key, data in G.out_edges(node, keys=True, data=True):
            base = float(data.get("dynamic_time", data.get("travel_time", 1.0)))
            risk = float(data.get("risk", 0.0))

            # softer, more stable GNN cost
            data["gnn_time"] = base * (1 + 0.15 * (node_congestion - 1)) + (risk * 2)

    return G