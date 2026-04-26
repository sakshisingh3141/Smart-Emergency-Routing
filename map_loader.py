import osmnx as ox
import os

def load_map():

    if os.path.exists("map.graphml"):
        print("📂 Loading map...")
        return ox.load_graphml("map.graphml")

    print("🌍 Downloading map...")

    G = ox.graph_from_point(
        (30.3165, 78.0322),
        dist=800,   # smaller = faster
        network_type="drive"
    )

    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    ox.save_graphml(G, "map.graphml")

    return G