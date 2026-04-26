from flask import Flask, render_template, jsonify, request
from map_loader import load_map
from simulation.traffic_simulator import simulate_traffic
from routing import get_path
from utils import get_nearest_nodes
from hospital import HOSPITALS
import networkx as nx
import osmnx as ox
import math
from dotenv import load_dotenv

from gnn.gnn_predictor import apply_gnn_weights
from rl.rl_predictor import rl_route
from live_traffic import get_flow_segment, get_incidents

load_dotenv()

app = Flask(__name__)

print("🚀 Starting app...")

G = load_map()

if not nx.is_strongly_connected(G):
    G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
    print("✅ Using largest connected component")


def convert(o):
    import numpy as np
    import torch

    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.int32, np.int64)):
        return int(o)
    if isinstance(o, torch.Tensor):
        return o.item()
    if isinstance(o, list):
        return [convert(i) for i in o]
    if isinstance(o, dict):
        return {k: convert(v) for k, v in o.items()}
    return o


def get_path_cost(G_graph, path, weight):
    total = 0.0

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        edge_data = G_graph.get_edge_data(u, v)
        if not edge_data:
            continue

        first_key = list(edge_data.keys())[0]
        total += float(edge_data[first_key].get(weight, 1.0))

    return total


def apply_priority_weights(G_graph, mode):
    for u, v, key, data in G_graph.edges(keys=True, data=True):
        base = float(data.get("dynamic_time", data.get("travel_time", 1.0)))
        risk = float(data.get("risk", 0.1))

        if mode == "ambulance":
            data["priority_time"] = base * 0.70 + (risk * 2)
        elif mode == "fire":
            data["priority_time"] = base * 0.75 + (risk * 3)
        elif mode == "police":
            data["priority_time"] = base * 0.80 + (risk * 2)
        else:
            data["priority_time"] = base * 0.75 + (risk * 3)

    return G_graph


def get_top_hospitals(G_graph, source, hospitals, top_k=3):
    results = []

    for hospital in hospitals:
        try:
            target = ox.distance.nearest_nodes(
                G_graph, hospital["lng"], hospital["lat"]
            )
            _, cost = get_path(G_graph, source, target, "priority_time")

            time_min = round(cost / 60, 2)
            if time_min <= 0:
                time_min = 0.1

            results.append({
                "name": hospital["name"],
                "lat": hospital["lat"],
                "lng": hospital["lng"],
                "address": hospital.get("address", "Dehradun, Uttarakhand"),
                "time": time_min
            })
        except Exception as e:
            print(f"Hospital route failed for {hospital['name']}: {e}")

    results.sort(key=lambda x: x["time"])
    return results[:top_k]


def find_hospital_by_name(query):
    query = query.strip().lower()
    if not query:
        return None

    for h in HOSPITALS:
        if h["name"].lower() == query:
            return h

    for h in HOSPITALS:
        if query in h["name"].lower():
            return h

    return None


def path_to_coords_with_endpoints(G_graph, path, start_lat, start_lng, end_lat, end_lng):
    coords = []

    # exact start marker point
    coords.append([float(start_lat), float(start_lng)])

    # road network path
    coords.extend([
        [float(G_graph.nodes[n]["y"]), float(G_graph.nodes[n]["x"])]
        for n in path
    ])

    # exact destination marker point
    coords.append([float(end_lat), float(end_lng)])

    return coords


def calculate_path_distance_km(coords):
    total_km = 0.0

    for i in range(len(coords) - 1):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[i + 1]

        dx = (lon2 - lon1) * 111.32 * math.cos(math.radians((lat1 + lat2) / 2))
        dy = (lat2 - lat1) * 110.57
        total_km += math.sqrt(dx * dx + dy * dy)

    return round(total_km, 2)


def apply_live_traffic_to_graph(G_graph, sample_every=20):
    nodes = list(G_graph.nodes())
    sampled_nodes = nodes[::sample_every] if len(nodes) > sample_every else nodes

    live_cache = {}

    for node in sampled_nodes:
        lat = float(G_graph.nodes[node]["y"])
        lng = float(G_graph.nodes[node]["x"])

        try:
            flow = get_flow_segment(lat, lng)
            live_cache[node] = flow
        except Exception:
            print("Live traffic unavailable, using simulated traffic fallback.")

    for node, flow in live_cache.items():
        try:
            fsd = flow["flowSegmentData"]
            current_speed = fsd["currentSpeed"]
            free_speed = fsd["freeFlowSpeed"]

            if free_speed <= 0:
                continue

            speed_ratio = max(0.2, min(1.5, current_speed / free_speed))

            for u, v, key, data in G_graph.out_edges(node, keys=True, data=True):
                base = float(data.get("dynamic_time", data.get("travel_time", 1.0)))
                data["dynamic_time"] = base / speed_ratio
                data["live_speed_ratio"] = speed_ratio

        except Exception:
            continue

    return G_graph


def add_incident_penalty(G_graph, min_lon, min_lat, max_lon, max_lat):
    bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    try:
        incidents = get_incidents(bbox)
    except Exception:
        return G_graph

    incident_points = []

    for item in incidents.get("incidents", []):
        try:
            geometry = item.get("geometry", {})
            coords = geometry.get("coordinates", [])
            if coords:
                lng, lat = coords[0]
                incident_points.append((lng, lat))
        except Exception:
            pass

    for u, v, key, data in G_graph.edges(keys=True, data=True):
        uy = float(G_graph.nodes[u]["y"])
        ux = float(G_graph.nodes[u]["x"])

        for lng, lat in incident_points[:100]:
            dist = math.hypot(uy - lat, ux - lng)
            if dist < 0.002:
                data["dynamic_time"] = float(
                    data.get("dynamic_time", data.get("travel_time", 1.0))
                ) * 1.8
                data["risk"] = min(1.0, float(data.get("risk", 0.1)) + 0.4)
                break

    return G_graph


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/route")
def route():
    try:
        mode = request.args.get("mode", "ambulance")
        traffic_mode = request.args.get("traffic_mode", "medium")
        live_traffic = request.args.get("live_traffic", "false").lower() == "true"

        slat = float(request.args.get("start_lat"))
        slon = float(request.args.get("start_lng"))

        print("MODE:", mode)
        print("TRAFFIC MODE:", traffic_mode)
        print("LIVE TRAFFIC:", live_traffic)
        print(f"📍 Start: {slat},{slon}")

        G_temp = simulate_traffic(G.copy(), traffic_mode=traffic_mode)

        if live_traffic:
            G_temp = apply_live_traffic_to_graph(G_temp)

            min_lat = min(float(G_temp.nodes[n]["y"]) for n in G_temp.nodes())
            max_lat = max(float(G_temp.nodes[n]["y"]) for n in G_temp.nodes())
            min_lon = min(float(G_temp.nodes[n]["x"]) for n in G_temp.nodes())
            max_lon = max(float(G_temp.nodes[n]["x"]) for n in G_temp.nodes())

            G_temp = add_incident_penalty(G_temp, min_lon, min_lat, max_lon, max_lat)

        G_temp = apply_priority_weights(G_temp, mode)
        source = ox.distance.nearest_nodes(G_temp, slon, slat)

        # ---------------- AMBULANCE MODE ----------------
        if mode == "ambulance":
            hospital_name = request.args.get("hospital_name", "").strip()

            # Only show nearby hospitals first
            if not hospital_name:
                top_hospitals = get_top_hospitals(G_temp, source, HOSPITALS, top_k=3)

                return jsonify(convert({
                    "mode": mode,
                    "traffic_mode": traffic_mode,
                    "live_traffic": live_traffic,
                    "status": "choose_hospital",
                    "message": "Choose a hospital from the list or search one.",
                    "top_hospitals": top_hospitals,
                    "selected_hospital": None,
                    "hospital_name": None,
                    "hospital_lat": None,
                    "hospital_lng": None,
                    "destination": None,
                    "normal_route": [],
                    "emergency_route": [],
                    "gnn_route": [],
                    "rl_route": [],
                    "normal_time": 0,
                    "emergency_time": 0,
                    "gnn_time": 0,
                    "rl_time": 0,
                    "time_saved": 0,
                    "distance_km": 0
                }))

            selected_hospital = find_hospital_by_name(hospital_name)

            if not selected_hospital:
                return jsonify(convert({
                    "mode": mode,
                    "traffic_mode": traffic_mode,
                    "live_traffic": live_traffic,
                    "status": "hospital_not_found",
                    "message": f"Hospital '{hospital_name}' not found.",
                    "top_hospitals": get_top_hospitals(G_temp, source, HOSPITALS, top_k=3),
                    "selected_hospital": None,
                    "hospital_name": None,
                    "hospital_lat": None,
                    "hospital_lng": None,
                    "destination": None,
                    "normal_route": [],
                    "emergency_route": [],
                    "gnn_route": [],
                    "rl_route": [],
                    "normal_time": 0,
                    "emergency_time": 0,
                    "gnn_time": 0,
                    "rl_time": 0,
                    "time_saved": 0,
                    "distance_km": 0
                }))

            target = ox.distance.nearest_nodes(
                G_temp, selected_hospital["lng"], selected_hospital["lat"]
            )

            normal_path, normal_cost = get_path(G_temp, source, target, "dynamic_time")
            emergency_path, emergency_cost = get_path(G_temp, source, target, "priority_time")

            G_gnn = apply_gnn_weights(G_temp.copy())
            gnn_path, gnn_cost = get_path(G_gnn, source, target, "gnn_time")

            try:
                rl_path = rl_route(G_temp, source, target)
                if rl_path is None or len(rl_path) < 3:
                    raise Exception("RL path incomplete")
                rl_cost = get_path_cost(G_temp, rl_path, "priority_time")
            except Exception as e:
                print("⚠️ RL fallback triggered:", e)
                rl_path = normal_path
                rl_cost = normal_cost

            normal_coords = path_to_coords_with_endpoints(
                G_temp, normal_path, slat, slon,
                selected_hospital["lat"], selected_hospital["lng"]
            )

            emergency_coords = path_to_coords_with_endpoints(
                G_temp, emergency_path, slat, slon,
                selected_hospital["lat"], selected_hospital["lng"]
            )

            gnn_coords = path_to_coords_with_endpoints(
                G_gnn, gnn_path, slat, slon,
                selected_hospital["lat"], selected_hospital["lng"]
            )

            rl_coords = path_to_coords_with_endpoints(
                G_temp, rl_path, slat, slon,
                selected_hospital["lat"], selected_hospital["lng"]
            )

            distance_km = calculate_path_distance_km(emergency_coords)

            print("NORMAL COST:", normal_cost)
            print("EMERGENCY COST:", emergency_cost)
            print("GNN COST:", gnn_cost)
            print("RL COST:", rl_cost)

            return jsonify(convert({
                "mode": mode,
                "traffic_mode": traffic_mode,
                "live_traffic": live_traffic,
                "status": "route_ready",
                "message": f"Selected Hospital: {selected_hospital['name']}",
                "selected_hospital": selected_hospital,
                "hospital_name": selected_hospital["name"],
                "hospital_lat": selected_hospital["lat"],
                "hospital_lng": selected_hospital["lng"],
                "destination": {
                    "lat": selected_hospital["lat"],
                    "lng": selected_hospital["lng"]
                },
                "top_hospitals": get_top_hospitals(G_temp, source, HOSPITALS, top_k=3),
                "normal_route": normal_coords,
                "emergency_route": emergency_coords,
                "gnn_route": gnn_coords,
                "rl_route": rl_coords,
                "normal_time": round(normal_cost / 60, 2),
                "emergency_time": round(emergency_cost / 60, 2),
                "gnn_time": round(gnn_cost / 60, 2),
                "rl_time": round(rl_cost / 60, 2),
                "time_saved": round((normal_cost - emergency_cost) / 60, 2),
                "distance_km": distance_km
            }))

        # ---------------- POLICE / FIRE MODE ----------------
        elat = float(request.args.get("end_lat"))
        elon = float(request.args.get("end_lng"))
        print(f"📍 End: {elat},{elon}")

        source, target = get_nearest_nodes(G_temp, slat, slon, elat, elon)

        normal_path, normal_cost = get_path(G_temp, source, target, "dynamic_time")
        emergency_path, emergency_cost = get_path(G_temp, source, target, "priority_time")

        G_gnn = apply_gnn_weights(G_temp.copy())
        gnn_path, gnn_cost = get_path(G_gnn, source, target, "gnn_time")

        try:
            rl_path = rl_route(G_temp, source, target)
            if rl_path is None or len(rl_path) < 3:
                raise Exception("RL path incomplete")
            rl_cost = get_path_cost(G_temp, rl_path, "priority_time")
        except Exception as e:
            print("⚠️ RL fallback triggered:", e)
            rl_path = normal_path
            rl_cost = normal_cost

        normal_coords = path_to_coords_with_endpoints(
            G_temp, normal_path, slat, slon, elat, elon
        )

        emergency_coords = path_to_coords_with_endpoints(
            G_temp, emergency_path, slat, slon, elat, elon
        )

        gnn_coords = path_to_coords_with_endpoints(
            G_gnn, gnn_path, slat, slon, elat, elon
        )

        rl_coords = path_to_coords_with_endpoints(
            G_temp, rl_path, slat, slon, elat, elon
        )

        distance_km = calculate_path_distance_km(emergency_coords)

        print("NORMAL COST:", normal_cost)
        print("EMERGENCY COST:", emergency_cost)
        print("GNN COST:", gnn_cost)
        print("RL COST:", rl_cost)

        return jsonify(convert({
            "mode": mode,
            "traffic_mode": traffic_mode,
            "live_traffic": live_traffic,
            "status": "route_ready",
            "message": "Route ready.",
            "selected_hospital": None,
            "hospital_name": None,
            "hospital_lat": elat,
            "hospital_lng": elon,
            "destination": {
                "lat": elat,
                "lng": elon
            },
            "top_hospitals": [],
            "normal_route": normal_coords,
            "emergency_route": emergency_coords,
            "gnn_route": gnn_coords,
            "rl_route": rl_coords,
            "normal_time": round(normal_cost / 60, 2),
            "emergency_time": round(emergency_cost / 60, 2),
            "gnn_time": round(gnn_cost / 60, 2),
            "rl_time": round(rl_cost / 60, 2),
            "time_saved": round((normal_cost - emergency_cost) / 60, 2),
            "distance_km": distance_km
        }))

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({
            "mode": "ambulance",
            "traffic_mode": "medium",
            "live_traffic": False,
            "status": "error",
            "message": str(e),
            "selected_hospital": None,
            "hospital_name": None,
            "hospital_lat": None,
            "hospital_lng": None,
            "destination": None,
            "top_hospitals": [],
            "normal_route": [],
            "emergency_route": [],
            "gnn_route": [],
            "rl_route": [],
            "normal_time": 0,
            "emergency_time": 0,
            "gnn_time": 0,
            "rl_time": 0,
            "time_saved": 0,
            "distance_km": 0
        })


if __name__ == "__main__":
    app.run(debug=True, port=5050)
    