from models.predictor import predict_traffic
from datetime import datetime
import random


def simulate_traffic(G, traffic_mode="medium"):
    current_hour = datetime.now().hour

    for u, v, key, data in G.edges(keys=True, data=True):
        base = float(data.get("travel_time", 1.0))
        length = float(data.get("length", 100.0))

        highway = data.get("highway", "residential")
        if isinstance(highway, list):
            highway = highway[0]

        road_map = {
            "motorway": 5,
            "trunk": 5,
            "primary": 4,
            "secondary": 3,
            "tertiary": 2,
            "residential": 1
        }

        road_type = road_map.get(highway, 2)

        rain = 0
        accident = 0
        extra_multiplier = 1.0

        if traffic_mode == "low":
            rain = 0
            accident = 0
            extra_multiplier = 0.85

        elif traffic_mode == "medium":
            rain = random.choice([0, 1])
            accident = 1 if random.random() > 0.98 else 0
            extra_multiplier = 1.0

        elif traffic_mode == "high":
            rain = random.choice([0, 1])
            accident = 1 if random.random() > 0.95 else 0
            extra_multiplier = 1.25

        elif traffic_mode == "rush_hour":
            rain = random.choice([0, 1])
            accident = 1 if random.random() > 0.96 else 0
            extra_multiplier = 1.5

        elif traffic_mode == "accident":
            rain = random.choice([0, 1])
            accident = 1 if random.random() > 0.85 else 0
            extra_multiplier = 1.8

        congestion = predict_traffic(
            length, base, road_type, current_hour, rain, accident
        )

        risk = 0.3 * rain + 0.5 * accident + random.uniform(0, 0.2)
        risk = min(1, risk)

        final_multiplier = congestion * extra_multiplier

        if risk > 0.97 and traffic_mode == "accident":
            data["dynamic_time"] = 999999
        else:
            data["dynamic_time"] = base * final_multiplier

        data["risk"] = risk
        data["traffic_mode"] = traffic_mode

    return G