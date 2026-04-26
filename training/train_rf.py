import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

X, Y = [], []

for _ in range(3000):
    length = np.random.uniform(50, 500)
    base = np.random.uniform(10, 120)
    road = np.random.randint(1, 5)
    hour = np.random.randint(0, 24)

    rain = np.random.choice([0, 1])
    accident = 1 if np.random.rand() > 0.98 else 0

    features = [
        length/500,
        base/120,
        road/5,
        hour/24,
        rain,
        accident
    ]

    congestion = 1 + base/120 + length/500 + road*0.1

    if 8 <= hour <= 10 or 17 <= hour <= 19:
        congestion *= 1.5

    congestion += 0.3 * rain + 0.5 * accident

    X.append(features)
    Y.append(congestion)

model = RandomForestRegressor(n_estimators=150)
model.fit(X, Y)

joblib.dump(model, "model_rf.pkl")
print("✅ RF trained")