import torch
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.lstm_model import LSTMModel
from models.gru_model import GRUModel

lstm = LSTMModel()
lstm.load_state_dict(torch.load("model_lstm.pth"))
lstm.eval()

gru = GRUModel()
gru.load_state_dict(torch.load("model_gru.pth"))
gru.eval()

rf = joblib.load("model_rf.pkl")

X, Y = [], []

for _ in range(200):
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

    seq = [features]*10

    X.append(seq)
    Y.append(congestion)

X_torch = torch.tensor(X, dtype=torch.float32)

lstm_pred = lstm(X_torch).detach().numpy().flatten()
gru_pred = gru(X_torch).detach().numpy().flatten()
rf_pred = rf.predict([x[0] for x in X])

def metrics(name, y, pred):
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    print(f"{name} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    return mae

print("\n=== MODEL COMPARISON ===")

m1 = metrics("LSTM", Y, lstm_pred)
m2 = metrics("GRU", Y, gru_pred)
m3 = metrics("RF", Y, rf_pred)

best = min({"LSTM": m1, "GRU": m2, "RF": m3}, key=lambda k: {"LSTM": m1, "GRU": m2, "RF": m3}[k])

print("\n🏆 BEST MODEL:", best)