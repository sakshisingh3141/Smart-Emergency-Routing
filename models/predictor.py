import torch
import numpy as np
import joblib

from models.lstm_model import LSTMModel
from models.gru_model import GRUModel

# Load models
lstm = LSTMModel()
lstm.load_state_dict(torch.load("model_lstm.pth", map_location="cpu"))
lstm.eval()

gru = GRUModel()
gru.load_state_dict(torch.load("model_gru.pth", map_location="cpu"))
gru.eval()

rf = joblib.load("model_rf.pkl")

print("✅ All models loaded successfully")


def create_sequence(features):
    seq = []
    for _ in range(10):
        noise = np.random.uniform(-0.05, 0.05)
        temp = features.copy()
        temp[1] += noise  # add variation in travel_time
        seq.append(temp)
    return seq


def predict_traffic(length, travel_time, road_type, hour, rain, accident):

    features = [
        length / 500,
        travel_time / 120,
        road_type / 5,
        hour / 24,
        rain,
        accident
    ]

    seq = create_sequence(features)
    x = torch.tensor([seq], dtype=torch.float32)

    with torch.no_grad():
        lstm_pred = lstm(x).item()
        gru_pred = gru(x).item()

    rf_pred = rf.predict([features])[0]

    # Ensemble
    pred = (lstm_pred + gru_pred + rf_pred) / 3

    return max(0.7, min(3.0, pred))