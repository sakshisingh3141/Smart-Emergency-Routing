import torch
import numpy as np
from models.gru_model import GRUModel

model = GRUModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

def generate_data(batch=32):
    X, Y = [], []

    for _ in range(batch):
        length = np.random.uniform(50, 500)
        base = np.random.uniform(10, 120)
        road = np.random.randint(1, 5)
        hour = np.random.randint(0, 24)

        rain = np.random.choice([0, 1])
        accident = 1 if np.random.rand() > 0.98 else 0

        seq = []
        for _ in range(10):
            t = base + np.random.uniform(-5, 5)
            seq.append([length/500, t/120, road/5, hour/24, rain, accident])

        congestion = 1 + base/120 + length/500 + road*0.1

        if 8 <= hour <= 10 or 17 <= hour <= 19:
            congestion *= 1.5

        congestion += 0.3 * rain + 0.5 * accident

        X.append(seq)
        Y.append([congestion])

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

for epoch in range(300):
    x, y = generate_data()
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print("Epoch", epoch, "Loss:", loss.item())

torch.save(model.state_dict(), "model_gru.pth")
print("GRU trained")