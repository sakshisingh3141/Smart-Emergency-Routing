import matplotlib.pyplot as plt

models = ["LSTM", "GRU", "RF"]
mae = [0.22, 0.02, 0.02]
rmse = [0.24, 0.03, 0.02]

plt.figure()

plt.plot(models, mae, marker='o', label="MAE")
plt.plot(models, rmse, marker='o', label="RMSE")

plt.title("Model Comparison")
plt.xlabel("Models")
plt.ylabel("Error")
plt.legend()

plt.savefig("results.png")
plt.show()