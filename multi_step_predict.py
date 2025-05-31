import pandas as pd
import numpy as np
import torch
from utils.preprocess import normalize
from models.traffic_gru import TrafficGRU
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/traffic.csv")
junction_id = 1
junction_df = df[df["Junction"] == junction_id]["Vehicles"].values.reshape(-1, 1)

# Normalize
scaled, scaler = normalize(junction_df)

seq_length = 10
num_future_steps = 10  # Predict next 10 steps

# Load model
model = TrafficGRU()
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

# Take last sequence from data as input
input_seq = scaled[-seq_length:].reshape(1, seq_length, 1)
input_tensor = torch.tensor(input_seq, dtype=torch.float32)

predictions = []

with torch.no_grad():
    for _ in range(num_future_steps):
        pred = model(input_tensor).numpy()  # predict next step
        predictions.append(pred[0, 0])

        # Append prediction to input sequence and slide window
        new_seq = np.append(input_tensor.numpy()[0, 1:, 0], pred[0, 0])
        input_tensor = torch.tensor(new_seq.reshape(1, seq_length, 1), dtype=torch.float32)

# Inverse scale predictions
predictions = np.array(predictions).reshape(-1, 1)
predictions_original = scaler.inverse_transform(predictions)

print(f"Predicted next {num_future_steps} traffic counts:")
print(predictions_original.flatten())

# Plot predictions
plt.plot(range(len(junction_df)), junction_df, label="Historical")
plt.plot(range(len(junction_df), len(junction_df) + num_future_steps), predictions_original, label="Predicted")
plt.xlabel("Time step")
plt.ylabel("Vehicle Count")
plt.title("Multi-step Traffic Forecast")
plt.legend()
plt.show()
