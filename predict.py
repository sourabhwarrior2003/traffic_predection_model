import pandas as pd
import torch
import numpy as np
from models.traffic_gru import TrafficGRU
from utils.preprocess import normalize, create_sequences

# Load and filter data (same as training)
df = pd.read_csv("data/traffic.csv")
junction_id = 1  # Same junction ID as training
junction_df = df[df["Junction"] == junction_id]["Vehicles"].values.reshape(-1, 1)

# Normalize and create sequences (same as training)
scaled, scaler = normalize(junction_df)
X, y = create_sequences(scaled, seq_length=10)

# Load model
model = TrafficGRU()
model.load_state_dict(torch.load("trained_model.pth", weights_only=True))
model.eval()

# Prepare input tensor: use the last sequence for prediction
x_input = torch.tensor(X[-1].reshape(1, 10, 1), dtype=torch.float32)

# Predict next value
with torch.no_grad():
    pred_scaled = model(x_input).numpy()

# If you want to get the prediction back in original scale
pred_original = scaler.inverse_transform(pred_scaled)

print("Predicted vehicles count (scaled):", pred_scaled)
print("Predicted vehicles count (original scale):", pred_original)
