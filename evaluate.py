import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.preprocess import create_sequences, normalize
from models.traffic_gru import TrafficGRU
from utils.visualization import plot_predictions
# Load data
df = pd.read_csv("data/traffic.csv")
junction_id = 1
junction_df = df[df["Junction"] == junction_id]["Vehicles"].values.reshape(-1, 1)

# Normalize full dataset
scaled, scaler = normalize(junction_df)

# Split data: 80% train, 20% test
split_idx = int(len(scaled) * 0.8)
train_scaled = scaled[:split_idx]
test_scaled = scaled[split_idx - 10:]  # Include overlap for sequence window

# Create sequences for test data
seq_length = 10
X_test, y_test = create_sequences(test_scaled, seq_length)

# Convert to tensor
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Load model
model = TrafficGRU()
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

# Predict
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()

# Inverse scale
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test)

# Calculate metrics
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(y_true, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.title("Traffic Count Prediction on Test Set")
plt.xlabel("Time step")
plt.ylabel("Vehicles")
plt.legend()
plt.show()
# After prediction and evaluation:
plot_predictions(y_true, y_pred, scaler=scaler)