import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.traffic_gru import TrafficGRU
from utils.preprocess import create_sequences, normalize

# Load and filter data
df = pd.read_csv("data/traffic.csv")

print("Available junctions:", df["Junction"].unique())
junction_id = 1  # Use any valid ID from the print above

junction_df = df[df["Junction"] == junction_id]["Vehicles"].values.reshape(-1, 1)

if junction_df.shape[0] == 0:
    raise ValueError(f"No data found for junction ID {junction_id}")

# Normalize and create sequences
scaled, scaler = normalize(junction_df)
X, y = create_sequences(scaled, seq_length=10)

# PyTorch data prep
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train
model = TrafficGRU()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for xb, yb in loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# ✅ Save the trained model
torch.save(model.state_dict(), "trained_model.pth")
print("✅ Model saved as trained_model.pth")
