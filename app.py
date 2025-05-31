import streamlit as st
import pandas as pd
import numpy as np
import torch
from models.traffic_gru import TrafficGRU
from utils.preprocess import create_sequences, normalize
import matplotlib.pyplot as plt

# Load and preprocess data
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

# Load trained model
@st.cache_resource
def load_model(path):
    model = TrafficGRU()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

st.title("🚦 Traffic Prediction Dashboard")
st.markdown("Predict future traffic using GRU-based model")

# Load data
df = load_data("data/traffic.csv")
junctions = sorted(df["Junction"].unique())

# Select junction
junction_id = st.selectbox("Select Junction ID", junctions)

# Filter data
junction_df = df[df["Junction"] == junction_id]["Vehicles"].values.reshape(-1, 1)

if junction_df.shape[0] < 11:
    st.error("Not enough data for prediction.")
    st.stop()

# Normalize and prepare input
scaled, scaler = normalize(junction_df)
X, y = create_sequences(scaled, seq_length=10)
x_input = torch.tensor(X[-1].reshape(1, 10, 1), dtype=torch.float32)

# Predict
model = load_model("trained_model.pth")
with torch.no_grad():
    prediction = model(x_input).numpy()
    predicted_value = scaler.inverse_transform(prediction)[0][0]

# Show prediction
st.subheader("🔮 Predicted Traffic Count:")
st.metric(label=f"Next Time Step (Junction {junction_id})", value=f"{predicted_value:.2f}")

# Plot recent data
st.subheader("📊 Recent Traffic Data")
fig, ax = plt.subplots()
ax.plot(scaler.inverse_transform(scaled[-50:]), label="Recent Traffic", color='orange')
ax.axhline(predicted_value, color='blue', linestyle='--', label="Predicted")
ax.legend()
st.pyplot(fig)
