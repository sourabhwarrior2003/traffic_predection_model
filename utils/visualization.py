import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_true, y_pred, scaler=None, title="Traffic Prediction vs Actual"):
    if hasattr(y_true, 'detach'):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, 'detach'):
        y_pred = y_pred.detach().cpu().numpy()
    if scaler:
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual", color="blue")
    plt.plot(y_pred, label="Predicted", color="red", linestyle='--')
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Vehicle Count")
    plt.legend()
    plt.show()
