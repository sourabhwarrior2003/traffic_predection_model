
# 🚦 Traffic Volume Prediction with GRU

This project implements a traffic volume prediction system using a GRU (Gated Recurrent Unit) neural network in PyTorch. It predicts future traffic counts at specific junctions based on historical traffic data and provides a clean Streamlit-based UI for interaction.

---

## 🛣️ Real-World Use Case: Why This Project Matters

Traffic congestion at urban intersections is a growing challenge — leading to fuel waste, delays, and frustration.

This project addresses a key smart city problem: **predicting traffic congestion before it happens** using time-series data and a GRU model.

By forecasting traffic volume at specific junctions, city systems can:

- 🚦 **Dynamically adjust signal timings** to reduce waiting time
- 🚌 **Optimize bus routes and schedules** based on predicted loads
- 🚑 **Enable emergency vehicles** to choose less congested paths
- 🏙️ **Support data-driven infrastructure planning**
- 📊 **Generate live dashboards** for traffic monitoring centers

### 🔧 Practical Example:
If the model predicts a **spike in vehicle count** at **Junction 1**, traffic controllers can:
- Extend green light duration preemptively
- Suggest alternate routes to GPS/navigation systems
- Deploy traffic personnel at bottlenecks before peak load

This kind of predictive insight **transforms traffic systems from reactive to proactive**, making roads smarter and safer.

---

## 📁 Project Structure

```
traffic_predection_model/
├── app.py                    # ✅ Streamlit UI for prediction
├── train.py                  # 🧠 Script to train the GRU model
├── predict.py                # 🔮 Script to make predictions from CLI
├── evaluate.py               # 📊 Model performance and visualization
├── models/
│   └── traffic_gru.py        # 📐 GRU model definition
├── utils/
│   ├── preprocess.py         # 🔧 Data scaling, sequence creation
│   ├── visualization.py      # 📈 Plotting functions (optional)
│   └── __init__.py
├── data/
│   └── traffic.csv           # 🚘 Raw traffic dataset
├── trained_model.pth         # 💾 Trained PyTorch model (excluded from GitHub)
└── README.md                 # 📘 Project overview and guide
```

---

## 📺 Streamlit Dashboard (New!)

Launch an interactive traffic forecasting app in your browser:

```bash
streamlit run app.py
```

- Choose a junction
- View recent traffic patterns
- Predict the next traffic count
- Visualize results on a line chart

---

## 🚀 Quick Setup

Make sure you have **Python 3.8+** and install the dependencies:

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, use:

```bash
pip install pandas numpy torch matplotlib scikit-learn streamlit
```

---

## 🧠 Training the Model

To train the model on your dataset:

```bash
python train.py
```

- Trains for 10 epochs on `data/traffic.csv`
- Uses a sequence window of 10 time steps
- Saves the model to `trained_model.pth`

---

## 🔮 Making Predictions

To generate predictions from CLI:

```bash
python predict.py
```

- Uses the latest time window
- Loads the trained model and predicts the next value
- Supports inverse-scaling to get readable output

---

## 📊 Model Evaluation

To check performance:

```bash
python evaluate.py
```

Outputs:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Actual vs Predicted traffic plots

---

## 🖼️ Example Output

> You can add a visual here if needed:

![Prediction vs Actual](images/multi_step_traffic_forecast.png)

---

## ⚙️ Configuration Notes

- Sequence length: `10`
- Model: 2-layer GRU with 64 hidden units
- Optimizer: Adam (lr=0.001)
- Loss: Mean Squared Error (MSE)
- Normalization: MinMaxScaler
- Data: filtered by junction IDs (e.g., `1`, `2`, `3`, `4`)

---

## 💡 Future Improvements

- Use BiGRU or LSTM for sequence modeling
- Add support for multi-step forecasting
- Integrate advanced spatial-temporal models (e.g., GCN)
- Real-time data streaming & dashboard

---

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [METR-LA Dataset](https://github.com/liyaguang/DCRNN)
- [Traffic Forecasting Survey Paper](https://arxiv.org/abs/1708.04811)

---

## 👨‍💻 Author

**Sourabh Gorkhe**  
📧 [sourabhgorkhe22@gmail.com](mailto:sourabhgorkhe22@gmail.com)  
🔗 GitHub: [Thewarrior2003](https://github.com/Thewarrior2003)

---

## 🙌 Contributions Welcome

Pull requests and feedback are welcome! If you have suggestions, feel free to create an issue or fork and improve it.

---

*This project is built with ❤️ using PyTorch and Streamlit.*
