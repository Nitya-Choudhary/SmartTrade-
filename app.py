
import gradio as gr
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- Predict Function (Shortened Example) ---
def predict_stock(symbol):
    df = yf.download(symbol, start="2020-01-01", end="2025-10-01", progress=False)
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    if len(scaled_data) < 61:
        return f"Not enough data for {symbol}", None

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X = np.array(X).reshape(-1, 60, 1)
    y = np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X, y, epochs=1, batch_size=32, verbose=0)

    last_60 = scaled_data[-60:]
    X_test = np.reshape(last_60, (1, 60, 1))
    pred_price = scaler.inverse_transform(model.predict(X_test))[0][0]

    fig, ax = plt.subplots()
    ax.plot(df.index, df['Close'], label="Historical")
    ax.set_title(f"{symbol} Stock Forecast")
    ax.legend()
    return f"Predicted next closing price: ${pred_price:.2f}", fig

# --- Gradio Interface ---
demo = gr.Interface(
    fn=predict_stock,
    inputs=gr.Textbox(label="Enter Stock Symbol", value="AAPL"),
    outputs=[gr.Textbox(label="Prediction"), gr.Plot(label="Chart")],
    title="📈 SmartTrade: LSTM Stock Predictor",
    description="Forecast next-day closing prices using LSTM models on historical stock data."
)

demo.launch()
