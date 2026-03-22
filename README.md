# Forecasting Currency Volatility: A Deep Learning Study (GBP/IDR)

## 📌 Executive Summary
This project explores fundamental deep learning applications to predict the exchange rate between the British Pound (GBP) and the Indonesian Rupiah (IDR). The study implements Vanilla RNN, LSTM, and GRU architectures to understand how temporal dependencies in financial domain can be captured by deep learning.

## 🛠 Framework and Tools
* **Data Acquisition:** `yfinance` (Yahoo Finance API)
* **Data Manipulation:** `Pandas`, `NumPy`
* **Visualization:** `Matplotlib`, `Plotly`
* **Deep Learning Framework:** `TensorFlow` / `Keras`
* **Preprocessing:** `MinMaxScaler` (Scikit-Learn)

## 🔑 Key Components
1. **Data Pipeline**
    - **Ticker:** GBPIDR=X
    - **Period:** Historical daily closing prices (2015-2026).
    - **Scaling:** Data was normalized to a range of $[0, 1]$ to ensure stable gradient descent during the training of the neural network.
    - **Window Sliding:** Using the past 30 days to predict the next day.
2. **Architecture**
    - **Vanilla RNN:** Captures basic short-term sequential dependencies
    - **LSTM:** Designed to retain long-term memory via input, forget, and output gates
    - **GRU:** Simplified LSTM variant by using fewer gates
3. **Training**
    - **Loss Function:** Mean Squared Error (MSE)
    - **Optimizer:** Adam
    - **Data Ratio:** Train/validation/test split to evaluate generalization.
    - **Callback:** `EarlyStopping` and `ReduceLROnPlateau` to prevent overfitting and adjust learning rate.

## 🔍 Key Findings
| **Model** | **MSE** | **RMSE** | **MAE** |
|-------|-----|------|-----|
| RNN | 1,231,925.92 | 1,109.92 | 1,059.07 |
| LSTM | 1,668,575.98 | 1,291.73 | 1,177.99 |
| GRU | 969,015.54 | 984.39 | 852.75 |

**Observations:**
- GRU performed best, showing the lowest errors across all metrics.
- LSTM is underperformed, likely due to highly volatile, noisy univariate data, where long-term memory offers limited advantage.
- Vanilla RNN does not produce the best result, but still higher than LSTM

**Data Behavior Insights:**
- GBP/IDR exchange rates exhibit high volatility and sudden spikes (around 2016 and 2020 pandemic), being challenging to predict accurately and precisely.
- The models operate on a single feature (closing price), limiting the ability to capture broader market influences.
- This implementation revelas the nature of financial data where data is highly volatile, non-stationary, and uncertain, which require specific treatment.

## 📂 Project Structure
* `notebook/`: Central script containint preprocessing, pipeline, and evaluation
* `images/`: Visualizations and performance plots.

**Notes:**
> The notebook is still being actively developed, so any updates may still be added in the future.
