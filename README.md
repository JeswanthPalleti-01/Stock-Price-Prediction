# Stock Price Prediction Using Stacked LSTM

Predict future stock prices with a deep-learning model using stacked LSTM networks.

---

##  Overview

This project implements a **stacked Long Short-Term Memory (LSTM)** neural network to forecast stock prices. We use historical stock data to train a sequence-to-one regression model, evaluating performance using **Mean Absolute Error (MAE)** and **Huber loss**.

---
<img width="969" height="374" alt="Screenshot 2025-08-25 at 1 16 40 PM" src="https://github.com/user-attachments/assets/31b338a4-d7ab-4ae7-abd4-953e89663d0c" />
<img width="969" height="374" alt="Screenshot 2025-08-25 at 1 16 40 PM" src="https://github.com/user-attachments/assets/aba5d630-87fd-41c5-bcbc-1597a29e72e0" />


##  Key Highlights

- **Data Preparation**: Loads and preprocesses time series data, including feature scaling and optional feature engineering (e.g., time features like day-of-week, logarithmic returns, moving averages).
- **Sliding Window Generation**: Converts the time series into supervised learning format using  sequences (e.g., `time_steps = 30`).
- **Stacked LSTM Architecture**:
  - 1st LSTM layer: `64` units, `return_sequences=True`
  - 2nd LSTM layer: `32` units
  - Optional dense layer(s) before final output
- **Regularization & Optimization**:
  - Dropout applied between layers (e.g., 30%)
  - Optimizer: Adam with learning rate tuning
  - Loss: Huber (robust to outliers)
- **Callbacks**: Early Stopping and Reduce LR on Plateau for efficient training.
- **Evaluation & Visualization**:
  - Train/validation/test split (e.g., 80/20)
  - Plot of training & validation loss curves
  - Actual vs Predicted stock price comparison

---

### 1. Data Loading & Preprocessing
- Load historical price dataset
- Handle missing values
- Feature scaling (e.g., MinMax or RobustScaler)

### 2. Feature Engineering (Optional)
- Time features: day of week, month, etc.
- Technical indicators: moving averages, RSI, etc.

### 3. Windowing Data for LSTM
- Convert time series into 3D array: `(samples, time_steps, features)`

### 4. Model Building
```python
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_steps, num_features)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])

