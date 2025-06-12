# 📈 Stock Price Prediction Using RNN (LSTM) – Deep Learning Approach

This project presents a deep learning-based approach to predicting future stock prices using historical data from the National Stock Exchange (NSE). By leveraging Recurrent Neural Networks (RNN) with Long Short-Term Memory (LSTM) layers, this project aims to model the temporal dependencies and nonlinear patterns found in stock market data.

> 🔬 Built as part of an academic project by **Segu Aravinda Sree**, UG Scholar at Mohan Babu University, Tirupati, India.

---

## 🚀 Project Overview

📉 **Problem Statement**: Traditional models often fail to capture the complex and dynamic behavior of stock prices. This project aims to improve prediction accuracy using LSTM-based RNN models trained on past stock data.

🧠 **Solution**: We use historical closing prices of Tata Global Beverages Limited from NSE. The model is built using deep LSTM layers to predict future closing prices.

📊 **Use Case**: This can aid investors and financial analysts in identifying price trends and making data-driven decisions.

---

## 📂 Files Included

- `rnn_final.py`: Final Python script to build, train, and evaluate the RNN-LSTM model
- `NSE-Tata-Global-Beverages-Limited.csv`: Dataset of stock closing prices (add manually if not uploaded)
- `README.md`: This documentation

---

## 📈 Model Architecture

- 4 LSTM layers (each with 50 units) and Dropout regularization
- Final Dense layer to predict closing price
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)

---

## 📊 Model Performance

| Metric        | Value    |
|---------------|----------|
| MAE           | ~4.3     |
| MSE           | ~34.8    |
| RMSE          | ~5.9     |
| R² Score      | ~0.54    |
| MAPE          | ~2%      |
| Accuracy      | ~97%     |

---

## 📉 Output

- **Actual vs Predicted Stock Prices**

![actual vs predicted](https://github.com/user-attachments/assets/441ae181-54ec-4aff-b729-1075f580b208)

The model closely tracks real closing prices, proving its ability to identify meaningful patterns from past data.

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy, Matplotlib
- TensorFlow / Keras
- Scikit-learn

---

