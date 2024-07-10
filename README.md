# Stock-Price-Prediction

![Python](https://img.shields.io/badge/Python-3.8-blue)
![Framework](https://img.shields.io/badge/Framework-Keras-red)
![Frontend](https://img.shields.io/badge/Frontend-Streamlit-green)
![API](https://img.shields.io/badge/API-yfinance-yellow)
![Library](https://img.shields.io/badge/Library-Scikit--learn-orange)

This repository contains code for predicting stock prices using Long Short-Term Memory (LSTM) neural networks. The project fetches historical stock data, preprocesses it, trains an LSTM model, and predicts future stock prices. The project is deployed using Streamlit. The [Stock_Price_Prediction.ipynb](https://github.com/harshita-daga/stock-price-prediction-lstm/blob/main/Stock_Price_Prediction.ipynb) contains the complete model.

## Link to the application
Check out the live demo: [https://stockpriceprediction-lstm/](https://stockpriceprediction-lstm.streamlit.app/)

If you can't find the stock you're looking for, simply type the name of the stock and press "enter". 

## Project Overview

This project uses the following technologies:
- `yfinance` for fetching historical stock data.
- `pandas` for data manipulation.
- `matplotlib` for data visualization.
- `scikit-learn` for data scaling.
- `keras` and `tensorflow` for building and training the LSTM model.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/harshita-daga/stock-price-prediction-lstm.git
    ```
2. Navigate to the project directory:
    ```bash
    cd stock-price-prediction-lstm
    ```
3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Fetch and preprocess historical stock data.
2. Visualize the data.
3. Preprocess the data for the LSTM model.
4. Build and train the LSTM model.
5. Predict and visualize the results.
6. Predict future stock prices.
7. Save the model.

## Deployment

The project is deployed using Streamlit. You can run the Streamlit app with the following command:
```bash
streamlit run app.py
