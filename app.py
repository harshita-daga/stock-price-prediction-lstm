from datetime import date
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model # type: ignore

# Load your trained model
model = load_model("stock_price_prediction.h5")

# Streamlit app layout
st.title("Stock Price Prediction")
st.write("Enter the stock ticker symbol to get started.")

# Input for ticker symbol
ticker = st.text_input("Ticker Symbol", "MOSCHIP.BO")

# Fetch stock data
stock = yf.Ticker(ticker)
info = stock.info

#date
currentYear= date.today().year
today = date.today().strftime("%Y-%m-%d")

# Display stock info
st.subheader(f"Information about {ticker}")
st.write(f"**Name:** {info.get('longName')}")
st.write(f"**Sector:** {info.get('sector')}")
st.write(f"**Industry:** {info.get('industry')}")
st.write(f"**Country:** {info.get('country')}")
st.write(f"**Market Cap:** {info.get('marketCap')}")
st.write(f"**Volume:** {info.get('volume')}")
st.write(f"**Website:** {info.get('website')}")

# Fetch historical market data
df = stock.history(start='2010-01-01', end=today, actions=False)
df = df.drop(['Open', 'High', 'Volume', 'Low'], axis=1)

# Plot Price over the years
st.subheader("Price over the years")
fig, ax = plt.subplots(figsize=(20, 7))
ax.plot(df['2020-05-31':today])
ax.set_title("Price over the years")
ax.set_ylabel("Price in INR")
ax.set_xlabel("Time")
st.pyplot(fig)

# Plot Price changes in year 2024
st.subheader(f"Price changes in year {currentYear}")
fig, ax = plt.subplots(figsize=(20, 7))
ax.plot(df[f'{currentYear}-01-01':today], color='green')
ax.set_title(f"Price changes in year {currentYear}")
ax.set_ylabel("Price in INR")
ax.set_xlabel("Months")
st.pyplot(fig)

# Prepare data for model prediction
data = df.values
train_len = int(len(data) * 0.92)
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = min_max_scaler.fit_transform(data)

# Create test data
test_data = scaled_data[train_len-60:, :]
x_test = []
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions
predictions = model.predict(x_test)
predictions = min_max_scaler.inverse_transform(predictions)

# Create dataframe for plotting
train_data = df[:train_len]
valid_data = df[train_len:]
valid_data['Predictions'] = predictions


# Plot Model Prediction vs Actual Price (Zoomed)
st.subheader("Model Prediction vs Actual Price ")
fig, ax = plt.subplots(figsize=(20, 7))
ax.plot(valid_data['Close'])
ax.plot(valid_data['Predictions'])
ax.set_title("Model Prediction vs Actual Price")
ax.set_ylabel("Price in INR (in Million)")
ax.set_xlabel("Date")
ax.legend(['Actual Price', 'Model Prediction'], loc='lower right', fontsize=15)
st.pyplot(fig)

# Predict tomorrow's price
last_60_days = df[-60:].values
last_60_days_scaled = min_max_scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

tomorrow_prediction = model.predict(X_test)
tomorrow_prediction = min_max_scaler.inverse_transform(tomorrow_prediction)
st.subheader("Tomorrow's Predicted Price")
st.write(f"The predicted price for tomorrow is: {tomorrow_prediction[0][0]:.2f} INR")
