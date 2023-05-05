import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import custom_object_scope

start = "2010-01-01"
end = "2022-12-31"

st.title("Stock Price Prediction & Analysis")

user_input = st.text_input('Enter Stock Ticker', 'NFLX')
df = yf.download(user_input, start, end)

st.subheader('Data from 2010-2022')
st.write(df.describe())

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 200MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.7): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_train_array = scaler.fit_transform(data_train)

# Splitting XTRAIN YTRAIN
x_train = []
y_train = []

for i in range(100, data_train_array.shape[0]):
    x_train.append(data_train_array[i-100: i])
    y_train.append(data_train_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Load model with custom optimizer
with custom_object_scope({'Adam': Adam}):
    model = load_model('keras_model.h5')

# Testing
past_100_days = data_train.tail(100)
final_df = pd.concat([past_100_days, data_test], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predictions
y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plot Predictions vs Original
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Stock Price')
plt.plot(y_predicted, 'r', label = 'Predicted Stock Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
