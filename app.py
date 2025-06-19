import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

st.title('Stock trend prediction')
user_input = st.text_input('Enter stock ticker', 'AAPL')
data = yf.download(user_input, start="2010-01-01", end="2024-01-01")

#describing
st.subheader('Data from 2010 - 2024')
st.write(data.describe())

#visualization
st.subheader('Closing price vs time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing price vs time chart with 100 moving average')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close, label = 'Actual')
plt.plot(ma100, 'r', label='Mean of past 100 days')
plt.legend()
st.pyplot(fig)

st.subheader('Closing price vs time chart with 100MA and 200MA')
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close, label = 'Actual')
plt.plot(ma100, 'r', label='Mean of past 100 days')
plt.plot(ma200, 'g', label='Mean of past 200 days')
plt.legend()
st.pyplot(fig)


#spliting data into training and testing
data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70) : int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler(feature_range=(0,1))

data_training_array = scalar.fit_transform(data_training)


#SPLITTING DATA INTO X_train and y_train
x_train = []
y_train = []
for i in range (100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100 : i])
    y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)











