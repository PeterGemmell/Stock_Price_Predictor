# This program uses an artificial recurrent nueral network called Long Short Term Memory or (LSTM) for short.
# We are going to use the LSTM to predict the closing price of a particular company's stock, using the past 60 day stock price.

# Here we are importing the required libraries.
# Also had to install TensorFlow in terminal.
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# Here we are going to get the stock quote. Creating a variable called df, short for data frame.
df = web.DataReader('AAPL', data_source = 'yahoo', start = '2015-01-01', end = '2020-02-17')
# Show the data.
print(df)
