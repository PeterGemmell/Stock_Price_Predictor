
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


# Here we are going to get the stock quote. Creating a variable called df, short for dataframe.
df = web.DataReader('AAPL', data_source = 'yahoo', start = '2015-01-01', end = '2020-07-27')

# Show the data.
# print(df) Will print the historic stock data in the terminal.

# Get the number of rows and columns in the data set.
# print(df.shape) COMMENT IN TO VIEW THE AMOUNT OF COLUMNS AND ROWS.

# Visualise the closing price history. Running plt.show() opens a line graph showing closing price history over the years.
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price GBP(£)', fontsize=18)
# COMMENT BACK IN TO VIEW LINE GRAPH.
# plt.show()

# Create a new dataframe with only the 'Close column'
data = df.filter(['Close'])
# Convert the dataframe to a numpy array.
dataset = data.values
# Get the number of rows to train the LSTM model on.
# math.ceil will round up.
training_data_len = math.ceil(len(dataset) * .8)

# print(training_data_len) This returns the length of the training dataset.

# Scale the data.
scaler = MinMaxScaler(feature_range=(0,1))
# Create variable to hold the dataset that is now scaled.
scaled_data = scaler.fit_transform(dataset)

# print(scaled_data)

# Create the training dataset
# Create the scaled training dataset
# This will contain all the values from index 0 to training_data_len and also all of the columns.
train_data = scaled_data[0:training_data_len, :]

# Split the data into x_train and y_train datasets
# x_train will be the independent training features, y_train will be the dependent features.
x_train = []
y_train = []

# Here we are appending the last 60 values to the x_train dataset. This does not reach i. We also get the 0 column.
# On the loop x_train will contain 60 values and those values will be indexed from 0 to 59 and the y_train will contain the
# 61st value at postion 60.

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        # print(x_train)
        # prints the 61st value that we want our model to predict.
        # print(y_train)
        print()


# Convert the x_train and y_train to numpy arrays.
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data. This is becuase an LSTM network expects the data to be 3 dimensional and right now the data set is
# 2 dimensional.
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

# Build the LSTM model.
model = Sequential()
# Next add a layer to our model.
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
