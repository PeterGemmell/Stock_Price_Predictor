
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
df = web.DataReader('TSLA', data_source = 'yahoo', start = '2015-01-01', end = '2020-07-28')

# Show the data.
# print(df) Will print the historic stock data in the terminal.

# Get the number of rows and columns in the data set.
# print(df.shape) COMMENT IN TO VIEW THE AMOUNT OF COLUMNS AND ROWS.

# Visualise the closing price history. Running plt.show() opens a line graph showing closing price history over the years.
plt.figure(figsize=(16,8))
plt.title('Tesla Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD($)', fontsize=18)
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

# Reshape the data. This is because an LSTM network expects the data to be 3 dimensional and right now the data set is
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

# Next we will Compile the model.
# An optimizer is used to imporve upon the loss function and the loss function is used to measure how well the model did
# on training.
model.compile(optimizer='adam', loss='mean_squared_error')

# Next we will Train the model.
# Epochs is the number of times a dataset is passed back and forward through a neural network.
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing dataset.
# Create a new array containing scaled values from 1061 to 2003? Where is 2003 coming from??? 1289 or 1401.
test_data = scaled_data[training_data_len - 60: , :]
# Create the datasets x_test and y_test
# y_test will be all of the values that we want our model to predict.
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data. Again because it is currently 2 dimensional and LSTM requires it to be 3 dimensional.
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values for the x_test dataset.
# From this we want predictions to contain the same values of our y_test dataset. We are getting these from the x_test
# dataset.
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE) this is a good measure of how accurate the model predicts the response.
# It would be good practise to evaluate your model with other metrics as well.
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
print(rmse)

# Next we want to Plot the data.
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Visualise the data. So this is, similar to at the start, giving us our line graph with the new data.
# Blue in the graph is the data the model was trained on.
# Redish Orange is the ACTUAL closing stock price for AAPL.
# Yellow is what our model PREDICTED the values to be.
plt.figure(figsize=(16,8))
plt.title('Tesla Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# Next I want to show the Actual Price and the Predicted prices
print(valid)

# Get the quote
tesla_quote = web.DataReader('TSLA', data_source='yahoo', start='2015-01-01', end='2020-07-28')
# Create a new dataframe
new_df = tesla_quote.filter(['Close'])
# Get the last 60 day closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
# Scale the data to be values between 0 and 1
# We are not using fit.transform as we want it to transform the data using the same min and max values that we used when we first
# transformed the data.
last_60_days_scaled = scaler.transform(last_60_days)
# Create an empty list.
X_test = []
# Append the past 60 days
X_test.append(last_60_days_scaled)
# Convert the X_test dataset to a numpy array
X_test = np.array(X_test)
# Reshape the data again so it is 3 dimensional
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Get the predicted scaled price
pred_price = model.predict(X_test)
# Undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

# Get the quote
tesla_quote2 = web.DataReader('TSLA', data_source='yahoo', start='2020-07-29', end='2020-07-29')
print(tesla_quote2['Close'])
