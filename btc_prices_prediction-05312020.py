#!/usr/bin/env python
# coding: utf-8

# # Bitcoin Prices Prediction with Deep Learning/Machine Learning

# In[1]:


# Data analysis and wrangling
import pandas as pd
import numpy as np
import os
import string
import csv

# Visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn
from datetime import datetime, timedelta

# Model prediction
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend, models, layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


# In[2]:


price_data = pd.read_csv('BTC-USD_05312020.csv', 
                    header = 0, 
                    error_bad_lines=False,
                    engine='python')
                 
price_data.head()


# In[3]:


price_data.tail()


# ## Visualize with scaling the data

# In[4]:


signal = np.copy(price_data['Close'].values)
std_signal = (signal - np.mean(signal)) / np.std(signal)
series = pd.Series(std_signal)
series.describe(percentiles = [0.25,0.5,0.75,0.85,0.95])


# In[5]:


# visualize with mim max scaler
scaler = MinMaxScaler()
# transform data
minmax = scaler.fit(price_data[['Low', 'Close']])
scaled = minmax.fit_transform(price_data[['Low', 'Close']])


# In[6]:


# The time-series can be visualized with the low and Close prices comparison below:

plt.figure(figsize=(20,7))
plt.plot(np.arange(len(signal)), scaled[:,0], label = 'Scaled low price')
plt.plot(np.arange(len(signal)), scaled[:,1], label = 'Scaled closed price')

plt.xticks(np.arange(len(signal))[::20], price_data.Date[::15], rotation='vertical')
plt.legend()
plt.show()


# ## Training and Testing sets

# In[7]:


# Encode the date
price_data['date'] = pd.to_datetime(price_data['Date']).dt.date
group = price_data.groupby('date')


# In[8]:


# Split data
prediction_days = 60
Real_Price = group['Close'].mean()
price_train = Real_Price[:len(Real_Price)-prediction_days]
price_test = Real_Price[len(Real_Price)-prediction_days:]


# In[9]:


# Process data
training_set = price_train.values
training_set = np.reshape(training_set, (len(training_set), 1))


# ## Scaling the training set

# In[10]:


# Define scaler
scaler = MinMaxScaler()

training_set = scaler.fit_transform(training_set)
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]


# ## Reshape the train data for the Model

# In[11]:


X_train = np.reshape(X_train, (len(X_train), 1, 1))


# ## Build the Model

# In[12]:


# Build a RNN-LSTM model

backend.clear_session()
model = Sequential()

# Adding the input layer - LSTM layer to create our RNN
model.add(LSTM(units = 100, activation = 'sigmoid', input_shape = (None, 1)))
model.add(layers.Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 1, activation='linear'))

# Compiling the model
model.compile(loss = 'mean_squared_error',
             optimizer = 'adam',
             metrics = ['accuracy'])
model.summary()

# Fitting the model to the Training set
model.fit(X_train, y_train, 
          batch_size = 10, 
          epochs = 50, 
          verbose = 0)

# Making the price predictions
test_set = price_test.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = scaler.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
pred_btc_price = model.predict(inputs)
pred_btc_price_inverse = scaler.inverse_transform(pred_btc_price)

# Visualising the results
plt.figure(figsize = (25,15), dpi=50, facecolor='w', edgecolor='k')
ax = plt.gca()  
plt.plot(test_set, color = 'red', label = 'Actual BTC Prices')
plt.plot(pred_btc_price_inverse, color = 'blue', label = 'Predicted BTC Prices')
plt.title('BTC Price Prediction', fontsize=20)
price_test = price_test.reset_index()
x = price_test.index
labels = price_test['date']
plt.xticks(x, labels, rotation = 'vertical')
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(15)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(15)
plt.xlabel('Time(Date)', fontsize=20)
plt.ylabel('BTC Price(USD)', fontsize=20)
plt.legend(loc=2, prop={'size': 25})
plt.show()

