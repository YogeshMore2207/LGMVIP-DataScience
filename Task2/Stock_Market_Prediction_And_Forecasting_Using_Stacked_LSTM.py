#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd #for analysis and manipulation of numerical tables
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


os.chdir("D:\Yogesh")


# In[10]:


stock_data = pd.read_csv('stock.csv')
stock_data.head()


# In[11]:


stock_data.tail()


# In[12]:


df=stock_data.reset_index()
df


# In[13]:


print(stock_data.isnull().sum())


# In[14]:


stock_data.describe()


# In[15]:


stock_data1 = stock_data.reset_index()


# In[16]:


stock_data1.shape


# In[17]:


stock_data1 = df['Close']
plt.plot(stock_data1)


# In[18]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
stock_data1 = scaler.fit_transform(np.array(stock_data1).reshape(-1,1))


# In[19]:


stock_data1


# In[20]:


train_size = int(len(stock_data1)*0.65)
test_size = len(stock_data1) - train_size
train_data, test_data = stock_data1[0:train_size,:],stock_data1[train_size:len(stock_data1),:1]


# In[21]:


def create_dataset(dataset, time_step=1):
  x, y = [],[]
  for i in range(len(dataset)-time_step-1):
    a=dataset[i:(i+time_step),0]
    x.append(a)
    y.append(dataset[i+time_step,0])
  return np.array(x),np.array(y)


# In[22]:


time_step = 100
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)


# In[23]:


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


# In[24]:


x_train.shape


# In[25]:


y_train.shape


# In[26]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.models import Sequential


# In[27]:


model = Sequential()
#adding layers
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[28]:


model.summary()


# In[29]:


model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=64,verbose=1)


# In[30]:


import tensorflow as tf #importing the tensorflow library to predict
train_prediction = model.predict(x_train)
test_prediction = model.predict(x_test)


# In[31]:


train_prediction = scaler.inverse_transform(train_prediction)
test_prediction = scaler.inverse_transform(test_prediction)


# In[32]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_prediction)) #for training data


# In[33]:


math.sqrt(mean_squared_error(y_test,test_prediction)) #for testing data


# In[34]:


look_back=100
trainPredictionPlot = np.empty_like(stock_data1) 
trainPredictionPlot[:, :] = np.nan
trainPredictionPlot[look_back:len(train_prediction)+look_back, :] = train_prediction


# In[35]:


testPredictionPlot = np.empty_like(stock_data1)
testPredictionPlot[:, :] = np.nan
testPredictionPlot[len(train_prediction)+(look_back*2)+1:len(stock_data1)-1, :] = test_prediction


# In[36]:


plt.plot(scaler.inverse_transform(stock_data1))
plt.plot(trainPredictionPlot)
plt.plot(testPredictionPlot)
plt.show()


# In[37]:


len(test_data)


# In[38]:


len(test_data), x_test.shape


# In[39]:


x_input=test_data[613:].reshape(1,-1)
x_input.shape


# In[40]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[41]:


from numpy import array

final_output=[]
n_steps=100
i=1
while(i<=30):
    
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        ypred = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,ypred))
        temp_input.extend(ypred[0].tolist())
        temp_input=temp_input[1:]
        final_output.extend(ypred.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        ypred = model.predict(x_input, verbose=0)
        print(ypred[0])
        temp_input.extend(ypred[0].tolist())
        print(len(temp_input))
        final_output.extend(ypred.tolist())
        i=i+1
    

print(final_output)


# In[42]:


stock_data = stock_data1.tolist()
stock_data.extend(final_output)
stock_data = scaler.inverse_transform(stock_data).tolist()
plt.plot(stock_data)

