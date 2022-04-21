#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[17]:


os.getcwd()


# In[18]:


os.chdir("D:\Yogesh")


# In[19]:


iris=pd.read_csv("Irisdata.csv")


# In[20]:


iris.shape


# In[21]:


iris.describe()


# In[22]:


iris['Species'].value_counts()


# In[23]:


iris.head()


# In[24]:


iris.tail()


# In[25]:


print("Target Labels", iris["Species"].unique())


# In[26]:


import plotly.express as px


# In[29]:


fig=px.scatter(iris, x='SepalWidthCm',y='SepalLengthCm',color='Species')
fig.show()


# In[30]:


sns.scatterplot(iris['SepalLengthCm'],iris['PetalLengthCm'],hue=iris['Species'])
plt.show()


# In[31]:


sns.boxplot(iris['Species'],iris['PetalLengthCm'])
plt.show()


# In[32]:


sns.pairplot(iris, hue='Species')


# In[33]:


numeric_columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
sns.pairplot(iris[numeric_columns])

iris['Species']


# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


x=iris.drop(columns=['Species'])
y=iris['Species']
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2)


# In[36]:


from sklearn.linear_model import LogisticRegression


# In[37]:


log_reg=LogisticRegression()


# In[38]:


log_reg.fit(x_train,y_train)


# In[39]:


log_reg.score(x_test,y_test)


# In[40]:


log_reg.score(x,y)

