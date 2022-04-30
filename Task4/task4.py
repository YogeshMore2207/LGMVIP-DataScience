#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
from matplotlib import pyplot as plt
import os


# In[3]:


os.chdir("D:\Yogesh")


# In[4]:


image = cv2.imread('wolf.jpg')
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),cmap='gray')
plt.show()


# In[5]:


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(10,10))
plt.imshow(gray_image,cmap='gray')
plt.show()


# In[6]:


inverted_image = 255 - gray_image
plt.figure(figsize=(10,10))
plt.imshow(inverted_image,cmap='gray')
plt.show()


# In[7]:


blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)
plt.figure(figsize=(10,10))
plt.imshow(blurred,cmap='gray')
plt.show()


# In[8]:


inverted_blurred = 255 - blurred
pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)


# In[9]:


plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),cmap='gray')
plt.show()


# In[10]:


plt.figure(figsize=(10,10))
plt.imshow(pencil_sketch,cmap='gray')
plt.show()

