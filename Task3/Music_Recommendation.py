#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[2]:


os.chdir("D:\Yogesh")


# In[3]:


song_data = pd.read_csv('songs.csv')


# In[4]:


song_data.head()


# In[5]:


user_data = pd.read_csv('users.csv')


# In[6]:


user_data.head()


# In[7]:


song_data["year"] = song_data["year"].astype('Int64')


# In[8]:


song_data.rename(columns={"song_id":"SongId","title":"Title","release":"Album","artist_name":"Artist","year":"Year"},inplace=True)


# In[9]:


user_data["listen_count"] = user_data["listen_count"].astype('Int64')


# In[10]:


user_data.rename(columns={"user_id":"UserId","song_id":"SongId","listen_count":"ListenCount"},inplace=True)


# In[11]:


final_data = pd.merge(user_data, song_data.drop_duplicates(["SongId"]), on='SongId', how='left')
final_data['Song'] = final_data['Title'] + ' by ' + final_data['Artist']
final_data = final_data.drop(['Title'],axis=1)
final_data = final_data.head(50000)
final_data.head()


# In[12]:


print(len(song_data), len(user_data))


# In[13]:


len(final_data)


# In[14]:


#Using countplot to see the number of songs per year
plt.figure(figsize=(20,10))
sns.countplot(x='Year', data=final_data[-(final_data['Year']==0)])
plt.xticks(rotation=90)
plt.title("No. of songs per year")
plt.show()


# In[15]:


#Using barplot to see the number of songs per artist
plt.figure(figsize=(20,10))
sns.barplot(final_data['Artist'].value_counts()[:10].index,final_data['Artist'].value_counts()[:10].values)
plt.title("No. of songs per artist")
plt.show()


# In[16]:


#Using barplot to see the most popular songs
plt.figure(figsize=(20,10))
sns.barplot(final_data['Song'].value_counts()[:10].values,final_data['Song'].value_counts()[:10].index)
plt.title("Most popular songs")
plt.show()


# In[17]:


class Recommendation():
    def __init__(self, data, user_id, song):
        self.data = data
        self.user_id = user_id
        self.song = song
        self.glcm = None
        
    def song_history(self, user):
        user_data = self.data[self.data[self.user_id] == user]
        return list(user_data[self.song].unique())
        
    def users(self, item):
        item_data = self.data[self.data[self.song] == item]
        return set(item_data[self.user_id].unique())
        
    def all_songs(self):
        return list(self.data[self.song].unique())
        
    def get_glcm(self, user_songs, all_songs):
        users = []        
        for p in range(0, len(user_songs)):
            users.append(self.users(user_songs[p]))   
        glcm = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)

        for p in range(0,len(all_songs)):
            songs_p_data = self.data[self.data[self.song] == all_songs[p]]
            users_p = set(songs_p_data[self.user_id].unique())
            
            for q in range(0,len(user_songs)):           
                users_q = users[q]
                users_intersection = users_p.intersection(users_q)
               
                users_union = users_p.union(users_q)
                glcm[q,p] = float(len(users_intersection))/float(len(users_union))

        return glcm

    def generate(self, user, glcm, all_songs, user_songs):
        scores = glcm.sum(axis=0)/float(glcm.shape[0])
        scores = np.array(scores)[0].tolist()
        sort_index = sorted(((e,p) for p,e in enumerate(list(scores))), reverse=True)
        columns = ['UserID', 'Song', 'Score', 'Rank']
        final_data = pd.DataFrame(columns=columns)
         
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                final_data.loc[len(final_data)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        
        print("Music Recommendations: \n")
        return final_data.drop(['UserID'], axis=1)

    def get_recommendations(self, user):
        user_songs = self.song_history(user)    
        all_songs = self.all_songs()
        glcm = self.get_glcm(user_songs, all_songs)
        return self.generate(user, glcm, all_songs, user_songs)

    def get_similar_songs(self, item_list):
        user_songs = item_list
        all_songs = self.all_songs()
        glcm = self.get_glcm(user_songs, all_songs)
        return self.generate("", glcm, all_songs, user_songs)


# In[18]:


r = Recommendation(final_data,'UserId','Song')
history=r.song_history(final_data['UserId'][5])


# In[19]:


print("Song history of the user:\n")
for song in history:
  print(song)


# In[20]:


r.get_recommendations(final_data['UserId'][5])


# In[21]:


r.get_similar_songs(['The Cove by Jack Johnson'])

