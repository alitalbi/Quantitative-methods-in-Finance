#!/usr/bin/env python
# coding: utf-8

#  #Getting time series from Banque de France and INSEE 
# 

# In[136]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 19:35:37 2021

@author: talbi
"""

import numpy as np
import pandas as pd 
import os
#P.34

#Question3
os.chdir('/Users/talbi/Downloads')
data= pd.read_csv('Webstat_Export_20210908.csv',sep = ';', encoding = 'latin1', skiprows = [0,1,2,3,4])

print(data)

#We replaced ',' by '.'
for i in range(0,len(data)):
    data.iloc[i,1]= data.iloc[i,1].replace(',','.')
float=np.vectorize(float)    
#Now we can transform the data into float 
data.iloc[:,1] = float(data.iloc[:,1])


data=data[::-1]
data.index=data['Source :']
data

data.iloc[:,1].plot(title='French proprety prices')
data.drop('Source :',axis=1,inplace=True)

ax=data.iloc[:,0].plot(title='French proprety prices')
ax.vlines(x=93,ymin=0,ymax=120)

data['Var'] = data.iloc[:,0]/data.iloc[:,0].shift(1) - 1

#ax2=data['Var'].plot()
#ax2.vlines(x=93,ymin=-0.05,ymax=0.06)

data_new=pd.DataFrame(data['Banque de France (FR2)'])
data_new.index = pd.date_range('1996','2022',freq='Q')[:-3]
print(data_new)
import plotly.express as px
fig = px.line(data_new.iloc[:,0])
fig.update_xaxes(rangeslider_visible = True)
fig.show()


# For the French proprety prices, we remark an up trend since 1996. 
# 
# In June, 2009 the prices decreased to bounce back at the previous level in Sep. 2011

# In[137]:


data.index= data_new.index
data['Var'].plot(title='Variation of the French proprety prices')


# In[138]:


from statsmodels.tsa.seasonal import seasonal_decompose
result2 = seasonal_decompose(data.iloc[1:,1],model='additive')
result2.plot()


# We can see that the proprety prices increased before the 20's. Up by 0.05% in 2004/5, the prices dropped after the 2008 krash. It is regaining some strength and coming back to the level of prices of 2010/11.
# 
# The trend was already up before the pandemic. It didn't have a significant impact on the French proprety Prices.

# In[139]:


data.hist()


# In[140]:


#importing the data
valeurs_mensuelles = pd.read_csv('valeurs_mensuelles.csv',encoding='latin1',sep=';',skiprows=[0,1,2])

#modifying the index as dates
valeurs_mensuelles.index=pd.date_range('1975','2022',freq='M')[:-4]

#Droping useless columns
valeurs_mensuelles = valeurs_mensuelles.drop(['PÃ©riode','Unnamed: 2'],axis=1)

#Rename the first column
valeurs_mensuelles.columns = ['French Population']


# In[141]:


valeurs_mensuelles


# In[142]:


fig2 = px.line(valeurs_mensuelles.iloc[:,0])
fig2.update_xaxes(rangeslider_visible = True)
fig2.show()


# We see that the french population is in a down trend since the 70's. 
# 
# By using the slider, we can select the period from 2019 when the the pandemi start and see how 'flat/slightly decreasing' is the French Population 

# In[143]:


valeurs_mensuelles['var'] = valeurs_mensuelles/valeurs_mensuelles.shift(1)
fig3 = px.line(valeurs_mensuelles.iloc[:,1])
fig3.update_xaxes(rangeslider_visible = True)
fig3.show()


# The trend stayed the a bit flat since the 20's, we can check that by using seasonal_decompose. 
# The result object provides access to the trend and seasonal series as arrays.

# In[144]:



result = seasonal_decompose((valeurs_mensuelles['var'][1:]),model='additive')
result.plot()


# Indeed since Oct 1997, with a rise in the population, the variation didn't increase drastically and kept the same trend. 

# In[145]:


valeurs_mensuelles['var'].hist()


# We confirmed that the variation since 2019 is not an outlier and we don't see a pandemic effect
