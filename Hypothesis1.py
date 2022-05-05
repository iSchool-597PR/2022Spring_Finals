#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


# loading datasets
coal = pd.read_csv('data/coal-consumption-by-country-terawatt-hours-twh.csv')
gas = pd.read_csv('data/gas-consumption-by-country.csv')
fossil = pd.read_csv('data/fossil-fuel-primary-energy.csv')
oil = pd.read_csv('data/oil-consumption-by-country.csv')
wind = pd.read_csv('data/primary-energy-wind.csv')
solar = pd.read_csv('data/primary-energy-consumption-from-solar.csv')
hydro = pd.read_csv('data/primary-energy-hydro.csv')
renewable = pd.read_csv('data/primary-energy-renewables.csv')


# In[3]:


def generate_key(data):
    """
    This function returns a unique key by merging country and the year.  
    """
    data['keys'] = data['Entity'] + data['Year'].astype('str')


# In[4]:


generate_key(coal)
generate_key(gas)
generate_key(fossil)
generate_key(oil)
generate_key(wind)
generate_key(solar)
generate_key(hydro)
generate_key(renewable)


# In[5]:


# merging all the datasets based on their keys
from functools import partial, reduce

dfs = [coal[['keys', 'Coal Consumption - TWh']], gas[['keys','Gas Consumption - TWh']]
       , fossil, oil[['Oil Consumption - TWh', 'keys']], wind[['keys', 'Wind (TWh – sub method)']], 
       solar[['keys', 'Solar (TWh – sub method)']], hydro[['keys','Hydro (TWh – sub method)']], 
       renewable[['keys', 'Renewables (TWh – sub method)']]]

merged = partial(pd.merge, on = 'keys', how = 'outer')
merge_df = reduce(merged, dfs)


# In[6]:


merge_df.isnull().sum()


# In[7]:


# for visualization purpose filling NAs with Missing in Entity attribute
merge_df['Entity'].fillna('Missing', inplace = True)


# In[8]:


sns.heatmap(merge_df.isnull())


# In[9]:


# find correlation between attributes
cor = merge_df.corr()
cor = cor[cor>0.5]
sns.heatmap(cor, annot=True)


# In[10]:


# visualizing correlatoin 
sns.pairplot(merge_df, kind = 'reg')


# In[11]:


# understanding trends of energy consumption
numer = [x for x in merge_df.columns if (merge_df[x].dtypes != 'O') and (x != 'Year')]
for i in numer:
    merge_df.groupby('Year')[i].median().plot()
    plt.xlabel('Year')
    plt.ylabel(i)
    plt.show()


# In[12]:


merge_df.dtypes


# In[13]:


import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


# In[14]:


# converting year into datetime format
import datetime as dt

merge_df['Year'] = pd.to_datetime(merge_df['Year'], format = '%Y').dt.year


# In[15]:


def interactiveline_plot(data, xaxis, yaxis, color):
    """
    This function will return an interactive line plot 
    """
    return px.line(data, x = xaxis, y = yaxis, color= color)

interactiveline_plot(merge_df, merge_df['Year'], merge_df['Coal Consumption - TWh'], merge_df['Entity'])


# In[16]:


interactiveline_plot(merge_df, merge_df['Year'], merge_df['Gas Consumption - TWh'], merge_df['Entity'])


# In[17]:


interactiveline_plot(merge_df, merge_df['Year'], merge_df['Fossil Fuels (TWh)'], merge_df['Entity'])


# In[18]:


interactiveline_plot(merge_df, merge_df['Year'], merge_df['Hydro (TWh – sub method)'], merge_df['Entity'])


# In[19]:


interactiveline_plot(merge_df, merge_df['Year'], merge_df['Oil Consumption - TWh'], merge_df['Entity'])


# In[20]:


interactiveline_plot(merge_df, merge_df['Year'], merge_df['Renewables (TWh – sub method)'], merge_df['Entity'])


# In[21]:


interactiveline_plot(merge_df, merge_df['Year'], merge_df['Solar (TWh – sub method)'], merge_df['Entity'])


# In[22]:


interactiveline_plot(merge_df, merge_df['Year'], merge_df['Wind (TWh – sub method)'], merge_df['Entity'])


# In[23]:


def growth(data, columns):
    """
    This function will create new column that contains growth for the respective years
    """
    data[columns+'growth'] = data[columns].diff()

for i in numer:
    growth(merge_df, i)


# In[24]:


merge_df


# In[25]:


merge_df.describe()


# In[26]:


interactiveline_plot(merge_df, merge_df['Year'], merge_df['Coal Consumption - TWhgrowth'], merge_df['Entity'])


# In[27]:


interactiveline_plot(merge_df, merge_df['Year'], merge_df['Fossil Fuels (TWh)growth'], merge_df['Entity'])


# In[28]:


interactiveline_plot(merge_df, merge_df['Year'], merge_df['Gas Consumption - TWhgrowth'], merge_df['Entity'])


# In[29]:


interactiveline_plot(merge_df, merge_df['Year'], merge_df['Hydro (TWh – sub method)growth'], merge_df['Entity'])


# In[30]:


interactiveline_plot(merge_df, merge_df['Year'], merge_df['Oil Consumption - TWhgrowth'], merge_df['Entity'])


# In[31]:


interactiveline_plot(merge_df, merge_df['Year'], merge_df['Renewables (TWh – sub method)growth'], merge_df['Entity'])


# In[32]:


interactiveline_plot(merge_df, merge_df['Year'], merge_df['Solar (TWh – sub method)growth'], merge_df['Entity'])


# In[33]:


interactiveline_plot(merge_df, merge_df['Year'], merge_df['Wind (TWh – sub method)growth'], merge_df['Entity'])


# In[34]:


recent = merge_df[merge_df['Year'] > 2014]


# In[35]:


recent.head()


# Trend Calculations

# In[36]:


col = ['Oil Consumption - TWh', 'Gas Consumption - TWh', 'Fossil Fuels (TWh)', 'Oil Consumption - TWh',
       'Wind (TWh – sub method)', 'Solar (TWh – sub method)','Hydro (TWh – sub method)', 
       'Renewables (TWh – sub method)']


# In[37]:


# creating a dataframe that contains data from mean of the energies consumed yearly
x= recent.groupby('Year')['Coal Consumption - TWh'].mean().reset_index()
x['oil'] = recent.groupby('Year')['Oil Consumption - TWh'].mean().reset_index().iloc[:,1]
x['gas'] = recent.groupby('Year')['Gas Consumption - TWh'].mean().reset_index().iloc[:,1]
x['fossil'] = recent.groupby('Year')['Fossil Fuels (TWh)'].mean().reset_index().iloc[:,1]
x['wind'] = recent.groupby('Year')['Wind (TWh – sub method)'].mean().reset_index().iloc[:,1]
x['solar'] = recent.groupby('Year')['Solar (TWh – sub method)'].mean().reset_index().iloc[:,1]
x['hydro'] = recent.groupby('Year')['Hydro (TWh – sub method)'].mean().reset_index().iloc[:,1]
x['renewable'] = recent.groupby('Year')['Renewables (TWh – sub method)'].mean().reset_index().iloc[:,1]


# In[38]:


x


# In[39]:


# finding slope
(1972.938159- 2072.567026)/5


# Finding Slope will not help us understand the trend since it speaks only about the 1st and the last year. You will not understand what other points are trying to say.

# In[40]:


def trend(array, order = 1):
    """
    This function returns a float value that indicates the trend of your data
    Referred from: https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
    """
    return np.polyfit(x['Year'], array, order)[-2]

for i in x.iloc[:, 1:].columns:
    plt.bar(i, trend(x[i]))
    plt.xlabel("Energy Sources")
    plt.ylabel("Trend")
    plt.plot()
    plt.xticks(rotation = 'vertical')


# In[41]:


# creating a dataframe that contains data from 2018 onwards
recent1 = merge_df[merge_df['Year'] > 2017]

y = pd.DataFrame()
y['coal'] = recent.groupby('Year')['Coal Consumption - TWh'].mean().reset_index().iloc[:,1]
y['oil'] = recent.groupby('Year')['Oil Consumption - TWh'].mean().reset_index().iloc[:,1]
y['gas'] = recent.groupby('Year')['Gas Consumption - TWh'].mean().reset_index().iloc[:,1]
y['fossil'] = recent.groupby('Year')['Fossil Fuels (TWh)'].mean().reset_index().iloc[:,1]
y['wind'] = recent.groupby('Year')['Wind (TWh – sub method)'].mean().reset_index().iloc[:,1]
y['solar'] = recent.groupby('Year')['Solar (TWh – sub method)'].mean().reset_index().iloc[:,1]
y['hydro'] = recent.groupby('Year')['Hydro (TWh – sub method)'].mean().reset_index().iloc[:,1]
y['renewable'] = recent.groupby('Year')['Renewables (TWh – sub method)'].mean().reset_index().iloc[:,1]


# In[42]:


# resulting dataframe
y


# In[43]:


# plotting to find correlaton between attributes
sns.pairplot(y, kind = 'reg')


# Earlier it was observed that all elements had a postive correlation due to the extent of posterior data, where the values was continuously increasing since 1965. 
# But when we consider the data from 2018, we find that only Gas has a positive correlation with renewable energies while other non-renewable energies have a negative correlation. This is a good sign indeed. But this rejects our hypothesis.
# Even though the dependecy on Gas has increased in the recent terms, the next case study tells us why it will be difficult to completely shift to renewable resources.

# In[ ]:




