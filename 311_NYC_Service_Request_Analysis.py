#!/usr/bin/env python
# coding: utf-8

# ## NYC Service Request Analysis
# ### Laxman Parab

# In[1]:


#Importing required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


#importing the dataset
data = pd.read_csv(r'311_Service_Requests_from_2010_to_Present.csv')


# In[4]:


data.head()


# In[5]:


#Looking at each column's name and datatype
data.info()


# In[6]:


#Convert 'Created Date' and 'Closed Date' to datetime.
data['Closed Date'] = pd.to_datetime(data['Closed Date'])
data['Created Date'] = pd.to_datetime(data['Created Date'])
data['Request_Closing_Time'] = data['Closed Date'] - data['Created Date']


# In[7]:


#Number of rows and columns 
data.shape


# In[8]:


#Checking for missing values
data.isnull().sum()


# In[9]:


#Deleting the columns with almost all missing values.
data = data.drop(["School or Citywide Complaint","Vehicle Type","Taxi Company Borough","Taxi Pick Up Location","Garage Lot Name","Ferry Direction","Ferry Terminal Name"],axis = 1)


# In[10]:


#Filling the City column with 'Unknown City' instead of dropping the records with missing Cities
data['City'].fillna('Unknown City', inplace =True)


# In[11]:


data.shape


# In[12]:


#Descriptor is the compliant type label.
data['Descriptor'].unique()


# In[13]:


#What are the types of complaint made?
data['Complaint Type'].unique()


# In[14]:


#How many types of complaints are made?
data['Complaint Type'].nunique()


# In[15]:



complaint = data['Complaint Type'].value_counts()
complaint_df = complaint.to_frame()
complaint_df = complaint_df.rename(columns = {'Complaint Type':'Counts'})
complaint_df


# In[16]:


#Which type of complaint is maximum?
plt.figure(figsize = (15,10))
sns.countplot(y ='Complaint Type',data = data,order = data['Complaint Type'].value_counts().iloc[:10].index) 


# In[17]:


#We can see that most of the compliant types are of 'Blocked Driveway' followed by 'Illegal Parking' and so on.


# In[18]:


#What is the satatus of the tickets?
plt.figure(figsize = (7,7))
data['Status'].value_counts().plot(kind ='bar')


# In[19]:


#Most of the cases are closed which is a good sign for NYC 311


# In[20]:


plt.figure(figsize = (7,7))
sns.countplot(y ='Location Type',data = data,order = data['Location Type'].value_counts().iloc[:5].index) 


# In[21]:


#Most of the complaints are from Streets/Sidewalks.


# In[22]:


plt.figure(figsize = (7,7))
sns.countplot(y ='Descriptor',data = data,order = data['Descriptor'].value_counts().iloc[:5].index) 


# In[23]:


#Loud music/Party are one of the biggest problems for the people.


# In[24]:


plt.figure(figsize = (7,7))
sns.countplot(y ='City',data = data,order = data['City'].value_counts().iloc[:5].index)


# In[25]:


#Brooklyn has the highest number of callers i.e. they have many problems.


# In[26]:


#Creating a new colmun of 'Request_Closing _Time' in hours
data_2 = data[['City','Complaint Type','Request_Closing_Time']]
data_2.dropna(subset = ['City','Complaint Type','Request_Closing_Time'], inplace = True)
data_2['rct_hr'] = np.around((data_2['Request_Closing_Time'].astype(np.int64)/(pow(10,9)*3600)),decimals=2)


# In[27]:


data_2.head()


# In[28]:


avg_time = np.around((data_2['rct_hr'].mean()),decimals=2)
avg_time


# In[29]:


#Ordering the complaint types based on the average ‘Request_Closing_Time’
complaint_group = data_2.groupby(['City','Complaint Type']).agg({'rct_hr':'mean'})
complaint_group = complaint_group.sort_values(['City','rct_hr'])


# In[30]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
complaint_group


# In[31]:


import scipy.stats as stat


# In[32]:


#Are the type of complaint or service requested and location related?


# In[33]:


data_3 = data[['Complaint Type','City']]
data_3 = data_3.dropna()


# In[34]:


citycomp = pd.crosstab(data_3['Complaint Type'], data_3['City'])
citycomp.head()


# In[35]:


#Applying ANOVA for some combinations.


# In[36]:


f_val,p_val = stat.f_oneway(citycomp['ARVERNE'],citycomp['ASTORIA'])


# In[37]:


print('F-statistic is ',round(f_val,2))
print('p-value is ',round(p_val,2))


# In[38]:


f_val,p_val = stat.f_oneway(citycomp['ARVERNE'],citycomp['BROOKLYN'])


# In[39]:


print('F-statistic is ',round(f_val,2))
print('p-value is ',round(p_val,2))


# In[40]:


#p-value is around 0.05 lets conduct chi-square contingency test


# In[41]:


#Null hypothesis: There is no relationship between the complaint and city
#v/s Alternate hypothesis: There is relationship between complaint and city


# In[42]:


chi2, p_val, df, exp_frq = stat.chi2_contingency(citycomp)


# In[43]:


chi2


# In[44]:


p_val


# In[45]:


#Since p-value is less than 0.05 we reject null hypothesis and conclude
#that complaint or service request and location are related

