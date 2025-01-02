#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#load the data
df_store=pd.read_csv(r'C:\Users\pc\Desktop\project 6\store.csv')
df_train=pd.read_csv(r'C:\Users\pc\Desktop\project 6\train.csv')


# In[79]:


# Merge the data on basic of one common colomn 
df=pd.merge(df_store,df_train, on='Store', how='inner')


# In[80]:


#Check the data 
df.head(3)


# In[81]:


#Count of row and colomns
df.shape


# In[82]:


#Generar informtion of data (mean, max,count, percentile)
df.describe()


# In[83]:


#General inf
df.info()


# In[84]:


# % value of null values in each columnss
df.isnull().sum()*100/(df.shape[0])


# In[91]:


# graphical representaion of null values 
sns.heatmap(data=df.isnull())
plt.show


# In[92]:


# Name of columns 
df.columns


# In[93]:


# filling of null values in object type columns 
for col in df.columns:
    if df[col].dtype=='object':
        df[col].fillna(df[col].mode()[0],inplace=True)


# In[94]:


# filling of null values in numerical  type columnns
for col in df.columns:
    if df[col].dtype=='int64':
        df[col].fillna(df[col].mean(),inplace=True)


# In[96]:


# filling of null values in numerical  type columnns
for col in df.columns:
    if df[col].dtype=='flot64':
        df[col].fillna(df[col].mean(),inplace=True)


# In[97]:


# filling of null values in numerical  type columnns
for i in df.select_dtypes(include='float64').columns:
    df[i].fillna(df[i].mode()[0],inplace=True)


# In[98]:


df.isnull().sum()


# In[99]:


# Graphical representaion after filling null values 
sns.heatmap(data=df.isnull())
plt.show


# In[100]:


# Catagorical and numerical columns 
catagorial_col=df.select_dtypes(include=['object'])
numerica_col=df.select_dtypes(include=['float','int'])
catagorial_col


# In[101]:


#Encoding boject type data into numrica 
for col in df.columns:
    if df[col].dtype=='object':
        df[col]=df[col].astype(str)


# In[102]:


la=LabelEncoder()
for col in df.columns:
    if df[col].dtype=='object':
        df[col]=la.fit_transform(df[col])


# In[103]:


df.head(3)


# In[104]:


df.columns


# In[105]:


# relation betwween sales and promotion 
plt.figure(figsize=(4,4))
plt.title('Reation between Sales and Promootion ')
sns.barplot(x='Promo', y='Sales', data=df)
plt.xlabel('Promotion')
plt.ylabel('Sales')
plt.show()


# In[106]:


# relation betwween sales and store type
plt.figure(figsize=(4,4))
plt.title('Reation between Sales and StoreType ')
sns.barplot(x='StoreType', y='Sales', data=df)
plt.xlabel('StoreType')
plt.ylabel('Sales')
plt.show()


# In[107]:


# relation betwween sales and assortment
plt.figure(figsize=(4,4))
plt.title('Reation between Sales and Assortment ')
sns.barplot(x='Assortment', y='Sales', data=df)
plt.xlabel('Assortment')
plt.ylabel('Sales')
plt.show()


# In[108]:


# relation betwween sales and customer
plt.figure(figsize=(10,4))
plt.title('Reation between Sales and Customers')
sns.lineplot(x='Customers', y='Sales', data=df)
plt.xlabel('Customers')
plt.ylabel('Sales')
plt.show()


# In[50]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[109]:


X=df.drop(['Sales'],axis=1)
y=df['Sales']


# In[110]:


# Split the dataset into training and testing sets
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)


# In[111]:


rf_regressor=RandomForestRegressor(n_estimators=10, random_state=0,oob_score=True)


# In[112]:


rf_regressor.fit(X,y)


# In[113]:


#Making predictions on the same data or new data
predictions = rf_regressor.predict(X)


# In[114]:


# Evaluating the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y, predictions)
print(f'Mean Squared Error: {mse}')


# In[115]:


r2 = r2_score(y, predictions)
print(f'R-squared: {r2}')


# In[ ]:




