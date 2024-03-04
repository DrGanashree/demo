#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import libraries
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf


# In[3]:


# import dataset
ndata=pd.read_csv('NewspaperData.csv')
ndata


# In[4]:


ndata.info()


# In[5]:


sns.distplot(ndata['daily'])


# In[6]:


# Visualization of Correlation beteen x and y
sns.regplot(x=ndata['daily'],y=ndata['sunday'])  # regplot = regression plot


# In[7]:


# Fitting a Linear Regression Model
# model = smf.ols("y~x",data=defined_data).fit()   ; ols= ordinary least square methodreg.predict([[3300]])


# In[8]:


model=smf.ols("sunday~daily",data=ndata).fit() 


# In[9]:


# as Y = Beta0 + Beta1*(X)


# In[10]:


# Finding Coefficient Parameters (Beta0 and Beta1 values)
model.params 


# In[11]:


# Here, (Intercept) Beta0 value = 13.8356 & (daily) Beta1 value = 1.3397
# Hypothesis testing of X variable by finding t_values and P_values for Beta1 i.e if (P_value < α=0.05 ; Reject Null)
# Null Hypothesis as Beta1=0 (No Slope) and Alternate Hypthesis as Beta1≠0 (Some or significant Slope)


# In[12]:


print(model.tvalues,'\n',model.pvalues)


# In[13]:


# (Intercept) Beta0: tvalue=0.386427 , pvalue=7.017382e-01
# (daily)     Beta1: tvalue=18.934840, pvalue= 6.016802e-19
# As (pvalue=0)<(α=0.05); Reject Null hyp. Thus, X(daily) variable has good slope and variance w.r.t Y(sunday) variable


# In[14]:


# R-squared measures the strength of the relationship between your model and the dependent variable on a 0 – 100% scale.


# In[15]:


# Measure goodness-of-fit by finding rsquared values (percentage of variance)
model.rsquared,model.rsquared_adj 


# In[16]:


#Automatic prediction for say 200 and 300 daily circulations
new_data=pd.Series([200,300])
new_data


# In[17]:


data_pred=pd.DataFrame(new_data,columns=['daily'])
data_pred


# In[19]:


model.predict(data_pred)


# In[20]:


model.predict(data_pred)


# In[21]:


model.predict(data_pred)


# In[ ]:





# In[ ]:





# In[ ]:




