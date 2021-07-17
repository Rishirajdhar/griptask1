#!/usr/bin/env python
# coding: utf-8

# 
# # Task 1- Prediction using Supervised ML
# 
# ### Task: Predict the percentage of a student based on the no. of study hours.

# ## The Sparks Foundation(GRIP), July 2021

# ####  By: Rishi Raj Dhar

# In[11]:


#importing the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


#Reading the data

url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"


# In[13]:


data=pd.read_csv(url)


# In[14]:


print(data)


# In[15]:


#See the first 5 rows of the data
data.head(5)


# In[16]:


#See the last 5 rows of the data
data.tail(5)


# In[17]:


data.shape


# In[18]:


data.info()


# In[19]:


data.describe()


# In[20]:


#Check for the null values if any.
data.isnull().sum()


# ### As there is no null values, we can now visualize our data.

# In[21]:


# Plotting the distribution of scores

sns.scatterplot(y=data['Scores'], x=data['Hours'])
plt.title('Marks vs Study hours', size=18)
plt.ylabel('Marks Percentage', size=15)
plt.xlabel('Hours Studied', size=15)
plt.show()


# ####  From the above scatterplot, we can clearly see that there is a positive linear relation between the "Number of hours studied" and "Percentage of score".                                                                                                                                                                                Now plotting a regression line to confirm the correlation.
# 

# In[22]:


#plotting the regression line
sns.regplot(x=data['Hours'],y=data['Scores'])
plt.title('Regression Plot', size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()

#Correlation
print(data.corr())


# ### From the above output it is confirmed that the variables are postively correlated.

# # Preparing the data

# # The next step is to divide the data into "attributes"(inputs) and "labels"(outputs)

# In[23]:


#x- attributes, y- labels

x= data.iloc[:,:-1].values
y= data.iloc[:, 1].values


# ### Doing this by using Scikit-Learn's built-in train_test_split() method.

# In[24]:


#Splitting the data(Training & Test datasets)   

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)


# In[25]:


#We have split the dataset as 75% training data and 25% test data.


# #### Training the model

# ##### We will be using the Linear Regression which is a supervised machine learning algortithm

# In[26]:


from sklearn.linear_model import LinearRegression

lr= LinearRegression()
lr.fit(x_train, y_train)

print("Training complete.")


# # Making Predictions

# In[27]:


# Predicting the scores

y_pred=lr.predict(x_test)
y_pred


# In[28]:


df=pd.DataFrame({'Hours': [i[0] for i in x_test], 'Predicted Marks' : [k for k in y_pred]})
df


# In[29]:


# Comparing the Actual marks and the predicted marks

compare_scores = pd.DataFrame({'Actual Marks': y_test, 'Predicted Marks': y_pred})
compare_scores


# In[30]:


plt.scatter(x=x_test, y=y_test, color='blue')
plt.plot(x_test, y_pred, color='Black')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Actual Marks', size=15)
plt.xlabel('Predicted Marks', size=15)
plt.show()


# # Evaluating the model

# In[31]:


from sklearn import metrics as m

print('Accuracy of Actual and Predicted Scores R-Squared is:', m.r2_score(y_test,y_pred))

MSE= m.mean_squared_error(y_test, y_pred)
RMSE= np.sqrt(MSE)
MAE= m.mean_absolute_error(y_test,y_pred)

print('Mean Squared Error:', MSE)
print('Root Mean Squared Error:', RMSE)
print('Mean Absolute Error:', MAE)


# In[32]:


hours = [9.5]
answer = lr.predict([hours])
print('Score: {}'.format(round(answer[0],3)))


# ###### The accuracy is around 94% and the small value of error metrics indicates that the chances of error or wrong forecasting through the model are very less.

# ##           ................................. END OF TASK 1..................................................

# In[ ]:




