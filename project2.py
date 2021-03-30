#!/usr/bin/env python
# coding: utf-8

# Importing Libraries
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import Dataset

# In[2]:


#Import Dataset
ds= pd.read_csv("dataset_website.csv")
ds.head()


# In[3]:


#Analysing the data using pandas and Checking if the dataset contains any Null values.
ds.info()
ds.isnull().any() #no nullvalues


# In[ ]:





# Taking care of missing data

# In[4]:


ds.isnull().any()   #no missing data in the dataset


# No need of applying label/one hot encoding as there are no categorical columns in the dataset.

# Feature scaling is not required as data is already present in the same range.

# Splitting data as independent and dependent

# In[5]:


#Splitting data as independent and dependent
#removing index column in independent dataset
x=ds.iloc[:,1:31].values
y=ds.iloc[:,-1].values
print(x,y)


# In[6]:


y=ds.iloc[:,-1].values
y


# Splitting data into train and test 

# In[7]:


#Splitting data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# MODEL BUILDING -

# Since the output (result) is  categorical , it comes under classification model.

# Training and Testing the model -

# 1. LOGISTIC REGRESSION -

# In[8]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[9]:


lr.fit(x_train,y_train)


# In[10]:


y_pred1=lr.predict(x_test)


# In[11]:


y_pred1=lr.predict(x_test)
from sklearn.metrics import accuracy_score
log_reg=accuracy_score(y_test,y_pred1)
log_reg


# 2. KNN-euclidian distance

# In[12]:


from sklearn.neighbors import KNeighborsClassifier
kn1=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)  


# In[13]:


kn1.fit(x_train,y_train)


# In[14]:


y_pred2=kn1.predict(x_test)
from sklearn.metrics import accuracy_score
knn_euc=accuracy_score(y_test,y_pred2)
knn_euc


# 3. KNN-manhattan distance

# In[15]:


kn2=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=1)
kn2.fit(x_train,y_train) 


# In[16]:


y_pred3=kn2.predict(x_test)
from sklearn.metrics import accuracy_score
knn_man=accuracy_score(y_test,y_pred3)
knn_man


# 4. SVM -rbf

# In[17]:


from sklearn.svm import SVC
svm1=SVC(kernel='rbf')
svm1.fit(x_train,y_train) 


# In[18]:


y_pred4=svm1.predict(x_test)
from sklearn.metrics import accuracy_score
svm_rbf=accuracy_score(y_test,y_pred4)
svm_rbf


# 5. SVM -sigmoid

# In[19]:


svm2=SVC(kernel='sigmoid')
svm2.fit(x_train,y_train) 


# In[20]:


y_pred5=svm2.predict(x_test)
from sklearn.metrics import accuracy_score
svm_sig=accuracy_score(y_test,y_pred5)
svm_sig


# 6. DECISION TREE

# In[21]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)


# In[22]:


y_pred6=dt.predict(x_test)
from sklearn.metrics import accuracy_score
dec_tree=accuracy_score(y_test,y_pred6)
dec_tree


# 7. RANDOM FOREST

# In[23]:


from sklearn.ensemble import RandomForestRegressor
Rf=RandomForestRegressor(n_estimators=10,random_state=0,n_jobs=-1)     
Rf.fit(x_train,y_train)


# In[24]:


y_pred7=Rf.predict(x_test)
from sklearn.metrics import accuracy_score
rf=accuracy_score(y_test,y_pred7.round())
rf


# 8. NAIVE BAYES

# In[25]:


from sklearn.naive_bayes import GaussianNB
gb=GaussianNB()
gb.fit(x_train,y_train)


# In[26]:


y_pred8=gb.predict(x_test)
from sklearn.metrics import accuracy_score
nb=accuracy_score(y_test,y_pred1)
nb


# Model Comparision

# In[27]:


models = pd.DataFrame({
    'Model': [ 'Logistic Regression', 'KNN-eiclidian distance','KNN-manhattan distance','Naive Bayes','SVM-rbf','SVM-sigmoid','Decision Tree','Random Forest'],
    'Test Score': [ log_reg,knn_euc,knn_man,nb,svm_rbf,svm_sig,dec_tree,rf]})
models.sort_values(by='Test Score', ascending=False)


# To avoid overfitting , the model which is considered here is LOGISTIC REGRESSION

# ACCURACY OF DECISION TREE MODEL IS HIGHEST 

# In[28]:


import pickle
pickle.dump(lr,open('Phishing_Website.pkl','wb'))


# In[ ]:




