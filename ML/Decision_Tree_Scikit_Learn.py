
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[2]:


#load data
col_names = ['Age','Income','Gender','Marital Status','Buys']
buy_data = pd.read_csv("C:\\Users\\Amruta K\\Downloads\\transaction - Sheet1 (2).csv",header=None,names=col_names)
buy_data.head(10)


# In[3]:


#we need to convert above data in numeric format in order to use scikit-learn classifier

cols = list(buy_data)
buy_data = buy_data.drop(0,axis=0).reset_index(drop=True)
col_names = list(buy_data)
for col in col_names:
    column = buy_data[col]
    unique_vals = list(set(column))
    for j in range(len(column)):
        buy_data[col][j] = int(unique_vals.index(buy_data[col][j]))
     
buy_data.head()


# In[4]:


feature_cols = ['Age','Income','Gender','Marital Status']
X = buy_data[feature_cols]
y = buy_data.Buys
y=y.astype('int')


# In[5]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)


# In[6]:


clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)


# In[7]:


y_pred = clf.predict(X_test)


# In[8]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

