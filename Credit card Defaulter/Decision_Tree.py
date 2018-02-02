
# coding: utf-8

# KANNAN
# CREDIT CARD DEFAULT PREDICTION
# MODEL  : LOGISTIC REGRESSION
# DATA   : 12/04/2017 - PYTHON FINAL PRESENTATION

# In[44]:

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# In[45]:

df= pd.read_excel('credit.xlsx')


# In[46]:

df['RUN_PAID'] = (df['PAY_AMT1']+df['PAY_AMT2']+df['PAY_AMT3']+df['PAY_AMT4']+df['PAY_AMT5']+df['PAY_AMT6'])/(df['BILL_AMT1']+df['BILL_AMT2']+df['BILL_AMT3']+df['BILL_AMT4']+df['BILL_AMT5']+df['BILL_AMT6']+0.0000001)


# In[47]:

df.head()


# In[48]:

cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]


# In[49]:

df.head()


# In[50]:

df['PAY_AMT1'] = df['PAY_AMT1']/(df['BILL_AMT1']+0.000000000001)
df['PAY_AMT2'] = df['PAY_AMT2']/(df['BILL_AMT2']+0.000000000001)
df['PAY_AMT3'] = df['PAY_AMT3']/(df['BILL_AMT3']+0.000000000001)
df['PAY_AMT4'] = df['PAY_AMT4']/(df['BILL_AMT4']+0.000000000001)
df['PAY_AMT5'] = df['PAY_AMT5']/(df['BILL_AMT5']+0.000000000001)
df['PAY_AMT6'] = df['PAY_AMT6']/(df['BILL_AMT6']+0.000000000001)


# In[51]:

df['BILL_AMT1'] = df['BILL_AMT1']/(df['LIMIT_BAL']+0.000000000001)
df['BILL_AMT2'] = df['BILL_AMT2']/(df['LIMIT_BAL']+0.000000000001)
df['BILL_AMT3'] = df['BILL_AMT3']/(df['LIMIT_BAL']+0.000000000001)
df['BILL_AMT4'] = df['BILL_AMT4']/(df['LIMIT_BAL']+0.000000000001)
df['BILL_AMT5'] = df['BILL_AMT5']/(df['LIMIT_BAL']+0.000000000001)
df['BILL_AMT6'] = df['BILL_AMT6']/(df['LIMIT_BAL']+0.000000000001)


# In[52]:

for i in df.select_dtypes(include = ['float64', 'float32', 'int64']):
        df[i].fillna(value = 0, inplace = True)


# In[53]:

df.drop(['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'], axis=1, inplace=True)


# In[54]:

df.head()


# In[55]:

X = df.values[:,0:-2]

Y = df.values[:,-1]


# In[56]:

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


# In[57]:

Y_train


# In[58]:

dt_model1 = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=27, min_samples_leaf=5)
dt_model1.fit(X_train, Y_train)
Y_pred1 = dt_model1.predict(X_test)


# In[59]:

print "Accuracy is ", accuracy_score(Y_test,Y_pred1)*100
print "Confusion Matrix"
print pd.crosstab(Y_test, Y_pred1, rownames = ['Act'], colnames= ['Pred'])
print classification_report(Y_test, Y_pred1)
print "Cross Validation"
cross_val_score(dt_model1, X,Y, cv=5)


# In[60]:

dt_model2 = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=27, min_samples_leaf=5)
dt_model2.fit(X_train, Y_train)
Y_pred2 = dt_model2.predict(X_test)


# In[61]:

X_test.shape


# In[62]:

print "Accuracy is ", accuracy_score(Y_test,Y_pred2)*100
print "Confusion Matrix"
print pd.crosstab(Y_test, Y_pred2, rownames = ['Act'], colnames= ['Pred'])
print classification_report(Y_test, Y_pred2)
print "Cross Validation"
cross_val_score(dt_model2, X,Y, cv=5)


# In[154]:




# In[ ]:



