
# coding: utf-8

# In[186]:

# KANNAN
# CREDIT CARD DEFAULT PREDICTION
# MODEL  : LOGISTIC REGRESSION
# DATA   : 12/04/2017 - PYTHON FINAL PRESENTATION

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_score, cross_validate
from matplotlib import pyplot


# In[236]:

#importing the data and shuffuling 

df_full = pd.read_excel('credit.xls')

df = df_full

df=shuffle(df)

df.head()


# In[243]:

#Scaling the data


x1 = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x1)
df = pd.DataFrame(x_scaled)


# In[248]:

y = df.iloc[:,24]

#X = df.iloc[:,1:24]
X = df.iloc[:,[1,6,7,8,9,10,11,17,18,19,20,21,22]]
#X = df.iloc[:,[1,6,7,8,9,17,18,19,20,21,22]]

#X = df.iloc[:,[11,5,0,17,4,18,19,20,16,14,10]]

#X = df.drop(df.columns[[23]], axis=1)

X.head()


# In[257]:

#Test Train split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)


# In[258]:

LogReg = LogisticRegression()
LogReg.fit(X_train,y_train)



# In[259]:

import matplotlib.pyplot as plt
feature_importance = abs(LogReg.coef_[0])
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

featfig = plt.figure()
featax = featfig.add_subplot(1, 1, 1)
featax.barh(pos,feature_importance[sorted_idx],  align='center')
featax.set_yticks(pos)
featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=8)
featax.set_xlabel('Relative Feature Importance')

plt.title('Feature Importance')
plt.tight_layout()   
plt.show()


# In[260]:

y_pred = LogReg.predict(X_test)


# In[77]:

pro = LogReg.predict_proba(X_test)


# In[88]:

y_pred_prob = pd.DataFrame(pro[:,1]).head(20)


# In[116]:

pro = LogReg.predict_proba(X_test)
y_pred_prob = pd.DataFrame(pro[:,1])
y_pred_prob[y_pred_prob > 0.3] = 1
y_pred_prob[y_pred_prob < 0.3] = 0
y_test = np.asarray(y_test)
y_test = y_test.reshape(len(y_test),)
y_pred_prob = np.asarray(y_pred_prob)
y_pred_prob = y_pred_prob.reshape(len(y_pred_prob),)
y_pred_prob


# In[192]:

accuracy_score(y_test,y_pred_prob)


# In[251]:

print(classification_report(y_test,y_pred_prob))


# In[220]:

y_pred


# In[261]:

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

