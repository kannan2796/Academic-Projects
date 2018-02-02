
# coding: utf-8

# In[316]:

# KANNAN
# CREDIT CARD DEFAULT PREDICTION
# MODEL  : NEURAL NETWORK USING KERAS
# DATA   : 12/04/2017 - PYTHON FINAL PRESENTATION


import keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict


# In[317]:

#importing the data and shuffuling 

df_full = pd.read_excel('credit.xls')

df = df_full

df=shuffle(df)

df.head()


# In[318]:

#Scaling the data


x1 = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x1)
df = pd.DataFrame(x_scaled)


# In[319]:

# Selecting the Traget and predictor variables

y = df.iloc[:,24]
X = df.iloc[:,[1,6,7,8,9,10,11,17,18,19,20,21,22]]

X.head()


# In[320]:

#Test Train split

X_train, X_test, y_train, y_test = train_test_split(X,y)

X_train, X_test, y_train, y_test = np.asarray(X_train),np.asarray(X_test),np.asarray(y_train),np.asarray(y_test)


# In[321]:

#Building the Neural Network Convolutional Graph

model1 = Sequential()

model1.add(Dense(26,input_dim = 13, activation='relu'))

model1.add(Dense(13,activation='relu'))

model1.add(Dense(1, activation='sigmoid'))

model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[323]:

#Spliting the probability with baseline 0.5

y_pred = model1.predict(X_test)
y_pred[y_pred > 0.50] = 1
y_pred[y_pred < 0.50] = 0


# In[324]:

y_pred = y_pred.reshape(len(y_test),)


# In[331]:

# Generating the Default probability split Results 


print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

print(accuracy_score(y_test,y_pred))


# In[326]:

# Code to generate Recall, Accuracy & Cost for vaiying probability Cut_off level


cst1 = []
cut_off = []
accu = []
reca = []

for i in range(10,60,1):
    y_pred = model1.predict(X_test)
    y_pred[y_pred > (i/100)] = 1
    y_pred[y_pred < (i/100)] = 0
    cost = (mat[0,1] * 0.25 ) + (mat[1,0]*1)
    acc = accuracy_score(y_test,y_pred)
    mat = confusion_matrix(y_test,y_pred)
    rec = mat[1,1]/(mat[1,0]+mat[1,1])
    cst1.append(cost)
    cut_off.append(i/100)
    accu.append(acc)
    reca.append(rec)
    
    
cut_off = np.asarray(cut_off)
cut_off = cut_off.reshape(len(cut_off),1)
cst1 = np.asarray(cst1)
cst1 = cst1.reshape(len(cst1),1)
accu = np.asarray(accu)
accu = accu.reshape(len(accu),1)
reca = np.asarray(reca)
reca = reca.reshape(len(reca),1)
com = np.concatenate((cut_off,cst1,accu,reca),axis=1)
cst_tab= pd.DataFrame(com,columns=['Cut_off','Cost','Accu','Reca'])


# In[332]:

#Plot to Check the Cost Vs Cut_off

import matplotlib.pyplot as plt
plt.plot(cst_tab['Cut_off'],cst_tab['Cost'])
plt.xlabel('Cut_off')
plt.ylabel('Cost')
plt.grid(True)
plt.title("Cost Vs Cut_off")
plt.show()


# In[333]:

#Plot to check the Cut_off Vs Recall and Accuracy

import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.plot(cst_tab['Cut_off'],cst_tab['Reca'],label='Recall')
plt.plot(cst_tab['Cut_off'],cst_tab['Accu'],label='Accuracy')
#plt.plot(cst_tab['Cut_off'],cst_tab['Cost'],label='Cost')
plt.legend()
plt.xlabel('Cut_off')
plt.ylabel('Recall & Accu')
plt.grid(True)
plt.title('Cut_off Vs Recall and Accuracy')
plt.show()


# In[334]:

#All three parameters combined Plot

fig, ax1 = plt.subplots()
ax1.plot(cst_tab['Cut_off'],cst_tab['Reca'],label='Recall')
ax1.plot(cst_tab['Cut_off'],cst_tab['Accu'],label='Accuracy')
ax1.set_xlabel('Cut_off')
ax1.set_ylabel('Accuracy & Recall', color='b')
#ax1.tick_params('y')
plt.legend(prop={'size': 10})
plt.grid(True)
ax2 = ax1.twinx()
ax2.plot(cst_tab['Cut_off'],cst_tab['Cost'],'--',label='Cost',color='green')
ax2.set_ylabel('Cost_Value', color='r')
ax2.tick_params('y', colors='r')
plt.legend(bbox_to_anchor=(0.67,0.91),
           bbox_transform=plt.gcf().transFigure,prop={'size': 10})
plt.title('Attributes Trade off Plot')

fig.tight_layout()
plt.show()


# In[335]:

# Final Results

y_pred = model1.predict(X_test)
y_pred[y_pred > 0.22] = 1
y_pred[y_pred < 0.22] = 0
print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

print("The model Accuracy is %f" %accuracy_score(y_test,y_pred))

mat = confusion_matrix(y_test,y_pred)

Cost = (mat[0,1] * 0.25 ) + (mat[1,0]*1)


print("The overall Cost to company for False predictions is %d unit points" % Cost)

