# changing the directory
import os
print (os.getcwd())                                                   # see where you are
os.chdir(r'C:\Users\iamka\Desktop\Quarter 5\Python\Final project')

# KANNAN
# CREDIT CARD DEFAULT PREDICTION
# MODEL  : LOGISTIC REGRESSION
# DATA   : 12/04/2017 - PYTHON FINAL PRESENTATION
# Libraries
import pandas as pd
import numpy as np
import sklearn as sk


# Load Data File
df = pd.read_excel('default of credit card clients1.xlsx')


# Transformation of columns
#j='default payment next month'
#df[j] = df[j].astype(int)
#corr = {'asis':0, 'log':0, 'exp':0, 'sqrt':0, 'pow2':0}
#for i in df.columns.tolist():
#    if df.dtypes[i] != 'object':
#        corr['asis'] = abs(np.corrcoef(df[i], df[j])[1][0])
#        if all((df[i]>=0)):
#            corr['log'] = abs(np.corrcoef(np.log(df[i] + 0.00001), df[j])[1][0])
#            corr['sqrt'] = abs(np.corrcoef(np.sqrt(df[i] + 0.00001), df[j])[1][0])
#        else:
#            corr['log'] = 0  
#            corr['sqrt'] = 0        
#        corr['exp'] = abs(np.corrcoef(np.exp(df[i]), df[j])[1][0])
#        corr['pow2'] = abs(np.corrcoef(np.power(df[i],2), df[j])[1][0])
#        if max(corr,key=corr.get) == 'asis':
#            df[i]=df[i]
#            print 'asis:',i 
#        elif max(corr,key=corr.get) == 'log':
#            df[i]=abs(np.log(df[i]))
#            print 'log',i
#        elif max(corr,key=corr.get) == 'sqrt':
#            df[i]=np.sqrt(df[i])
#            print 'sqrt',i
#        elif max(corr,key=corr.get) == 'exp':
#            df[i]=abs(np.exp(df[i]))
#            print 'exp',i
#        else:
#            print 'power',i
#            df[i]=np.power(df[i],2)
#            
#
#df1= pd.DataFrame(df)
#df1=df1.replace([np.inf, -np.inf], np.nan)  # replace all infinite values with NA values
#df1=df1.dropna()                            # drop the row if any of the values is NA
#df=df1            


# Feature Extraction with RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
array = df.values
X = array[:,0:23]
Y = array[:,-1]
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 7)            # Selecting top 9 variables from df
fit = rfe.fit(X, Y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_
a=fit.ranking_


# Dropping columns based on variable selection 
df.drop(df.columns[[0,2,4,8,9,10,11,12,13,14,15,16,19,20,21,22]], axis=1, inplace=True)

# Splitting the predictor and target variable
x1 = df.values      #returns a numpy array
X = x1[:,0:6] 
Y=x1[:,-1]


# Scaling the filtered columns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
df1 = pd.DataFrame(rescaledX)                # summarize transformed data


# Assigning Values to X & Y after data transformation
X =df1.iloc[:,0:6] 
Y =df.iloc[:,-1]


# Splitting Data set into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)


# Create SVM classification object 
from sklearn import svm
model = svm.SVC(kernel='rbf', C=100, gamma=10) # tune model by using kernel='linear', C=1,10, gamma=0.2,10,100,1000)
model.fit(X_train, Y_train)              #Model created using train data
model.score(X_test, Y_test)              #Model Accuracy


#Predict Output
predicted= model.predict(X_test)


# Confusion Matrix, Accuracy & Classification Report
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
cnf_matrix = confusion_matrix(Y_test,predicted)     # Confusion matrix to calculate Precision, Recall
print(classification_report(Y_test,predicted))
print(accuracy_score(Y_test,predicted))


# Performing Cross Validation
from sklearn.model_selection import cross_val_score
cross= cross_val_score(model,X,Y,cv=5)             # for performing Cross validation # Runs for atleat 5min