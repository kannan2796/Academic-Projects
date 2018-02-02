    
# coding: utf-8

# # Import Packages

# In[135]:
# KANNAN
# CREDIT CARD DEFAULT PREDICTION
# MODEL  : LOGISTIC REGRESSION
# DATA   : 12/04/2017 - PYTHON FINAL PRESENTATION
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import time 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import datasets, linear_model
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


# # Mandatory Settings#
# 

# In[180]:

filename = "credit.csv"

#Custom_variables_flag = 'N'
#Var_List =[]

Drop_col_list = ['ID']
DropColumn_threshold = 0.9

#Replace missing for Numerical (Mean,Median,Mode)
replace_missing_num = 'mean' 

#User Defined Model
numTrees=100 
impurity="entropy" #( Supported values: “gini” or “entropy”. (default: “gini”)) 
maxDepth= 5 #( Depth 0 means 1 leaf node, Depth 1 means 1 internal node + 2 leaf nodes). (default: 4))
maxBins=106 #(Maximum number of bins used for splitting features. (default: 32))


# In[181]:

#INPUT-FILE_DETAILS
df = pd.read_csv(filename)
df.rename(columns={'default payment next month':'target'}, inplace=True)

nans = []
for i in df.columns:
    if (float(df[i].isnull().sum())/df[i].shape[0]) > DropColumn_threshold:
            nans.append(i)
df.drop(nans, axis = 1, inplace=True)

#replace_missing
if replace_missing_num == 'median':
    for i in df.select_dtypes(include = ['float64', 'float32', 'int']).columns:
        df[i].fillna(df[i].median(), inplace=True)
elif replace_missing_num == 'mean':
    for i in df.select_dtypes(include = ['float64', 'float32', 'int']).columns:
        df[i].fillna(df[i].mean(), inplace=True)
elif replace_missing_num =='mode':
    for i in df.select_dtypes(include = ['float64', 'float32', 'int']).columns:
        df[i].fillna(df[i].mode(), inplace=True)
elif replace_missing_num == 'zero':
    df[i].fillna(0, inplace=True)

df.drop(Drop_col_list,axis=1,inplace=True)
target='default payment next month'


# # Initial Analysis

# In[182]:

df.describe()


# In[183]:

n, bins, patches =plt.hist(df['AGE'], 10, normed=1, facecolor='b', alpha=0.5)
plt.xlabel('age')
plt.ylabel('Probability')
plt.title('Histogram of Age')
plt.axis([0, 100, 0, 0.05])
plt.grid(True)
plt.show()


# In[184]:

plt.figure(1, (8, 8))   # try plt.figure(1, (10, 10)) 
plt.subplot(111) 
box = plt.boxplot([df.loc[df['target']==0, 'LIMIT_BAL'].tolist(), df.loc[df['target']==1, 'LIMIT_BAL'].tolist()], patch_artist=True)
col = ['r', 'b']
i=0
for b in box['boxes']:
    b.set(color='r', linewidth=2)
    b.set(facecolor = col[i])
    i = i+1
plt.show()


# In[185]:

df_plot = df[['MARRIAGE','SEX','target']]
df_plot['MARRIAGE'] = df_plot['MARRIAGE'].astype(str)
replace_nums = {"SEX": {1: "Male", 2: "Female"},
               "MARRIARGE": {0: "None",1: "Married", 2: "Single", 3: "Others"},}
df_plot.replace(replace_nums, inplace=True)


# In[186]:

replace_nums = {"SEX": {1: "Male", 2: "Female"},
               "MARRIARGE": {0: "None",1: "Married", 2: "Single", 3: "Others"},}
df_plot.replace(replace_nums, inplace=True)


# In[187]:

sns.factorplot(x="MARRIAGE", y="target", hue = 'SEX', kind="bar", data=df_plot)
plt.show()


# In[188]:

columns1 = df.columns
f, (ax1) = plt.subplots(1, figsize=(15,10))
ax1.set_title("Pearson")
#assign the cbar to be in that axis using the cbar_ax kw
sns.heatmap(df.corr(), ax=ax1 ,yticklabels=columns1,xticklabels=columns1, annot=True, fmt='.0%')
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 
ax1.figure.tight_layout()
plt.show()


# In[189]:

all_input_var = ['AGE','BILL_AMT1','SEX','PAY_6','PAY_4','PAY_5','PAY_2','PAY_3','PAY_0','BILL_AMT5','BILL_AMT4','BILL_AMT6',
'BILL_AMT3','BILL_AMT2','PAY_AMT6','PAY_AMT5','PAY_AMT4','PAY_AMT3','PAY_AMT2','PAY_AMT1','LIMIT_BAL','MARRIAGE','EDUCATION']
vif = []
for i in range(df[all_input_var].shape[1]):
    vif.append(variance_inflation_factor(df[all_input_var].values,i))
vif = pd.DataFrame(vif)
names = pd.DataFrame(df[all_input_var].columns.tolist())
t = pd.concat([names,vif],axis=1)
t.columns = ['variable','vif_score']


# In[190]:

a= t.loc[t['vif_score'] <=5,['variable']].values.flatten().tolist()


# In[200]:

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.3,random_state=1991)


# In[201]:

test.shape


# In[11]:

train_x = train[a]
train_y = train[target]
test_x = test[a]
test_y = test[target]
features = train_x.columns


# In[12]:

# Feature Importance
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(train_x, train_y)
# display the relative importance of each attribute
print(model.feature_importances_)


# In[35]:

train_x.columns


# In[13]:

def Performance(Model,Y,X):
    # Perforamnce of the model
    fpr, tpr, _ = roc_curve(Y, Model.predict_proba(X)[:,1])
    AUC  = auc(fpr, tpr)
    print ('the AUC is : %0.4f' %  AUC)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % AUC)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


# # User Defined Model

# In[40]:

start_time2 = time.time()
RFclassifier = RandomForestClassifier(n_estimators = 10, min_samples_split=6,criterion='gini', max_depth= maxDepth) 
RFmodel = RFclassifier.fit(train_x,train_y)
RFpredictions1 = RFmodel.predict(train_x)
RFpredictions2 = RFmodel.predict(test_x)
RFauroc1 = roc_auc_score(train_y, RFpredictions1)
RFauroc2 = roc_auc_score(test_y, RFpredictions2)


# In[41]:

RFpredictions1


# In[42]:

pd.crosstab(train_y, RFpredictions1, rownames=['Actual Species'], colnames=['Predicted Species'])


# In[43]:

from sklearn.metrics import confusion_matrix
confusion_matrix(train_y, RFpredictions1)


# In[44]:

Performance(Model=RFmodel,Y=test_y,X=test_x)


# In[19]:

from sklearn.metrics import accuracy_score
accuracy_score(train_y, RFpredictions1)


# In[20]:

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(train_y, RFpredictions1,average = 'binary')


# In[21]:

pd.crosstab(test[target], RFpredictions2, rownames=['Actuals'], colnames=['Predictions'])


# In[22]:

from sklearn.metrics import accuracy_score
accuracy_score(test_y, RFpredictions2)


# In[23]:

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(test_y, RFpredictions2,average = 'binary')


# # Automatic Model Selection using Grid Search

# In[36]:

from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import KFold
grid_1 = { "n_estimators"      : [100,200,500],
               "criterion"         : ["gini","entropy"],
               "max_features"      : ['auto'],
               "max_depth"         : [3,4,6,10]}
RF=RandomForestClassifier()
grid_search = GridSearchCV(RF, grid_1, n_jobs=-1, cv=5)
grid_search.fit(train_x, train_y)


# In[37]:

grid_search.best_score_


# In[38]:

grid_search.best_params_


# In[219]:

Performance(Model=grid_search,Y=test_y,X=test_x)


# In[241]:

ypred = grid_search.predict_proba(test_x)


# In[164]:

ypred = grid_search.predict_proba(test_x)
ypred[ypred<0.5] = 0
ypred[ypred>=0.5] = 1
y_pred = [item[1] for item in ypred]


# In[165]:

precision_recall_fscore_support(test_y, y_pred,average = 'binary')


# In[166]:

accuracy_score(test_y, y_pred)


# In[236]:

ypred = grid_search.predict_proba(test_x)
ypred[ypred<0.25] = 0
ypred[ypred>=0.25] = 1
y_pred = [item[1] for item in ypred]


# In[237]:

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(test_y, y_pred,average = 'binary')


# In[169]:

from sklearn.metrics import accuracy_score
accuracy_score(test_y, y_pred)


# In[59]:

from sklearn.metrics import confusion_matrix
confusion_matrix(test_y, y_pred)


# In[86]:

from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
def cutoff_predict(clf,X,cutoff):
    return(clf.predict_proba(X)[:,1] > cutoff).astype(int)

scores = []

def custom_f1(cutoff):
    def f1_cutoff(clf,X,y):
        ypred = cutoff_predict(clf,X, cutoff)
        return precision_recall_fscore_support(y_true, y_pred, average='binary')
    return f1_cutoff

for cutoff in np.arange(0.1,0.9,0.1):
    clf = RandomForestClassifier(n_estimators = 500,criterion='gini', max_depth=6, max_features='auto')
    validated = cross_val_score(clf,train_x,train_y, cv = 5, scoring = custom_f1(cutoff))
    scores.append(validated)


# In[125]:

scores


# # F score calculation for choosing the best cut off

# In[168]:

sns.boxplot(x=np.arange(0.1,0.9,0.1),y=scores)
plt.title('F score for each tree')
plt.xlabel('each cutoff value')
plt.ylabel('customer F score')
plt.show()


# In[194]:

ypred = grid_search.predict_proba(train_x)
ypred[ypred<=0.25] = 0
ypred[ypred>0.25] = 1
y_pred = [item[1] for item in ypred]


# In[195]:

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(train_y, y_pred,average = 'binary')


# In[196]:

from sklearn.metrics import accuracy_score
accuracy_score(train_y, y_pred)


# In[238]:

y_pred = pd.DataFrame(y_pred)


# In[239]:

test.reset_index(drop=True, inplace=True)
y_pred.reset_index(drop=True, inplace=True)


# In[240]:

a=pd.concat([test,y_pred], axis=1)
a.head()


# In[245]:

ypred
ypred1 = pd.DataFrame([item[1] for item in ypred])
b=pd.concat([a,ypred1],axis=1)
b.columns = ['LIMIT_BAL', 'SEX','EDUCATION','MARRIAGE', 'AGE','PAY_0', 'PAY_2','PAY_3','PAY_4','PAY_5',
             'PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4',
             'BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3',
             'PAY_AMT4','PAY_AMT5','PAY_AMT6','target','predicted_class','predicted_probability']
b.dtypes


# In[253]:

b.dtypes


# In[244]:

import sklearn.metrics
import pandas as pd

def calc_cumulative_gains(df, actual_col, predicted_col, probability_col):
    df.sort_values(by=probability_col, ascending=False, inplace=True)

    subset = df[df[predicted_col] == True]

    rows = []
    for group in np.array_split(subset, 10):
        score = sklearn.metrics.accuracy_score(group[actual_col].tolist(),
                                               group[predicted_col].tolist(),
                                               normalize=False)

    rows.append({'NumCases': len(group), 'NumCorrectPredictions': score})

    lift = pd.DataFrame(rows)

#Cumulative Gains Calculation
    lift['RunningCorrect'] = lift['NumCorrectPredictions'].cumsum()
    lift['PercentCorrect'] = lift.apply(
        lambda x: (100 / lift['NumCorrectPredictions'].sum()) * x['RunningCorrect'], axis=1)
    lift['CumulativeCorrectBestCase'] = lift['NumCases'].cumsum()
    lift['PercentCorrectBestCase'] = lift['CumulativeCorrectBestCase'].apply(
        lambda x: 100 if (100 / lift['NumCorrectPredictions'].sum()) * x > 100 else (100 / lift[
            'NumCorrectPredictions'].sum()) * x)
    lift['AvgCase'] = lift['NumCorrectPredictions'].sum() / len(lift)
    lift['CumulativeAvgCase'] = lift['AvgCase'].cumsum()
    lift['PercentAvgCase'] = lift['CumulativeAvgCase'].apply(
        lambda x: (100 / lift['NumCorrectPredictions'].sum()) * x)

    #Lift Chart
    lift['NormalisedPercentAvg'] = 1
    lift['NormalisedPercentWithModel'] = lift['PercentCorrect'] / lift['PercentAvgCase']

    return lift


# In[292]:

liftmodel=calc_cumulative_gains(b,'target','predicted_class', 'predicted_probability')
liftmodel = pd.DataFrame(liftmodel)
liftmodel


# In[266]:

def calc_lift(x,y,clf,bins=10):
    #Actual Value of y
    y_actual = y
    #Predicted Probability that y = 1
    y_prob = clf.predict_proba(x)
    #Predicted Value of Y
    y_pred = clf.predict(x)
    cols = ['ACTUAL','PROB_POSITIVE','PREDICTED']
    data = [y_actual,y_prob[:,1],y_pred]
    df = pd.DataFrame(dict(zip(cols,data)))
    
    #Observations where y=1
    total_positive_n = df['ACTUAL'].sum()
    #Total Observations
    total_n = df.index.size
    natural_positive_prob = total_positive_n/float(total_n)


    #Create Bins where First Bin has Observations with the
    #Highest Predicted Probability that y = 1
    df['BIN_POSITIVE'] = pd.qcut(df['PROB_POSITIVE'],bins,labels=False)
    
    pos_group_df = df.groupby('BIN_POSITIVE')
    #Percentage of Observations in each Bin where y = 1 
    lift_positive = pos_group_df['ACTUAL'].sum()/pos_group_df['ACTUAL'].count()
    lift_index_positive = (lift_positive/natural_positive_prob)*100
    
    
    #Consolidate Results into Output Dataframe
    lift_df = pd.DataFrame({'LIFT_POSITIVE':lift_positive,
                               'LIFT_POSITIVE_INDEX':lift_index_positive,
                               'BASELINE_POSITIVE':natural_positive_prob})
    
    return lift_df


# In[273]:

lift=calc_lift(test_x,test_y,grid_search)


# In[281]:

lift


# In[333]:

x=np.arange(0,10)
x = x[::-1]
plt.plot(x, lift['LIFT_POSITIVE_INDEX']/100, 'ro')
plt.show()

