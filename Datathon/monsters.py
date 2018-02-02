import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB



train = pd.read_csv('./mdata/train.csv')
test = pd.read_csv('./mdata/test.csv')

print train.head(n=10)
sns.set()
sns.pairplot(train[["bone_length", "rotting_flesh", "hair_length", "has_soul", "color","type"]], hue="type")


train['hair_soul'] = train['hair_length'] * train['has_soul']
train['hair_bone'] = train['hair_length'] * train['bone_length']
# train["hair_flesh"] = train['hair_length'] * train['rotting_flesh']
# train["bone_flesh"] = train['bone_length'] * train['rotting_flesh']
train['hair_soul_bone'] = train['hair_length'] * train['has_soul'] * train['bone_length']
# train['hair_rotting_bone'] = train['hair_length'] * train['rotting_flesh'] * train['bone_length']
test['hair_soul'] = test['hair_length'] * test['has_soul']
test['hair_bone'] = test['hair_length'] * test['bone_length']
test['hair_soul_bone'] = test['hair_length'] * test['has_soul'] * test['bone_length']
# test["hair_flesh"] = test['hair_length'] * test['rotting_flesh']
# test["bone_flesh"] = test['bone_length'] * test['rotting_flesh']
# test['hair_rotting_bone'] = test['hair_length'] * test['rotting_flesh'] * test['bone_length']

sns.pairplot(train[["hair_soul", "hair_bone", "hair_soul_bone","type"]], hue="type")



plt.show()


test_id = test['id']
train.drop(['id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)


col = 'color'
dummies = pd.get_dummies(train[col], drop_first=False)
dummies = dummies.add_prefix("{}#".format(col))
train.drop(col, axis=1, inplace=True)
train = train.join(dummies)
dummies = pd.get_dummies(test[col], drop_first=False)
dummies = dummies.add_prefix("{}#".format(col))
test.drop(col, axis=1, inplace=True)
test = test.join(dummies)


X_train = train.drop('type', axis=1)
le = LabelEncoder()
Y_train = le.fit_transform(train.type.values)
X_test = test


clf = RandomForestClassifier(n_estimators=200)
clf = clf.fit(X_train, Y_train)
indices = np.argsort(clf.feature_importances_)[::-1]


for i in range(X_train.shape[1]):
    print('%d. feature %d %s (%f)' % (i + 1, indices[i], X_train.columns[indices[i]],clf.feature_importances_[indices[i]]))

best_features=X_train.columns[indices[0:7]]
X = X_train[best_features]
Xt = X_test[best_features]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y_train, test_size=0.30, random_state=100)

logreg = LogisticRegression()
#
parameter_grid = {'solver' : ['newton-cg'],
                  'multi_class' : ['multinomial'],
                  'C' : [0.01, 1, 10,100],
                  'tol': [0.000001,0.00001,0.0001, 0.001, 0.005]
                 }

grid_search = GridSearchCV(logreg, param_grid=parameter_grid, cv=StratifiedKFold(5))
grid_search.fit(Xtrain, ytrain)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


predicted = le.inverse_transform(grid_search.fit(X, Y_train).predict(Xt))
submission = pd.DataFrame({'id':test_id, 'type':predicted})
submission.to_csv('monster_submission.csv', index=False)

calibrated_clf = CalibratedClassifierCV(RandomForestClassifier())
log_reg1 = LogisticRegression()
gnb = GaussianNB()
VC = VotingClassifier(estimators=[('LR', log_reg1), ('CRF', calibrated_clf),('GNB', gnb)], voting='hard')

predicted2 = le.inverse_transform(VC.fit(X, Y_train).predict(Xt))
submission2 = pd.DataFrame({'id':test_id, 'type':predicted2})
submission2.to_csv('monster_submission2.csv', index=False)
