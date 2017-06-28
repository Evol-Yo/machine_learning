# Imports

# pandas
import matplotlib
import pandas as pd
from pandas import Series,DataFrame

import xgboost as xgb

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from tensorflow.contrib.distributions.python.ops.bijectors import inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)
test_df    = test_df.drop(['Name','Ticket'], axis=1)

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

# sns.factorplot('Embarked','Survived', data=titanic_df,size=4,aspect=3)

## embark
embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

embark_dummies_test  = pd.get_dummies(test_df['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)

titanic_df = titanic_df.join(embark_dummies_titanic)
test_df    = test_df.join(embark_dummies_test)

titanic_df.drop(['Embarked'], axis=1,inplace=True)
test_df.drop(['Embarked'], axis=1,inplace=True)

## Fare
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

titanic_df['Fare'] = titanic_df['Fare'].astype(int)
test_df['Fare']    = test_df['Fare'].astype(int)

# fare_not_survived = titanic_df["Fare"][titanic_df["Survived"] == 0]
# fare_survived     = titanic_df["Fare"][titanic_df["Survived"] == 1]
#
# avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
# std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])


## Age
average_age_titanic   = titanic_df["Age"].mean()
std_age_titanic       = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum()

average_age_test   = test_df["Age"].mean()
std_age_test       = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()

rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1
test_df["Age"][np.isnan(test_df["Age"])] = rand_2

titanic_df['Age'] = titanic_df['Age'].astype(int)
test_df['Age']    = test_df['Age'].astype(int)

titanic_df.drop("Cabin",axis=1,inplace=True)
test_df.drop("Cabin",axis=1,inplace=True)

## Family
titanic_df['Family'] =  titanic_df["Parch"] + titanic_df["SibSp"]
titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1
titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0

test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]
test_df['Family'].loc[test_df['Family'] > 0] = 1
test_df['Family'].loc[test_df['Family'] == 0] = 0

titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)
test_df    = test_df.drop(['SibSp','Parch'], axis=1)

## Sex

def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex

titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person,axis=1)
test_df['Person']    = test_df[['Age','Sex']].apply(get_person,axis=1)

titanic_df.drop(['Sex'],axis=1,inplace=True)
test_df.drop(['Sex'],axis=1,inplace=True)

person_dummies_titanic  = pd.get_dummies(titanic_df['Person'])
person_dummies_titanic.columns = ['Child','Female','Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

person_dummies_test  = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child','Female','Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

titanic_df = titanic_df.join(person_dummies_titanic)
test_df    = test_df.join(person_dummies_test)

titanic_df.drop(['Person'],axis=1,inplace=True)
test_df.drop(['Person'],axis=1,inplace=True)

## Pclass

pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

titanic_df.drop(['Pclass'],axis=1,inplace=True)
test_df.drop(['Pclass'],axis=1,inplace=True)

titanic_df = titanic_df.join(pclass_dummies_titanic)
test_df    = test_df.join(pclass_dummies_test)

X_train = titanic_df.drop("Survived",axis=1)
Y_train = titanic_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()

###########################################################################
# print("LogisticRegression start...")
# logreg = LogisticRegression()
#
# print("Fit...")
# logreg.fit(X_train, Y_train)
#
# print("Predicting...")
# Y_pred = logreg.predict(X_test)
#
# print("Scoring")
# score = logreg.score(X_train, Y_train)

###########################################################################
# svc = SVC()
#
# print("Svm fitting...")
# svc.fit(X_train, Y_train)
#
# print("Predicting...")
# Y_pred = svc.predict(X_test)
#
# print("Scoring...")
# score = svc.score(X_train, Y_train)
#
# print(score)

###########################################################################
# random_forest = RandomForestClassifier(n_estimators=100)
#
# print("Training...")
# random_forest.fit(X_train, Y_train)
#
# print("Predicting...")
# Y_pred = random_forest.predict(X_test)
#
# print("Scoring...")
# score = random_forest.score(X_train, Y_train)
#
# print(score)

#
# gaussian = GaussianNB()
#
# gaussian.fit(X_train, Y_train)
#
# Y_pred = gaussian.predict(X_test)
#
# score = gaussian.score(X_train, Y_train)
# print(score)


###########################################################################
params={
    'booster':['gbtree'],
    'objective': ['binary:logistic'],
    'max_depth':np.arange(5, 10, 1), # 构建树的深度 [1:]
    'subsample':np.arange(0.6, 1, 0.1), # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    'colsample_bytree': np.arange(0.6, 1, 0.1), # 构建树树时的采样比率 (0:1]
    'seed': [4321],
    'nthread':[2],# cpu 线程数,根据自己U的个数适当调整
}

xgtrain = xgb.DMatrix(X_train, label=Y_train)
xgtest = xgb.DMatrix(X_test)

num_rounds=200

xgb_model = xgb.XGBClassifier(num_rounds = 200, eta=0.1, silent=1)

clf = GridSearchCV(xgb_model, params, n_jobs=2,
                   cv = StratifiedKFold(Y_train, n_folds=5, shuffle=True),
                   scoring='roc_auc')
clf.fit(X_train, Y_train)

for field in clf.grid_scores_:
    print(field)
print(clf.best_params_)
print(clf.best_score_)
# model = xgb.train(params, xgtrain, num_rounds)

# results = xgb_model.predict(xgtest)
