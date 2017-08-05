# Imports

import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold

from sklearn.ensemble import RandomForestRegressor

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# machine learning
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

titanic_df = titanic_df.drop(['PassengerId', 'Ticket'], axis=1)
test_df    = test_df.drop(['Ticket'], axis=1)

## Name
titanic_df["NameLength"] = titanic_df["Name"].apply(lambda x: len(x))
test_df["NameLength"] = test_df["Name"].apply(lambda x: len(x))

bins = [0, 20, 40, 57, 85]
group_names = ['short', 'okay', 'good', 'long']
titanic_df['NlengthD'] = pd.cut(titanic_df['NameLength'], bins, labels=group_names)
test_df['NlengthD'] = pd.cut(test_df['NameLength'], bins, labels=group_names)

titanic_df = titanic_df.drop(['Name', 'NameLength'], axis=1)
test_df    = test_df.drop(['Name', 'NameLength'], axis=1)

## Embarked
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
test_df["Embarked"] = test_df["Embarked"].fillna("S")

## Fare
titanic_df["Fare"].fillna(titanic_df["Fare"].median(), inplace=True)
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

titanic_df['Fare'] = titanic_df['Fare'].astype(int)
test_df['Fare']    = test_df['Fare'].astype(int)

## Cabin
titanic_df["Deck"]=titanic_df.Cabin.str[0]
test_df["Deck"]=test_df.Cabin.str[0]

titanic_df.Deck.fillna('Z', inplace=True)
test_df.Deck.fillna('Z', inplace=True)

titanic_df.drop("Cabin",axis=1,inplace=True)
test_df.drop("Cabin",axis=1,inplace=True)

## Family
titanic_df["FamilySize"] = titanic_df["SibSp"] + titanic_df["Parch"] + 1
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1

titanic_df.loc[titanic_df["FamilySize"] == 1, "FsizeD"] = 'singleton'
titanic_df.loc[(titanic_df["FamilySize"] > 1)  &  (titanic_df["FamilySize"] < 5) , "FsizeD"] = 'small'
titanic_df.loc[titanic_df["FamilySize"] > 4, "FsizeD"] = 'large'

test_df.loc[test_df["FamilySize"] == 1, "FsizeD"] = 'singleton'
test_df.loc[(test_df["FamilySize"] > 1) & (test_df["FamilySize"] <5) , "FsizeD"] = 'small'
test_df.loc[test_df["FamilySize"] > 4, "FsizeD"] = 'large'

titanic_df = titanic_df.drop(['SibSp', 'Parch'], axis=1)
test_df    = test_df.drop(['SibSp', 'Parch'], axis=1)

## Sex

def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex

titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person,axis=1)
test_df['Person']    = test_df[['Age','Sex']].apply(get_person,axis=1)

titanic_df.drop(['Sex'],axis=1,inplace=True)
test_df.drop(['Sex'],axis=1,inplace=True)

labelEnc=LabelEncoder()
cat_vars=['Deck', 'FsizeD', 'NlengthD', 'Embarked', 'FsizeD', 'Pclass', 'Person']
for col in cat_vars:
    titanic_df[col]=labelEnc.fit_transform(titanic_df[col])
    test_df[col]=labelEnc.fit_transform(test_df[col])

def fill_missing_age(df):
    # Feature set
    age_df = df[['Age', 'NlengthD', 'Fare', 'Embarked', 'Deck', 'FsizeD', 'Pclass', 'Person']]
    # Split sets into train and test
    train = age_df.loc[(df.Age.notnull())]  # known Age values
    test = age_df.loc[(df.Age.isnull())]  # null Ages

    # All age values are stored in a target array
    y = train.values[:, 0]

    # All the other values are stored in the feature array
    X = train.values[:, 1::]

    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    rtr.fit(X, y)

    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(test.values[:, 1::])

    # Assign those predictions to the full data set
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df

titanic_df=fill_missing_age(titanic_df)
test_df=fill_missing_age(test_df)

std_scale = preprocessing.StandardScaler().fit(titanic_df[['Age', 'Fare']])
titanic_df[['Age', 'Fare']] = std_scale.transform(titanic_df[['Age', 'Fare']])

std_scale = preprocessing.StandardScaler().fit(test_df[['Age', 'Fare']])
test_df[['Age', 'Fare']] = std_scale.transform(test_df[['Age', 'Fare']])

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
#
# print(score)

# from sklearn import cross_validation
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import ShuffleSplit
#
# # predictors = ["Pclass", "Sex", "Fare", "Embarked", "Deck", "Age",
# #               "FsizeD", "NlengthD", "Parch"]
#
# # predictors = ["Pclass", "Sex", "Fare", "Embarked", "Age",
# #               "FsizeD",  "Parch"]
#
# # Initialize our algorithm
# lr = LogisticRegression(random_state=1)
# # Compute the accuracy score for all the cross validation folds.
# cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)
#
# scores = cross_val_score(lr, X_train, Y_train, scoring='f1', cv=cv)
# # Take the mean of the scores (because we have one for each fold)
# print(scores.mean())

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
# results = random_forest.predict(X_test)
#
# submission = pd.DataFrame({
#     "PassengerId": test_df["PassengerId"],
#     "Survived": results.astype(np.int16)
# })
# submission.to_csv('titanic.csv', index=False)


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
# params={
#     'booster':'gbtree',
#     'objective': 'binary:logistic',
#     'max_depth':5, # 构建树的深度 [1:]
#     'subsample':0.8, # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
#     'colsample_bytree': 0.6, # 构建树树时的采样比率 (0:1]
#     'eta':0.03,
#     'silent':0,
#     'seed': 1234,
#     'nthread': 2,# cpu 线程数,根据自己U的个数适当调整
# }
#
# xgtrain = xgb.DMatrix(X_train, label=Y_train)
# xgtest = xgb.DMatrix(X_test)
#
# num_rounds = 200
# xgb_model = xgb.train(params, xgtrain, num_rounds)
# results = xgb_model.predict(xgtest)
# results[results>=0.5] = 1
# results[results<0.5] = 0
#
# submission = pd.DataFrame({
#     "PassengerId": test_df["PassengerId"],
#     "Survived": results.astype(np.int16)
# })
# submission.to_csv('titanic.csv', index=False)

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
