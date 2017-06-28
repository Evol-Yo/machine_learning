import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

titanic = pd.read_csv("../input/train.csv")
titanic_test = pd.read_csv("../input/test.csv")

## Embarked
titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic_test["Embarked"] = titanic_test["Embarked"].fillna('S')

## Fare
def fill_missing_fare(df):
    median_fare = df["Fare"].median()
    df["Fare"] = df["Fare"].fillna(median_fare)
    return df

titanic=fill_missing_fare(titanic)
titanic_test=fill_missing_fare(titanic_test)

titanic['Fare'] = titanic['Fare'].astype(int)
titanic_test['Fare']    = titanic_test['Fare'].astype(int)

titanic["Deck"]=titanic.Cabin.str[0]
titanic_test["Deck"]=titanic_test.Cabin.str[0]

titanic.Deck.fillna('Z', inplace=True)
titanic_test.Deck.fillna('Z', inplace=True)

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]+1
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]+1

titanic.loc[titanic["FamilySize"] == 1, "FsizeD"] = 'singleton'
titanic.loc[(titanic["FamilySize"] > 1)  &  (titanic["FamilySize"] < 5) , "FsizeD"] = 'small'
titanic.loc[titanic["FamilySize"] >4, "FsizeD"] = 'large'

titanic_test.loc[titanic_test["FamilySize"] == 1, "FsizeD"] = 'singleton'
titanic_test.loc[(titanic_test["FamilySize"] >1) & (titanic_test["FamilySize"] <5) , "FsizeD"] = 'small'
titanic_test.loc[titanic_test["FamilySize"] >4, "FsizeD"] = 'large'

titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))

bins = [0, 20, 40, 57, 85]
group_names = ['short', 'okay', 'good', 'long']
titanic['NlengthD'] = pd.cut(titanic['NameLength'], bins, labels=group_names)
titanic_test['NlengthD'] = pd.cut(titanic_test['NameLength'], bins, labels=group_names)

def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex

titanic['Person'] = titanic[['Age','Sex']].apply(get_person,axis=1)
titanic_test['Person']    = titanic_test[['Age','Sex']].apply(get_person,axis=1)

labelEnc=LabelEncoder()
cat_vars=['Embarked', 'Sex', "FsizeD", "NlengthD", 'Deck', 'Person']
for col in cat_vars:
    titanic[col]=labelEnc.fit_transform(titanic[col])
    titanic_test[col]=labelEnc.fit_transform(titanic_test[col])


def fill_missing_age(df):
    # Feature set
    age_df = df[['Age', 'Embarked', 'Fare', 'Parch', 'SibSp',
                 'Pclass', 'FamilySize',
                 'FsizeD', 'NameLength', "NlengthD", 'Deck']]
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

titanic=fill_missing_age(titanic)
titanic_test=fill_missing_age(titanic_test)


std_scale = preprocessing.StandardScaler().fit(titanic[['Age', 'Fare']])
titanic[['Age', 'Fare']] = std_scale.transform(titanic[['Age', 'Fare']])

std_scale = preprocessing.StandardScaler().fit(titanic_test[['Age', 'Fare']])
titanic_test[['Age', 'Fare']] = std_scale.transform(titanic_test[['Age', 'Fare']])

##================================================================================================
## Linear Regression
##================================================================================================

# # Import the linear regression class
# from sklearn.linear_model import LinearRegression
# # Sklearn also has a helper that makes it easy to do cross validation
# from sklearn.cross_validation import KFold
#
# # The columns we'll use to predict the target
# predictors = ["Pclass", "Sex", "Age","SibSp", "Parch", "Fare",
#               "Embarked","NlengthD", "FsizeD", "Deck"]
# target="Survived"
# # Initialize our algorithm class
# alg = LinearRegression()
#
# # Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# # We set random_state to ensure we get the same splits every time we run this.
# kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
#
# predictions = []
#
# for train, test in kf:
#     # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
#     train_predictors = (titanic[predictors].iloc[train,:])
#     # The target we're using to train the algorithm.
#     train_target = titanic[target].iloc[train]
#     # Training the algorithm using the predictors and target.
#     alg.fit(train_predictors, train_target)
#     # We can now make predictions on the test fold
#     test_predictions = alg.predict(titanic[predictors].iloc[test,:])
#     predictions.append(test_predictions)
#
# predictions = np.concatenate(predictions, axis=0)
# # Map predictions to outcomes (only possible outcomes are 1 and 0)
# predictions[predictions > .5] = 1
# predictions[predictions <=.5] = 0
#
# accuracy=sum(titanic["Survived"]==predictions)/len(titanic["Survived"])
# print(accuracy)
# # 0.80

##================================================================================================
## Logistic Regression
##================================================================================================
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

# predictors = ["Pclass", "Sex", "Fare", "Embarked", "Deck", "Age",
#               "FsizeD", "NlengthD", "Parch"]

predictors = ["Pclass", "Sex", "Fare", "Embarked", "Deck", "Age",
              "FsizeD",  "Parch", 'NlengthD', 'Person']

# Initialize our algorithm
lr = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)

scores = cross_val_score(lr, titanic[predictors], titanic["Survived"],scoring='f1', cv=cv)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
# # 0.750964645324

##================================================================================================
## Random Forest
##================================================================================================
# from sklearn import cross_validation
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import ShuffleSplit
# from sklearn.cross_validation import KFold
#
#
# import numpy as np
# predictors = ["Pclass", "Sex", "Age",
#               "Fare","NlengthD","NameLength", "FsizeD", "Deck"]
#
# # Initialize our algorithm with the default paramters
# # n_estimators is the number of trees we want to make
# # min_samples_split is the minimum number of rows we need to make a split
# # min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
# rf = RandomForestClassifier(random_state=1, n_estimators=100, max_depth=9, min_samples_leaf=4)
# kf = KFold(titanic.shape[0], n_folds=5, random_state=1)
# cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)
#
# predictions = cross_validation.cross_val_predict(rf, titanic[predictors], titanic["Survived"], cv=kf)
# predictions = pd.Series(predictions)
# scores = cross_val_score(rf, titanic[predictors], titanic["Survived"], scoring='f1', cv=kf)
# # Take the mean of the scores (because we have one for each fold)
# print(scores.mean())
# # 0.758539765429

##================================================================================================
## AdaBoost
##================================================================================================
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import ShuffleSplit
# predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked","NlengthD",
#               "FsizeD", "Deck"]
# adb=AdaBoostClassifier()
# adb.fit(titanic[predictors],titanic["Survived"])
# cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)
# scores = cross_val_score(adb, titanic[predictors], titanic["Survived"], scoring='f1',cv=cv)
# print(scores.mean())
# # 0.765245173525
