import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def preprocess(df):
    
    try:
        df['Survived'] = df['Survived'].map({0: False, 1: True})
    except:
        print("Failed to handle survived, likely test set?")
    
    df['Pclass'] = df['Pclass'].astype('category').cat.codes

    df = df.rename(columns={'Sex': 'Male'})
    df['Male'] = df['Male'].map({'male': True, 'female':False})
    
    df.loc[df['Cabin'].notna(), 'HasCabin'] = True
    df.loc[df['Cabin'].isna(), 'HasCabin'] = False
    df['HasCabin'] = df['HasCabin'].astype('bool')
    
    df['Embarked'] = df['Embarked'].fillna('S')
    
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    df['Title'] = df['Title'].replace(['Mlle', 'Ms', 'Jonkheer', 'Lady'], 'Miss')
    df['Title'] = df['Title'].replace(['Mme', 'Dona', 'the Countess'], 'Mrs')
    df['Title'] = df['Title'].replace(['Don', 'Sir'], 'Mr')
    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Dr', 'Rev', 'Major'], 'Others')
    
    ages = df.groupby(['Title'])['Age'].transform('median')
    df['Age'] = df['Age'].fillna(ages)
    
    fares = df.groupby(['Pclass', 'Title'])['Fare'].transform('median')
    df['Fare'] = df['Fare'].fillna(fares)
    df.loc[df['Fare'] <= 17, 'BinnedFare'] = 0
    df.loc[(df['Fare'] > 17) & (df['Fare'] <= 30), 'BinnedFare'] = 1
    df.loc[(df['Fare'] > 30) & (df['Fare'] <= 100), 'BinnedFare'] = 2
    df.loc[df['Fare'] >= 100, 'BinnedFare'] = 3
    
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['HasFamily'] = df['FamilySize'] > 0
    
    return df
    

# process training data
df = pd.read_csv('./data/train.csv')

df = preprocess(df)

one_hot = ['Embarked', 'Title', 'Pclass', 'BinnedFare']
other = ['Male', 'HasCabin', 'Age', 'HasFamily']

df_one_hot = pd.get_dummies(df[one_hot], columns=one_hot)
X = df_one_hot.join(df[other + ['Survived']])
y = X['Survived']
X = X.drop(columns=['Survived'])

print(X.columns)

params = {
    'n_estimators': [200, 300, 400, 500, 600, 700, 800],
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3,5,7],
    'subsample': [0.5, 0.7, 1]
}

gridcv = GridSearchCV(XGBClassifier(), cv=5, n_jobs=-1, param_grid=params)
gridcv.fit(X,y)

print(gridcv.best_score_)

model = gridcv.best_estimator_
model.fit(X,y)

df_test = pd.read_csv('./data/test.csv')
df_test = preprocess(df_test)

df_one_hot = pd.get_dummies(df_test[one_hot], columns=one_hot)
X = df_one_hot.join(df[other])

print(X.columns)
print(X.isna().sum())

df_test['Survived'] = model.predict(X)

df_test[['PassengerId','Survived']].to_csv("./data/my_submission.csv", index=False)

