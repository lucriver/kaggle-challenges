import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

import re
import pandas as pd
import numpy as np
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

def feature_extract_CabinLetter(df):
    def get_cabin_letter(x):
        if type(x) == float:
            return np.NaN

        if x[:2] == 'F ':
            x = x[2:]

        return x[0]
    
    df['CabinLetter'] = df['Cabin'].apply(get_cabin_letter)
    return df


def feature_extract_CabinCount(df):
    def get_cabin_count(x):
        if type(x) == float:
            return x
        if 'F ' in x:
            x = x[2:]
        s = [char for char in x.lower() if bool(re.search(r'[a-zA-Z]', char))]
        return len(s)
    
    
    df['CabinCount'] = df['Cabin'].apply(get_cabin_count)
    return df
    
def feature_extract_LastName(df):
    df['LastName'] = df['Name'].apply(lambda x: x.split(",")[0])
    return df

def feature_extract_Title_TitleEncoded(df):
    df['Title'] = df['Name'].str.extract(r',\s*([a-zA-Z]+.)', expand=True)    

    def get_title(title):
        # Define subsets
        male_titles = ['Mr.', 'Don.', 'Sir.']
        male_young = ['Master.']
        female_titles = ['Mrs.', 'Mme.', 'Ms.', 'Mlle.', 'Lady.', 'the ']
        female_young = ['Jonkheer.', 'Miss.']
        navy = ['Col.', 'Capt.', 'Major.']
        profession = ['Rev.', 'Dr.']

        # Categorize titles
        if title in male_titles:
            return 'Male Titles'
        elif title in male_young:
            return 'Male Young'
        elif title in female_titles:
            return 'Female Ordinary'
        elif title in female_young:
            return 'Female Young'
        elif title in navy:
            return 'Navy Title'
        elif title in profession:
            return 'Profession Title'
        else:
            return np.NaN
        
    df['TitleEncoded'] = df['Title'].apply(get_title)
    
    return df

def feature_impute_age(df, metric='median'):
    consideration_cols = ['Pclass','TitleEncoded', 'Sex', 'Fare']
    if metric.lower() == 'median':
        ages = df.groupby(consideration_cols)['Age'].transform('median')
    elif metric.lower() == 'mean':
        ages = df.groupby(consideration_cols)['Age'].transform('mean')
    else:
        raise ArgumentError("NO")
        
    df['Age'] = df['Age'].fillna(ages)
        
    return df
    
    
def preprocess(df):
    df = feature_extract_CabinCount(df)
    df = feature_extract_CabinLetter(df)
    df = feature_extract_Title_TitleEncoded(df)
    df = feature_impute_age(df)
    return df

df = pd.read_csv('./data/train.csv')
df = preprocess(df)


categorical_features = ['Pclass', 'Sex', 'Embarked', 'CabinLetter', 'TitleEncoded']
continuous_features = ['SibSp', 'Parch', 'Age', 'Fare']
target_feature = 'Survived'
     
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.set_output(transform='pandas')
encoder.fit(df[categorical_features].dropna())
df_categorical = encoder.transform(df[categorical_features])

q_low = df['Fare'].quantile(0.01)
q_high = df['Fare'].quantile(0.99)
df['Fare'] = df['Fare'].clip(lower=q_low, upper=q_high)

X = df_categorical.join(df[continuous_features + [target_feature]])
X = X.drop(columns=['Survived'])
y = df[target_feature]

print(X.columns)

xgboost_params_grid = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800],
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9],
    'subsample': [0.6, 0.8, 1.0],
}

gridcv = GridSearchCV(XGBClassifier(n_jobs=-1), param_grid=xgboost_params_grid, n_jobs=-1, cv=2)
gridcv.fit(X,y.values.ravel())

print(f"Best Score: {gridcv.best_score_}")
print(f"Best Model: {gridcv.best_estimator_}")

model = gridcv.best_estimator_
model.fit(X,y.values.ravel())

test = True
if test:
    df = pd.read_csv('./data/test.csv')

    df = preprocess(df)
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'CabinLetter', 'TitleEncoded']
    continuous_features = ['SibSp', 'Parch', 'Age', 'Fare']
    
    df_test_categorical = encoder.transform(df[categorical_features])
    df_test_continuous = df[continuous_features]
    
    X = df_test_categorical.join(df_test_continuous)
    
    print(len(X))
    print(X.columns)
    
    df['Survived'] = model.predict(X)
    df[['PassengerId','Survived']].to_csv('./data/my_submission.csv',index=False)
    
    

