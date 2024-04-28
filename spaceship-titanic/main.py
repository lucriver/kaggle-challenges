import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

# basic modules
import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier

# Feature extraction
def extract_GroupId_PersonId(df):
    df['GroupId'] = df['PassengerId'].apply(lambda x: x.split("_")[0]).astype(int)
    df['PersonId'] = df['PassengerId'].apply(lambda x: x.split("_")[1]).astype(int)
    return df

def extract_Deck_Num_Side(df):
    df['Deck'] = df[df['Cabin'].notna()]['Cabin'].str.split('/').apply(lambda x: x[0])
    df['Num'] = df[df['Cabin'].notna()]['Cabin'].str.split('/').apply(lambda x: x[1])
    df['Side'] = df[df['Cabin'].notna()]['Cabin'].str.split('/').apply(lambda x: x[2])
    return df

def extract_FirstName_LastName(df):
    df['FirstName'] = df[df['Name'].notna()]['Name'].str.split(' ').apply(lambda x: x[0].strip())
    df['LastName'] = df[df['Name'].notna()]['Name'].str.split(' ').apply(lambda x: x[1].strip())
    return df

# Feature Imputation
def impute_HomePlanet_HighConfidence(df):
    def get_groups(df) -> list:
        groups = []
        group_ids = df['GroupId'].unique().tolist()
        for group_id in group_ids:

            # get sub-dataframe based off of group id
            group_df = df[df['GroupId'] == group_id]

            has_missing_planet = group_df['HomePlanet'].isna().any()
            has_one_distinct_home = group_df['HomePlanet'].dropna().nunique() == 1
            has_one_distinct_destination = group_df['Destination'].dropna().nunique() == 1
            has_one_distinct_last_name = group_df['LastName'].dropna().nunique() == 1

            if (
                has_missing_planet and
                has_one_distinct_last_name and
                has_one_distinct_home and
                has_one_distinct_destination        
            ):
                groups.append(group_df)

        return groups

    group_dfs = get_groups(df)
    while group_dfs:
        group_df = group_dfs.pop()
        home_planets = group_df['HomePlanet'].dropna().unique()
        if len(home_planets) > 1:
          raise ValueError(home_planets)
        home_planet = home_planets[0]
        df.loc[group_df['HomePlanet'].isna().index, 'HomePlanet'] = home_planet

    return df
  

train_data = './data/train.csv'
drop_na_train = True

# preprocess the data
df = pd.read_csv(train_data)

# extraction
df = extract_GroupId_PersonId(df)
df = extract_FirstName_LastName(df)
df = extract_Deck_Num_Side(df)

# impututation
df = impute_HomePlanet_HighConfidence(df)

# training block
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck' , 'Side']
continuous_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck' , 'GroupId', 'PersonId', 'Num']
target = 'Transported'

df_train = df[categorical_features + continuous_features + [target]]

if drop_na_train:
    df_train = df_train.dropna()

df_train = pd.get_dummies(df_train, columns=['HomePlanet', 'Destination'])
df_train = pd.get_dummies(df_train, columns=['Deck', 'Side'], drop_first=True)

display(df_train.columns)
                          
X = df_train.drop(columns=['Transported'])
y = df_train[['Transported']]

# scaler = StandardScaler()
# X[continuous_features] = scaler.fit_transform(X[continuous_features])

display(X[continuous_features])
                   
gb_param_grid = {
    'loss': ['log_loss', 'exponential'],
    'n_estimators': [50, 100, 200, 300, 400, 500],
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [2, 3, 4, 5, 6],
}

gridcv = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid=gb_param_grid, n_jobs=32, scoring='accuracy', cv=5, verbose=1)

grid_search = gridcv.fit(X,y.values.ravel())

best_params = grid_search.best_params_
print("Best parameters: ", best_params)
best_score = grid_search.best_score_
print("Best score: ", best_score)
best_estimator = grid_search.best_estimator_
print("Best estimator: ", best_estimator)