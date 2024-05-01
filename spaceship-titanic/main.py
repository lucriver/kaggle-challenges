import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

# basic modules
import pandas as pd
import numpy as np
import seaborn as sns
from IPython.display import display
import matplotlib.pyplot as plt

import random
from scipy.stats import chi2_contingency
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
from xgboost import XGBClassifier

# Feature extraction
def feature_extract_GroupId_PersonId(df):
    df["GroupId"] = df["PassengerId"].apply(lambda x: x.split("_")[0]).astype(int)
    df["PersonId"] = df["PassengerId"].apply(lambda x: x.split("_")[1]).astype(int)
    return df

def feature_extract_Deck_Num_Side(df):
    df["Deck"] = df[df["Cabin"].notna()]["Cabin"].str.split("/").apply(lambda x: x[0])
    df["Num"] = df[df["Cabin"].notna()]["Cabin"].str.split("/").apply(lambda x: x[1])
    df["Side"] = df[df["Cabin"].notna()]["Cabin"].str.split("/").apply(lambda x: x[2])
    return df

def feature_extract_FirstName_LastName(df):
    df["FirstName"] = df[df["Name"].notna()]["Name"].str.split(" ").apply(lambda x: x[0].strip())
    df["LastName"] = df[df["Name"].notna()]["Name"].str.split(" ").apply(lambda x: x[1].strip())
    return df

# Feature Imputation
def feature_impute_HomePlanet_HighConfidence(df):
    def get_groups(df) -> list:
        groups = []
        group_ids = df["GroupId"].unique().tolist()
        for group_id in group_ids:

            # get sub-dataframe based off of group id
            group_df = df[df["GroupId"] == group_id]

            has_missing_planet = group_df["HomePlanet"].isna().any()
            has_one_distinct_home = group_df["HomePlanet"].dropna().nunique() == 1
            has_one_distinct_destination = group_df["Destination"].dropna().nunique() == 1
            has_one_distinct_last_name = group_df["LastName"].dropna().nunique() == 1

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
        home_planets = group_df["HomePlanet"].dropna().unique()
        if len(home_planets) > 1:
          raise ValueError(home_planets)
        home_planet = home_planets[0]
        df.loc[group_df["HomePlanet"].isna().index, "HomePlanet"] = home_planet

    return df

def feature_impute_HomePlanet_MediumConfidence(df):
    def get_groups(df):
        groups =[]
        group_ids = df["GroupId"].unique().tolist()
        for group_id in group_ids:
            group_df = df[df["GroupId"] == group_id]

            missing_home_planet = group_df["HomePlanet"].isna().any()
            one_distinct_last_name = group_df["LastName"].dropna().nunique() == 1
            one_distinct_home_planet = group_df["HomePlanet"].dropna().nunique() == 1

            if (
                missing_home_planet and
                one_distinct_last_name and
                one_distinct_home_planet
            ):
                groups.append(group_df)

        return groups

    group_dfs = get_groups(df)

    while group_dfs:
        group_df = group_dfs.pop()

        planets = group_df["HomePlanet"].dropna().unique().tolist()

        if len(planets) != 1:
            raise ValueError("HUH")

        df.loc[group_df["HomePlanet"].isna().index, "HomePlanet"] = planets[0]    
        
    return df

def feature_impute_Destination_LowConfidence(df):
    def get_groups(df):
        groups = []
        group_ids = df["GroupId"].unique().tolist()

        for group_id in group_ids:

            group_df = df[df["GroupId"] == group_id]

            at_least_one_missing_destination_planet = group_df["Destination"].isna().any()
            only_one_distinct_destination_planet = group_df["Destination"].dropna().nunique() == 1    
            only_one_distinct_home_planet = group_df["HomePlanet"].dropna().nunique() == 1
            only_one_distinct_last_name = group_df["LastName"].dropna().nunique() == 1

            if (
                at_least_one_missing_destination_planet and
                only_one_distinct_home_planet and
                only_one_distinct_destination_planet and
                only_one_distinct_last_name
            ):
                groups.append(group_df)

        return groups

    groups = get_groups(df)
    while groups:
        group_df = groups.pop()
        destination_planets = group_df["Destination"].dropna().unique().tolist()
        if len(destination_planets) != 1:
            raise ValueError(len(destination_planets))
        df.loc[group_df["Destination"].isna().index, "Destination"] = destination_planets[0]
    
    return df

def feature_impute_CryoSleep_HighConfidence(df):
    spending_money_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    mask = df[(df["CryoSleep"].isna()) & (df[spending_money_cols].sum(axis=1) > 0.0)].index
    df.loc[mask, "CryoSleep"] = False
    return df
    
def feature_impute_SpendingMoneyColumns_MediumConfidence(df):
    spending_money_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    cond_1 = (df["CryoSleep"] == True)
    cond_2 = (df[spending_money_cols].sum(axis=1) == 0.0)
    cond_3 = (df[spending_money_cols].isna().any(axis=1))
    mask = df[cond_1 & cond_2 & cond_3].index
    df.loc[mask, spending_money_cols] = 0.0
    return df


def feature_create_TotalSpent_Strict(df):
    spending_money_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    cond_1 = (df[spending_money_cols].notna().all(axis=1))
    mask = df.loc[cond_1].index
    df.loc[mask, "TotalSpent"] = df.loc[mask][spending_money_cols].sum(axis=1)
    return df

def feature_create_TotalSpent(df):
    spending_money_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpent"] = df[spending_money_cols].sum(axis=1)
    return df


def optimal_preprocessing(df):
    # extract features
    df = feature_extract_GroupId_PersonId(df)
    df = feature_extract_FirstName_LastName(df)
    df = feature_extract_Deck_Num_Side(df)
    
    # imputation
    df = feature_impute_HomePlanet_HighConfidence(df)
    df = feature_impute_HomePlanet_MediumConfidence(df)
    df = feature_impute_Destination_LowConfidence(df)
    df = feature_impute_CryoSleep_HighConfidence(df)
    # df = feature_impute_SpendingMoneyColumns_MediumConfidence(df)    

    # create features
    df = feature_create_TotalSpent(df) # OR feature_create_TotalSpent_Strict(df)
    
    return df
    
train_data = "./data/train.csv"
test_model = True

# preprocess the data
df = pd.read_csv(train_data)
df = optimal_preprocessing(df)

# training block
categorical_features = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck" , "Side"]
continuous_features = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "GroupId", "Num", "PersonId" ]

engineered_continuous = ["TotalSpent"]
continuous_features = continuous_features + engineered_continuous

# engineered_categorical = []
# categorical_features = categorical_features + engineered_categorical

target = "Transported"

df_train = df[categorical_features + continuous_features + [target]]

drop_cols = [
    "GroupId",
    "Num",
    "PersonId",
]

if drop_cols:
    df_train = df_train.drop(columns=drop_cols)
    
df_train["VIP"] = df_train["VIP"].astype(pd.BooleanDtype())
df_train["CryoSleep"] = df_train["CryoSleep"].astype(pd.BooleanDtype())
df_train = pd.get_dummies(df_train, columns=["HomePlanet", "Destination"])
df_train = pd.get_dummies(df_train, columns=["Deck"], drop_first=False)  
df_train = pd.get_dummies(df_train, columns=["Side"], drop_first=True)   

X = df_train.drop(columns=["Transported"])
y = df_train[["Transported"]]

display(X.columns)

# TRAINING RUN
param_grid = {
    "n_estimators": [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
    "learning_rate": [0.1, 0.01, 0.001],
    "max_depth": [2, 3, 4, 5],
}
gridcv = GridSearchCV(XGBClassifier(n_jobs=-1), param_grid=param_grid, n_jobs=-1, scoring="accuracy", cv=5, verbose=1)


grid_search = gridcv.fit(X,y.values.ravel())

best_params = grid_search.best_params_
print("Best parameters: ", best_params)
best_score = grid_search.best_score_
print("Best score: ", best_score)
best_estimator = grid_search.best_estimator_
print("Best estimator: ", best_estimator)

model = gridcv.best_estimator_
    
# TEST MODEL ON KAGGLE SET
if test_model:
    df_test = pd.read_csv("./data/test.csv")
    df_test = optimal_preprocessing(df_test)

    df_test["VIP"] = df_test["VIP"].astype(pd.BooleanDtype())
    df_test["CryoSleep"] = df_test["CryoSleep"].astype(pd.BooleanDtype())
    
    categorical_features = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck" , "Side"]
    continuous_features = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "TotalSpent"]

    X = df_test[categorical_features + continuous_features]
    
    X = pd.get_dummies(X, columns=["HomePlanet", "Destination"])
    X = pd.get_dummies(X, columns=["Deck"], drop_first=False)  
    X = pd.get_dummies(X, columns=["Side"], drop_first=True)   
    
    y = model.predict(X)
    
    df_test["Transported"] = y.astype("bool")
    
    fname = str(random.randint(0,100)) + ".csv"
    
    df_test[["PassengerId","Transported"]].to_csv(f"./data/{fname}", index=False)
    
