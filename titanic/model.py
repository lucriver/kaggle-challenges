import os
import argparse

import pandas as pd
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np

os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

class TitanicModel:
  def __init__(self):
    self.age_transformer = PowerTransformer('box-cox')
    self.fare_transformer = PowerTransformer()
    self.parent_scaler = MinMaxScaler()
    self.sibling_scaler = MinMaxScaler()
    self.married_female_titles = ['Mrs.', 'Mme.', 'Lady.', 'Countess.']
    self.unmarried_female_titles = ['Miss.', 'Ms.', 'Mlle.', 'Jonkheer.']
    
    self.embarked_mapping = {
      'S': 0,
      'C': 1,
      'Q': 2
    }
    
    self.classifier = None
  
  def fit(self, file: str):
    df = pd.read_csv(file)
    
    # categorically encode class column    
    df['Pclass'] = df['Pclass'].astype('category').cat.codes

    # one-hot encode sex column
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

    # disregard missing embarked values and map them
    df = df[df['Embarked'].notna()]
    df['Embarked'] = df['Embarked'].map(self.embarked_mapping)
    
    # # NAMES HERE
    df.drop(columns=['Name'], inplace=True)
    
    # impute missing ages and normalize
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    self.age_transformer.fit(df[['Age']].values)
    df['Age'] = self.age_transformer.transform(df[['Age']].values)
    
    # trim fare outliers
    fares = df['Fare'].values
    fare_std = np.std(fares)
    cut_off = fare_std * 4
    df = df[df['Fare'] <= cut_off]
    self.fare_transformer.fit(df[['Fare']].values)
    df['Fare'] = self.fare_transformer.transform(df[['Fare']].values)

    # CABINS here
    df.drop(columns=['Cabin'], inplace=True)

    # Handle siblings
    sibling_max = 5
    df = df[df['SibSp'] <= sibling_max]
    self.sibling_scaler.fit(df[['SibSp']])
    df['SibSp'] = self.sibling_scaler.transform(df[['SibSp']])
    
    # Handle parents
    parents_max = 5
    df = df[df['Parch'] <= parents_max]
    self.parent_scaler.fit(df[['Parch']].values)
    df['Parch'] = self.parent_scaler.transform(df[['Parch']].values)
    
    # HANDLE TICKETS
    df.drop(columns='Ticket',inplace=True)
    
    X = df.drop(columns=['PassengerId', 'Survived'])
    y = df[['Survived']]

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 20, 30],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 22, 24, 26, 28],
        'min_samples_leaf': [1, 2, 3, 4, 5, 10]
    }

     
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=14)
    grid_search.fit(X,y)
      
    # Best parameters and best score
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
      
    # define the model
    self.model = DecisionTreeClassifier(**grid_search.best_params_)
    self.model.fit(X,y)
    
  def eval(self, file):
    df = pd.read_csv(file)

    df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
    
    df['Pclass'] = df['Pclass'].astype('category').cat.codes
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Age'] = self.age_transformer.transform(df[['Age']])
    df['SibSp'] = self.sibling_scaler.transform(df[['SibSp']].values)
    df['Parch'] = self.parent_scaler.transform(df[['Parch']].values)
    df['Fare'] = self.fare_transformer.transform(df[['Fare']].values)
    df['Embarked'] = df['Embarked'].map(self.embarked_mapping)
    
    X = df.drop(columns=['PassengerId'])  
    df['Survived'] = self.model.predict(X)
    
    submission_df = df[['PassengerId', 'Survived']]
    submission_df.to_csv('./data/my_submission.csv', index=False)

  
parser = argparse.ArgumentParser()
parser.add_argument('--train-file', type=str, help='path to training data file')
parser.add_argument('--test-file', type=str, help='Path to test submission file')

if __name__ == "__main__":
  args = parser.parse_args()
  
  train_file_path = args.train_file
   
  if not train_file_path or not os.path.exists(train_file_path):
    raise ValueError(f'Inavlid file path: {train_file_path}')
  
  model = TitanicModel()
  model.fit(train_file_path)
  
  test_file_path = args.test_file
  
  if test_file_path:
    model.eval(test_file_path)