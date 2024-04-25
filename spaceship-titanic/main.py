

import pandas as pd

train_data = './data/train.csv'

def preprocess_data(data):
  df = data
  
  def extract_PassengerId(df):
    df['GroupId'] = df['PassengerId'].apply(lambda x: x.split("_")[0]).astype(int)
    df['PersonId'] = df['PassengerId'].apply(lambda x: x.split("_")[1]).astype(int)
    return df
  
  def extract_Cabin(df):
      df['Deck'] = df[df['Cabin'].notna()]['Cabin'].str.split('/').apply(lambda x: x[0])
      df['Num'] = df[df['Cabin'].notna()]['Cabin'].str.split('/').apply(lambda x: x[1])
      df['Side'] = df[df['Cabin'].notna()]['Cabin'].str.split('/').apply(lambda x: x[2])
      return df
      
  def extract_Name(df):
    df['Last Name'] = df[df['Name'].notna()]['Name'].str.split(' ').apply(lambda x: x[1].strip())
    return df
  
  df = extract_PassengerId(df)
  df.drop(columns=['PassengerId'], inplace=True)
  
  df = extract_Cabin(df)
  df.drop(columns=['Cabin'], inplace=True)
  
  df = extract_Name(df)
  df.drop(columns=['Name'], inplace=True)
  
  return df
  
  
if __name__ == '__main__':
  data = pd.read_csv(train_data)
  df = preprocess_data(data)
  print(df)