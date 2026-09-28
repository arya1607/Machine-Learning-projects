import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#------------------------------------------------
#      Function Name = Load Data
#      Description = Load the Data from CSV
#       Input = Name of CSV file
#       output = Data frame
#       Author = Arya Dere
#       Date = 16/08/2026 
#------------------------------------------------
def LoadData(filename):
    df = pd.read_csv(filename)
    
    print("Dataset Loaded Successfully")
    print(df.head())
    
    return df


#------------------------------------------------
#      Function Name = PreprocessData
#      Description   = Perform Data Analysis
#      Input         = None
#      output        = None
#      Author        = Arya Dere
#      Date          = 16/08/2026 
#------------------------------------------------
def PreprocessData(df):
    df = df.drop([
        "Passengerid",
        "zero"
    ],
    errors = "ignore"
    )
    
    #Handle misss]ing value
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    #convert categorical
    df = pd.get_dummies(
        df,
        columns=["Embarked"],
        drop_first= True,
        dtype= int
    )
    
    print(df.head())
    print("Data Preprocessing Completed")
    return df
    
    
    
#------------------------------------------------
#      Function Name = main
#      Description   = Entry Point
#      Input         = None
#      output        = None
#      Author        = Arya Dere
#      Date          = 16/08/2026 
#------------------------------------------------
def main():
    #step 1
    df = LoadData("MarvellousTitanicDataset.csv")
    
    #step 2
    df = PreprocessData(df)

if __name__ == "__main__":
    main()