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

#------------------------------------------------
#      Function Name = main
#      Description   = Entry Point
#      Input         = None
#      output        = None
#      Author        = Arya Dere
#      Date          = 16/08/2026 
#------------------------------------------------
def main():
    LoadData("MarvellousTitanicDataset")

if __name__ == "__main__":
    main()