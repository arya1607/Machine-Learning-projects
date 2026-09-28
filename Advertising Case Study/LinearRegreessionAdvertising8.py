import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def MarvellousRegression(DataPath):
    
    Border = "-"*60
    
#step 1 : Load the data
    
    print(Border)
    print("Step 1 : Load the Data")
    print(Border)
        
    df = pd.read_csv(DataPath)
    print(df.head())    

#step 2 : Remove Unwanted Column

    print(Border)
    print("Step 2 : Remove Unwanted Column ")
    print(Border)
    
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        
    print(df.head())
    
#step 3 : Check Missing Value

    print(Border)
    print("step 3 : Check Missing Value")
    print(Border)
    
    print("Total missing Values : ")
    
    print(Border)
    print(df.isnull().sum())
    
#step 4 : Statistical summary

    print(Border)
    print("step 4 : Statistical summary")
    print(Border)

    print(df.describe())
    
#step 5 : Correlation

    print(Border)
    print("step 5 : Correlation")
    print(Border)
    
    print(df.corr())


#step 6 : separate independant and dependent variables

    print(Border)
    print("step 5 : Correlation")
    print(Border)
    
    X = df[["TV","radio","newspaper"]]
    Y = df["sales"]
    
    print("Independent Variables :")
    print(X.head())
    
    print(Border)
    
    print("Dependant variables : ")
    print(Y.head())
    
#step 7 : Split the dataset
    print(Border)
    print("step 7 : Split the dataset")
    print(Border)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size = 0.2,
        random_state = 42
    )
    
    print("Training Data : ",X_train.shape)
    print("Testing Data : ",X_test.shape)
    
#step 8 = create and train the model 
    print(Border)
    print("step 8: Create and train the model")
    print(Border)

    model = LinearRegression()
    
    model = model.fit(X_train,Y_train)
    print("Model trained successfully")
    
#step 9 = Test the model 
    print(Border)
    print("step 9 = Test the model")
    print(Border)
    
    Y_pred = model.predict(X_test)
    
    print("Expected Answers: ")
    print(Y_test[:5])

    print(Border)
    
    print("Predicted Answers: ")
    print(Y_pred[:5])

#step 10 = Evaluate the model 
    print(Border)
    print("step 10 = Evaluate the model")
    print(Border)

    MSE = mean_squared_error(Y_test,Y_pred)
    
    RMSE = np.sqrt(MSE)
    
    R2 = r2_score(Y_test,Y_pred)
    
    print("MSE : ",MSE)
    print("RMSE : ",RMSE)
    print("R2 : ",R2)

#step 11 : Display Coefficent
    print(Border)
    print("step 11 : Display Coefficent")
    print(Border)    
    
    print("TV coefficents :",model.coef_[0])
    print("Radio coefficents :",model.coef_[1])
    print("Newspaper coefficents :",model.coef_[2])
    
    print("Intercept : ",model.intercept_)

def main():
    MarvellousRegression("Advertising.csv")

if __name__ == "__main__" : 
    main()