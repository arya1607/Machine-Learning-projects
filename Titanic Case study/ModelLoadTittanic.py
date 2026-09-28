import pandas as pd
import joblib

def LoadModel(filename):
    model = joblib.load(filename)
    
    print("Model loaded successfully")
    print(model.feature_names_in_)
    
    return model

def Predictpassenger(model):
    print("Enter the Information")
    
    Pclass = int(input("Enter Pclass(1/2/3)"))
    sex = int(input("Enter Sex : (0-M/1-F)"))
    Age = int(input("Enter Age : "))
    sibsp = int(input("Enter Sibsp"))
    Parch = int(input("Enter Patch :"))
    Fare = int(input("Enter Fare"))
    Embarked = float(input("Enter Embarked : (0,1,2)"))
    
    passenger = pd.DataFrame([{
        "Pclass" = Pclass,
        "sex" = sex,
        "Age" = Age,
        "sibsp" = sibsp,
        "Parch" = Parch,
        "Fare" = Fare,
        "Embarked_1.0" : 1 if Embarked == 1 else 0,
        "Embarked_1.0" : 2 if Embarked == 2 else 0
    }])
    
    passenger = passenger[model.feature_names_in_]
    
    
def main():
    model = LoadModel("MarvellousTitanic.pkl")
    
    
if __name__ == "__main__":
    main()