import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#---------------------------------------------------------------------------------------

#step 1 = Load the dataset
df = pd.read_csv("breast_cancer.csv")

print("shape of dataset :", df.shape)

print("first Few records : ")
print(df.head())

#---------------------------------------------------------------------------------------


#step 2  = Seprate features and labels
print("-"*40)

X = df.drop("target", axis = 1)
Y = df["target"]

print("Shape of Data : ")
print("X shape: ", X.shape)
print("Y shape: ", Y.shape)

#---------------------------------------------------------------------------------------

#step 3  = spit dataset for training and testing dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.2)

#---------------------------------------------------------------------------------------

#step 4 = scale the features 
scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

#---------------------------------------------------------------------------------------

#step 5.1 = create the base model
base_model = DecisionTreeClassifier(random_state=42)

#step 5.2 = create the Bagging model
model = BaggingClassifier(
    estimator=base_model,   #it just 1 model
    n_estimators=10,        #10 means it will create 10 decision trees
    random_state=42
)

#---------------------------------------------------------------------------------------

#step 6 = train the model
model = model.fit(X_train,Y_train)

#---------------------------------------------------------------------------------------

#step 7 = test the data
Y_pred = model.predict(X_test)

#---------------------------------------------------------------------------------------

#Step 8 = evaluate the model 
print("-"*40)
print("Evaluation of model : ")

print("Accuracy : ",accuracy_score(Y_test,Y_pred))
print("Confusion Matrix : ")
print(confusion_matrix(Y_test,Y_pred))