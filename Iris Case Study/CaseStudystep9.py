import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import (
  accuracy_score,
  confusion_matrix,
  classification_report,
  ConfusionMatrixDisplay
)

Border = "-"* 40

##################################
#Step 1: Load the dataset
##################################

print(Border)
print("Step 1: Load the dataset")
print(Border)

DataPath = "iris.csv"

df = pd.read_csv(DataPath) 

'''
            pandas 
              |
     --------------------
     |        |         |
   Series   DataFrame  Panel
   1D        2D       3D       #Array  
   
'''


print("Dataset loaded successfully!")
print("Initial entries from the dataset are : ")

print(df.head())


##################################
#Step 2: Data Analysis(EDA)
##################################

print(Border)
print(Border)
print("Step 2: Data Analysis(EDA)")
print(Border)
print(Border)


#(150, 5)  # 150 rows and 5 columns
print("shape of the dataset is : ", df.shape)

print(Border)

#list of columns names
print("Columns names : ", list(df.columns))

print(Border)

# outlier = objection of EDA is to find outliers in the dataset. 
# Outliers are those values which are far away from the other values in the dataset.

#missing values in the dataset
print("Missing values per column : ")
print(df.isnull().sum())

print(Border)

print("Class distribution (species count)")
print(df["species"].value_counts())

print(Border)

print("Statistical report of the dataset : ")
print(df.describe())

##################################
#Step 3: Decide Independant and Dependant variables
##################################

print(Border)
print(Border)
print("Step 3: Decide Independant and Dependant variables")
print(Border)
print(Border)

#X : Independant Variable / features
#Y : Dependant Variable / features

feature_cols = [
  "sepal length (cm)",
  "sepal width (cm)",
  "petal length (cm)",
  "petal width (cm)"
  ]

X = df[feature_cols]
Y = df["species"]

print(Border)
print("X Shape : ",X.shape)
print("Y Shape : ",Y.shape)
print(Border)

##################################
#Step 4: Visualisation of dataset
##################################

print(Border)
print(Border)
print("Step 4: Visualisation of dataset ")
print(Border)
print(Border)

#scatterplot
plt.figure(figsize = (7,5))

for sp in df["species"].unique():
  temp = df[df["species"] == sp]
  plt.scatter(temp["petal length (cm)"], temp["petal width (cm)"], label = sp)
  
plt.title("Marvellous Iris Case study")

plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")

plt.legend()
plt.grid()
plt.show()

##################################
#Step 5: Split thew dataset
##################################

print(Border)
print(Border)
print("Step 5: Split thew dataset")
print(Border)
print(Border)

X_train , X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.5, random_state=42)
print("Dataset spliting Actiivity Done")

print(Border)
print("X_train :", X_train.shape)  #(150,4)
print("X_test :", X_test.shape)     #(150, 1)

print("Y_train :", Y_train.shape)
print("Y_test :", Y_test.shape)
print(Border)


##################################
#Step 6: Build the model
##################################

print(Border)
print(Border)
print("Step 6: Build the model")
print(Border)
print(Border)

model = DecisionTreeClassifier(max_depth = 5)

print("Model gets created Successfully")

##################################
#Step 7: Train the model
##################################

print(Border)
print(Border)
print("Step 7: Train the model")
print(Border)
print(Border)

model.fit(X_train, Y_train)
print("Model trained Successfully")

##################################
#Step 8: Evalue the model
##################################

print(Border)
print(Border)
print("Step 7: Test the model")
print(Border)
print(Border)

Y_pred = model.predict(X_test)

print("model testing done")

print("Expected answers: ")
print(Y_test)

print("predicted ansers: ")
print(Y_pred)

##################################
#Step 9: Evaluate the model performance
##################################


print(Border)
print("Step 9: Evaluate the model performance")
print(Border)

accuracy = accuracy_score(Y_test,Y_pred)
print("Accuracy of model is : ", accuracy*100)

print(Border)
print("confusion_matrix")
cm = confusion_matrix(Y_test,Y_pred)
print(cm)

print(Border)
print("classification report")
print(classification_report(Y_test,Y_pred))
