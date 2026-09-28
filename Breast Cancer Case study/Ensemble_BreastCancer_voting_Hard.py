import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingClassifier

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

#step 5.1 = create the Indivisual models

model_log = LogisticRegression(max_iter=1000) 

model_det = DecisionTreeClassifier(random_state=42)

model_knn = KNeighborsClassifier(n_neighbors=5)


#step 5.2 = create the voting model

model = VotingClassifier(
    estimators=[
        ('logistic',model_log),         #logistic ha keyword ahe from sklearn main web
        ('decision_tree',model_det),
        ('knn',model_knn)
        ],
    #it passes list
    voting='hard'  #pratekala sepp sep data jato
)

#to do = do .fit and .predict for each model without using ensemble

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