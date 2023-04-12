#Random Forest Classifier
#Ensemble Learning
#Meany algorithm into one algorithm
#Random Forest combines multiple Decision Tree algorithm
#A number of Decision tree algorithms are run in this case

#Pick at random K number of data points from the training set
#Build the decision tree associated to these data points
#Choose the n number of trees you want to build and repeat the above steps
#For a new data point you make, each N trees predicts the category the data 
#-> point belongs to and assigns the data point to the majority tree (with vote)

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Getting the dataset
#Input your file name in the following
dataset = pd.read_csv("Your_file_name.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Splitting the data into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Applying the Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
classifier.fit(x_train,y_train)
classifier.predict([[]])

#Getting the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, classifier.predict(x_test))
print(accuracy)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, classifier.predict(x_test))
print(cm)