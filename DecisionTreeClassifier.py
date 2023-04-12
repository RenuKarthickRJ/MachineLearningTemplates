#Decision Trees
#CART - Classification And Regression Trees
#Classification trees helps classify data
#Regression trees help predict the data
#These tree will be further upgraded to Random Forest classifiers
#Just because Random Forest classifiers are an upgrade, does not mean that the predictions will be better

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Getting the dataset
#Your file name in .csv format goes as the input in the following
dataset = pd.read_csv("Your_file_name")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Applyting the Decision Tree model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "entropy")
#Criterion is a quality of measure of the split
classifier.fit(x_train,y_train)
classifier.predict([[]])

#Getting the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, classifier.predict(x_test))

#Getting the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, classifier.predict(x_test))
print(cm)
print(accuracy)