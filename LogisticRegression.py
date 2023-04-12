#Logistic Regression is used to predict a categorical dependent variable
#It predicts this from a number of independent variables
#The best sigmoid curve is calculated by using maximum likelihood
#Likelihood is calculated by multiplying the probability of all possible options
#The curve with the maximum likelihood will be considered

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#Import the dataset
#Input your file name in the following as the variable
dataset = pd.read_csv("Your_file_name.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Splitting into test and train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Training the logistic regression model on training set x_train
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)
classifier.predict(sc.transform([[]))
y_pred = classifier.predict(x_test)

#Confusion matrix will show how many correct predictions were made
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Getting the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)