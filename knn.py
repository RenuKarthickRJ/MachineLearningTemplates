#KNN
#Choose the number K of neighbours
#Take the K nearest neighbours of a point according to Eucleduan
#Count the number of data points in each category
#Assign new data point to the category where there are most neighbours 

#importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Getting the dataset
#Input your file name in .csv format
dataset = pd.read_csv("Your_file_name.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Splitting into test and train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

#Training the KNN model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=4, p=2, metric="minkowski")
classifier.fit(x_train,y_train)
classifier.predict([[]])

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,classifier.predict(x_test))
print(cm)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,classifier.predict(x_test))
print(accuracy)
