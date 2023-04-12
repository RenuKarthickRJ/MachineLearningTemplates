#SVM
#Line that seperates two sets from two points
#Equidistant from those two points
#Two points are the support vector points
#The line in the middle is called the maximum margin hyperplane (3D)
#Anything to the right is classified as a part of the positive Hyperplane.

#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Getting the dataset
#Input your file name in .csv format
dataset = pd.read_csv("Your_file_name.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Splitting into test and train sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)

#Applying the feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Performing the SVM model
from sklearn.svm import SVC
classifier = SVC(kernel = "linear", random_state = 0)
classifier.fit(x_train,y_train)

#Input the following in a 2D array
classifier.predict(sc.transform([[]]))

#Getting the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, classifier.predict((x_test)))
#The above comes in a 2x2 matrix when it is printed
print (cm)

#Getting the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, classifier.predict((x_test)))
print(accuracy)