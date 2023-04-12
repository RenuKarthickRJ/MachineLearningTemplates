#Kernel SVM
#Used in cases when a line is complicated to be drawn
#Used in cases when the data is not linearly seperable
#Usually non-linear data can be dealt with adding an extra dimension
#The non-liner data will be mapped to a higher dimension
#But this can be compute-intensive
#Hence Kernels are used
#The Gaussian radial basis function kernal is vital for this case

#The sigma in the RBF decides on how wide the circumference is
#Hence proper attention must be paid to select the sigma value
#There can also be more than two kernels or even a single kernel
#The number of kernels is decided based on the application
#It might give more accuracy than SVM
#Depending on the dataset

#Types of Kernel functions
#RBF
#Sigmoid Kernel (function in this case is directional (either right or left))
#Polynomial Kernel

#Non-linear SVR
#RBF can be used as well
#In 3D SVR there will be two planes one above and below the actual one
#Any points now between them will not be considered for errors
#The remaining points outside will be considered fro support vectors

#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Getting the dataset and
#Splitting the dataset into tranining and test sets
#Input Your File Name in .csv format
dataset = pd.read_csv("Your_File_Name.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Splitting is similar to all the models till updated
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Training the model
from sklearn.svm import SVC
classifier = SVC(kernel = "rbf", random_state = 0)
classifier.fit(x_train,y_train)
classifier.predict(sc.transform([[]]))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, classifier.predict(x_test))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, classifier.predict(x_test))
