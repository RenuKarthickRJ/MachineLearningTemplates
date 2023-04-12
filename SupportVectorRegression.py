#Support vector regression
#Episilon insensitive tube
#Any points inside this tube will not be measured for errors
#Note that a tube will be drawn instead of a line
#Any point outside this line will be drawn
#The points outside the tube are respresented by vectors
#These vectors are called support vectors
#Any hence the name Support Vector Regression

#Non-linear support vector regression
#Importing the libraries
#Same as before
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#The dataset
#put in your file name as the variable in .csv format
dataset = pd.read_csv("Your_file_name.csv")
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Feature Scaling
#Even if there is no explit equation, there is an implicit equation
#Since the range of scales might be less than the range of other variables, 
#Feature scaling must be applied in this case
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
#Standard scaler expects a 2D array as its input
#In this case, y is a 1D array.

#Once feature scaling is completed, you must do fit_transform
x = sc_x.fit_transform(x)
#We won't use both the parameters with the same sc object
#Since there will be a mean and SD for x
#It might confuse with y
y = sc_y.fit_transform(y)

#Training the SVR model
from sklearn.svm import SVR
#The kernel can be a linear or a non-linear type
#The kernel in this case is the radial basis function type
regressor = SVR(kernel = "rbf")
regressor.fit(x,y)

#We have to rearrange such that the feature scaling is transformed
sc_y.inverse_transform(regressor.predict(sc_x.transform([[9]])).reshape(-1,1))
