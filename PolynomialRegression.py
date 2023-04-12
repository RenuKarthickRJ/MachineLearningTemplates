#Polynomial Regression
#Only one vairable ni features
#In multiple linear regression there are more than one parameters in Features
#Used in the cases when the data is not linear
#Similar to simple linear regression
#Used to describle how epidemics and pandemics spread
#y' = b0 + b1x1 + ..... + bnx1^n
#It is called polynomial linear regression
#It is called linear because of the co-efficients

#Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing the dataset
#Input the name of your file in .csv format
dataset = pd.read_csv("Your_file.csv")
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Training the Linear Regression model on dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#Training the polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
#Polynomial Features takes in the degree of the polynomial as input
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

#Since lin_reg_2 is a ploynomial function, we must pass in a polynomial
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

