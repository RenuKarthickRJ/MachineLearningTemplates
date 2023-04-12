#Equation for multiple linear regression
#y' = b0 + b1x1 + .....+ bnxn

#For linear regression requirement,
#1) Linearity
#2) Equal Variance
#3) Multivariate Normality
#4) Independence (no one row should affect the next row)
#5) Not predictor should be co-related
#6) The outlier check

#5 methods to build in models
#1) All in 
#2) Backward Elimination
#3) Forward Selection
#4) Bidirectional Elimination
#5) Score Comparision

#1) All in -> Giving all the variables (Only when you have prior knowledge)
# -> Can also prepare for backward elimination
#2) Backward Elimination -> Select a significance level to stay in model (SL)
# -> Fit the model with all the predictors
# -> Consider the predictor with highest predictor (P-value)
# -> Remove the predictor
# -> Fit the model without this predictor
# -> Keep repeating until the P-value is less than SL go to FIN
#3) Forward selection -> Select a SL to enter
# -> Fit all simple regression models. Select one with lowest P-value
# -> Keep the vairable just chosen, now construct all possible variables
# -> If P-value is lower that SL keep repeating else go to FIN
#4) Bidirectional elimination -> Enter the SL level to stay and SL to enter
# -> Perform all forward selection
# -> Perform all backward eliminatino
# -> When no variable can enter or leave, go to FIN
#5) Score Comparision (All Possible models) -> Select criteria of fit
# -> Construct all possible regression models (2^n - 1) total combinations
# -> Select one with best criterion of goodness of fit (eg. Akaike criterion)
# -> Model is ready (Go to fit)

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
#Ensure that you put in your name of the file in .csv format
dataset = pd.read_csv("Name_of_your_file.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#If you have categorical data, perform the following
"""
#Encoding the categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[3])], remainder = "passthrough")
x = np.array(ct.fit_transform(x))
"""

#Splitting the dataset into the training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=0)

#There is no need to apply feature scaling to multiple linear regresion
#This is because the co-efficients will compensate
#Even if data-set does not have linear datasets, you can still try
#Hence there is no need to check the requirements for multiple linear regression

#Training the multiple linear regression model on training set
#Multiple Linear Regression class will automatically avoid one of dummy variable
#The class will also automatically identify the highest P-values
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#you the following to get the result
#Pass in the argument in the form of a 2D array
regressor.predict([[]])