#Random Forest Regression
#Same algorithm multiple times and put it to get a powerful result
#Ensemble learning
#Pick ar random K data points from Training set
#Build the decision trees associated with these K data points
#Choose the number Ntree of trees you want to build and repeat again
#For a new data point,
# -> Make each one of Ntree trees predict the value of y to data points
# -> Assignthe new data point average across all of the predicted Y values

#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing the dataset
#Ensure to input your file name in .csv format
dataset = pd.read_csv("Your_file_name.csv")
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Training the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=1600)
regressor.fit(x,y)

#Input a 2D array in the following to get a prediction
regressor.predict([[]])