#Apriori technique
#People who did something also did something else

#It has three parts
#Support
#Confidence 
#Lift

#Step1:
#Set a minimum support and confidence
#Step2:
#Take all the subsets that are higher than the minimum support
#Step3:
#Take all the subsets that have higher confidence than minimum confidence
#Step4:
#Sort the rules by decreasing the fit

#Install apriori first using apyori
#!pip install apyori

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Getting the dataset
#Your file name in .csv format
dataset = pd.read_csv("Your_file_name.csv", header=None)
transactions = []
#We must convert to string because apriori model only takes in strings
for i in range(0,7501):
  transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

#Training the apriori model
from apyori import apriori
#min_length and max_length are for how many products you want on the left and right side (say buy one product and get one product so total is two)
#If buy one get ten, min_length will be 2 and max_length will be 11
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length =2)

results = list(rules)
#The above will store the results in a list datatype
#Make sure to print it using a for loop for better readability
for result in results:
  print(result)