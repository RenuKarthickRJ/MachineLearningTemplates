#Eclat model
#It is an associative rule learning 
#It is similar to apriori model

#We have the supprt factor similar to apriori model
#We only have support in this case
#It is calculated by set contanined in item divided by total number of items

#Steps:
#Take a minimum support 
#Take all the values with the minimum support at the least
#Sort the support by decreasing order

#Install apriori using apyori
#It is similar to apriori
#!pip install apyori

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Getting the dataset
#Input your file name in .csv format
dataset = pd.read_csv("Your_file_name", header = None)
transactions = []
for i in range(0,7501):
  new_list = []
  for j in range(0,20):
    new_list.append(str(dataset.values[i,j]))
  transactions.append(new_list)

  from apyori import apriori
rules = apriori(transactions=transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length =2)

#Store the results
#In this case, the results are stored in the form of a list
results = list(rules)

#All the steps are similar to apriori
#Note that only support will be considered
#Use for loop for better visualisation and understanding of results 
for result in results:
  print(result)