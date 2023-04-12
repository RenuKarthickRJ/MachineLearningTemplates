#Bayes Theorem
#P(a|b) = (P(b|a)*P(a))/P(b)
#P(a) -> Prior Probability
#P(b) -> Marginal Likelihood
#P(b|a) -> Likelihood
#P(a|b) -> Posterior Probability

#For two categories,
#Apply Bayes theorem for the first condition given the features
#Apply the same for another condition as well
#Compare the probabilities and which ever is higher, is the category it belongs

#What happens if there are more than 2 classes
#In these classes the probability for all the classes must be calculated
#It is done only once when there are 2 classes
#It is a probabilistic classification model

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Getting the dataset
#Input your file name in .csv format
dataset = pd.read_csv("Your_file_name.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state= 0)

#Feature scaling
#Similar to other models
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Performing the Naive Bayes model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
classifier.predict(sc.transform([[]]))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, classifier.predict(x_test))
print(accuracy)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, classifier.predict(x_test))
print(cm)