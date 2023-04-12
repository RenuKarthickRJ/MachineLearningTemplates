#Two types of Hierarchical clustering 
#Agglomerative & Divisive
#Agglomerative is bottom to top approach

#Make n number of clusters (n -> data points)
#Make 2 closest points cluster
#Then 3, 4 and so on upto there is only one cluster
#Closest clusters is an important key word
#Eucledian distance is used to measure the distance between two points
#Distance between two clusters when there are more than one point in each,
#There can be the following ways,
#1) Closest points distances
#2) Farthest points distances
#3) Average of all points distances
#4) Centroid distances
#It is upto us to select the required form

#The hierarchical cluster maintains memory on how it forms one single cluster
#Dendrograms play an important role in this case
#The height of dendrogram is the distance between two points in actual plot
#You can set a threshold at 1.7,
#It wont allow the disimilarities above that point
#Dendrograms help us in this case

#Standard approach is to look for the highest line in a dendrogram
#We can cut at that horizantal line and the number of lines it cuts is n
#In this case, n will be the number of clusters
#Note that the longest should not have horizontal lines in the middle
#This means, on one side, it can be large while on other it can be small
#It this case, it will not be optimal when considering to cut that line

#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Getting the dataset
#Input the name of your file in .csv format
dataset = pd.read_csv("Your_File_Name.csv")
#Input the value of x columns according to the dataset
x = dataset.iloc[:,[3,4]].values

#Using the dendrogram to find the number of clusters
import scipy.cluster.hierarchy as sch
#Ward is minimum variance method
#It is used to minimize the variance between the clusters
dendogram = sch.dendrogram(sch.linkage(x, method = "ward"))

#Training the hierarchical clustering model
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean")
cluster.fit_predict(x)

#If you want to predict for a new point, add it to the cluster and predict as the following
new_point = [30,45]
X_new = np.vstack([x, new_point])
cluster.fit_predict(X_new)