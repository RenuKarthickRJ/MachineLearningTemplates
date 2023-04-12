#Decide how many clusers you want in the data

#Place random centroid in the data
#Calculate the mass/centre of gravity for the data
#The centriods are moved to the centre of mass
#Again the same process is repeated
#Repeated until there can be no more changes to take place

#The elbow method
#If prior information on number of clusters is not available, look for elbow
#WCSS -> It looks at distance between each point and the centre of clusters
#-> square the distance and sum them up
#We do kmeans many times then find WCSS
#The more clusters we have, the smaller the WCSS
#When the WCSS is plotted against clusters, it comes to a shape like an elbow
#The point at the kink is considered as the number of clusters

#Kmeans ++ 
#When more than two clusters, the centriod initial position plays a huge role
#Some time you might fall into the random initilization trap
#Kmeeans++ combats the random initialization trap
#Choose the centrod at random
#For each remaining data, compute distance to the nearest centriod
#Choose next centriod using weighted random selection based on distance
#Repeated the above two steps until all centriods are ready

#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Getting the dataset
#Input your file name in the following
dataset = pd.read_csv("Your_File_name.csv")
#Change the values of the x columns according to the dataset
x = dataset.iloc[:,[3,4]].values



#Using the elbow method to find the number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
   kmeans = KMeans(n_clusters = i, init = "k-means++", random_state = 42)
   kmeans.fit(x)
   #Intertia gives the wcss value for the kmeans in this case for kmeans
   wcss.append(kmeans.inertia_)  
plt.plot(range(1,11), wcss)
plt.show()  

#Get the value for number of clusters (n_clusters from the elbow method)
#Using Kmeans from the elbow method
kmeans1 = KMeans(n_clusters = 5, init = "k-means++", random_state = 42)
#fit_predict will fit and predict
y_kmeans = kmeans1.fit_predict(x)

kmeans1.predict([[]])


