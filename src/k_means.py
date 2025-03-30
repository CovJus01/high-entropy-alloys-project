#k_means implementation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import general_tools as tools


def k_means(X, k, max_iter=10):

        #Initialize Centroids
        new_centroids = init_centroids(X, k)

        #KMeans loop
        for i in range(max_iter):
            centroids = new_centroids
            #Assign Centroids
            centroid_assignments = assign_centroid(X, centroids)
            #Calculate new centroids
            new_centroids = update_centroids(X, centroids, centroid_assignments, k)

        centroid_assignments = assign_centroid(X, new_centroids)

        return new_centroids, centroid_assignments


def assign_centroid(X, centroids):

  #Setup assignment array
  centroid_assignments = [None] * X.shape[0]
  X_no_nan = np.nan_to_num(X, nan=0.0)

  for i in range(X.shape[0]):
      #Loop over the centroids
      for x in range(len(centroids)):

        #Calculate the distance
        distance = np.linalg.norm(X_no_nan[i] - centroids[x])


        #In the first pass populate the array
        if(x == 0):
          centroid_assignments[i] = (x, distance)

        #Compare old distance with current and update if better
        elif(distance < centroid_assignments[i][1]):
          centroid_assignments[i] = (x, distance)

  return centroid_assignments

def update_centroids(X, centroids, centroids_assignments, k):
  #Setup new array
  new_centroids = np.zeros((k,X.shape[1]))

  #Loop through centroids
  for k in range(k):
    count = 0

    #Loop through pixels
    for i in range(X.shape[0]):

        #If the pixel is assigned to this centroid add to sum
        if(centroids_assignments[i][0] == k):
          new_centroids[k] += X[i]
          count += 1

    if(count != 0):
      new_centroids[k] = new_centroids[k] / count
    else:
      new_centroids[k] = centroids[k]

  return new_centroids

def init_centroids(X,k):

    #Randomly initialize centroids
    centroids = np.random.rand(k, X.shape[1])

    return centroids


dataset = pd.read_csv("../data/High_Entropy_Alloy_Parsed.csv")
fig_path = "../figures/k_means/"

X = dataset.drop(columns=["IDENTIFIER: Reference ID", "FORMULA","PROPERTY: Microstructure", "PROPERTY: Processing method", "PROPERTY: BCC/FCC/other", "PROPERTY: Type of test"])
labels = list(X)
X = X.to_numpy()
X = tools.preprocess(X)

# Run K-means
centroids, assignments = k_means(X, 2)

