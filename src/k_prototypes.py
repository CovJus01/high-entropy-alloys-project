
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from kmodes.kprototypes import KPrototypes
import general_tools as tools
import PCA_tools as pca
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# Rewrite data paths, my own folder is acting weirdly.
# One to one recreation of the example file in the library github here: https://github.com/nicodv/kmodes/blob/master/examples/stocks.py
dataset = pd.read_csv("../data/High_Entropy_Alloy_Parsed.csv")
fig_path = "../figures/ANN/"

#Below is copied straight from ANN.py. Only difference is that unused_cols include the one-hot encoded categorical variables.
# X, the working array, should just be our desired categorical variables and numerical variables
# Unused_cols really just drops every column that isnt the categorical variables or y variables; processing method and microstructure.
# Select the columns to be our inputs and our outputs
# X =  Different output properties
# Y =  Formula + Processing steps etc
Unused_cols = ["IDENTIFIER: Reference ID",
               "FORMULA",
               "Processing method",
                "Microstructure",
               "Type of test",
               "Ag",
               "B2",
               "B2",
              "BCC",
             "FCC",
             "Sec.",
             "HCP",
             "L12",
             "Laves",
             "Other",
             "C_test",
             "T_test",
             "ANNEAL","CAST","OTHER","POWDER","WROUGHT"]

Y_cols = ["grain size",
             "Exp. Density",
             "Calculated Density",
             "HV",
             "Test temperature",
             "YS",
             "UTS",
             "Elongation",
             "Elongation plastic",
             "Exp. Young modulus",
             "Calculated Young modulus",
             "B2",
             "BCC",
             "FCC",
             "Sec.",
             "HCP",
             "L12",
             "Laves",
             "Other"]

X_pca = dataset.drop(columns=["IDENTIFIER: Reference ID",
                          "FORMULA",
                          "Microstructure",
                          "Processing method",
                          "Type of test"])
# This is an identifier for the datapoints. Used later on.
syms = np.genfromtxt("../data/High_Entropy_Alloy_Parsed.csv", dtype=str, delimiter=',')[:, 0]
#Drops unused columns, and restricts data array to only columns with numerical values.
X = dataset.drop(columns = Unused_cols)
intermediate = dataset.to_numpy()
X = tools.preprocess(X.to_numpy()) # Pre-process data.

X_pca = X_pca.to_numpy()
X_pca = tools.preprocess(X_pca)

# Replacing NaN with mean. Copied from https://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns
#Obtain mean of columns as you need, nanmean is convenient. Should be restricted to only numerical columns.
col_mean = np.nanmean(X, axis=0)

#Find indices that you need to replace
inds = np.where(np.isnan(X))

#Place column means in the indices. Align the arrays using take
X[inds] = np.take(col_mean, inds[1])


intermediate = intermediate.astype(str) # Turning intermediate to string


X = np.concatenate((intermediate[:, 2:3],X), axis=1) # Concantenate columns with strings with pre-processed data.
                                                    # Columns 2 and 3 are the columns in the full csv with the desired categorical variable.
                                                    # Concantedned to the left.

T,P,R2 = pca.nipalspca(X_pca,10)
elbow_scores = {}

for N in range(2,25):
    kproto = KPrototypes(n_clusters=N, init='Cao', verbose=2)
    clusters = kproto.fit_predict(X, categorical=[0, 1]) # categorical =[0,1] describe which columns contain categorical variables.
    elbow_scores[N] = kproto.cost_
# Print cluster centroids of the trained model.
    print(kproto.cluster_centroids_)
# Print training statistics
    print(kproto.cost_)
    print(kproto.n_iter_)


# Uses ID values to show which samples correspond to which cluster. Some ID values belong to different clusters
# i.e, their properties were tested to be  different enough that they appear in different clusters.
# Exposes a huge vulnerability in the dataset, implies material scientists either made errors or made
# heterogenous samples.
# for s, c in zip(syms, clusters):
#     print(f"Symbol: {s}, cluster:{c}")

# Visualization Section
    if(N ==2):
        dataset = dataset.drop(columns=["Type of test", "Ag"])

# Extract features (everything from column 15 onwards)
    X_tsne = dataset.iloc[:, 4:].to_numpy()

# Replace NaNs with column means (or choose a different strategy)
    col_means = np.nanmean(X_tsne, axis=0)
    inds = np.where(np.isnan(X_tsne))
    X_tsne[inds] = np.take(col_means, inds[1])

# Standardize features
    X_scaled = StandardScaler().fit_transform(X_tsne)

# Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X_scaled)

# Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters, cmap="viridis", s=10, alpha=0.7) # Note that "c=" is a sequence of n numbers
    plt.title("t-SNE Embedding of HEA Dataset")                                                     # to be mapped using cmap.
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.show()

    #Plot PCA
    plt.scatter(T[0], T[1], c = clusters, cmap= "viridis", s = 10)
    plt.title("t2 vs t1")
    plt.show()
    plt.scatter(T[2], T[4], c = clusters, cmap= "viridis", s = 10)
    plt.title("t4 vs t3")
    plt.show()
#link for how I did the mapping:
# https://codesignal.com/learn/courses/k-means-clustering-decoded/lessons/visualizing-k-means-clustering-on-an-iris-dataset-with-matplotlib


#Plot Elbow plot
plt.plot(elbow_scores.keys(), elbow_scores.values())
plt.show()
