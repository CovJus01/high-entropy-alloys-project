
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from kmodes.kprototypes import KPrototypes
import general_tools as tools


# Rewrite data paths, my own folder is acting weirdly. 
# One to one recreation of the example file in the library github here: https://github.com/nicodv/kmodes/blob/master/examples/stocks.py
dataset = pd.read_csv("../high-entropy-alloys-project/data/High_Entropy_Alloy_Parsed.csv")
fig_path = "../figures/ANN/"

#Below is copied straight from ANN.py. Only difference is that unused_cols include the one-hot encoded categorical variables. 
# X, the working array, should just be our desired categorical variables and numerical variables
# Unused_cols really just drops every column that isnt the categorical variables or y variables; processing method and microstructure. 
# Select the columns to be our inputs and our outputs
# X =  Different output properties
# Y =  Formula + Processing steps etc
Unused_cols = ["IDENTIFIER: Reference ID",
               "FORMULA",
               #"Processing method",
               # "Microstructure",
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
# This is an identifier for the datapoints. Used later on.
syms = np.genfromtxt("../high-entropy-alloys-project/data/High_Entropy_Alloy_Parsed.csv", dtype=str, delimiter=',')[:, 0]

intermediate = dataset.drop(columns = Unused_cols)
X = tools.preprocess(intermediate.to_numpy()[:, 2:]) # Pre-process data.
"""
Line gives the below error. 
AttributeError: 'float' object has no attribute 'sqrt'

The above exception was the direct cause of the following exception:
"""
X = np.concatenate((intermediate.to_numpy()[:, :1],X), axis=1) # Concantenate strings with pre-processed data. 
# Replacing NaN with mean. Copied from https://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns
#Obtain mean of columns as you need, nanmean is convenient. Should be restricted to only numerical columns.
col_mean = np.nanmean(X[:,2:], axis=0)

#Find indices that you need to replace
inds = np.where(np.isnan(X[:,2:]))

#Place column means in the indices. Align the arrays using take
X[:,2:][inds] = np.take(col_mean, inds[1])


# Drops unused columns, and restricts data array to only columns after ID.

kproto = KPrototypes(n_clusters=3, init='Cao', verbose=2)
clusters = kproto.fit_predict(X, categorical=[0, 1]) # categorical =[0,1] describe which columns contain categorical variables. 

# Print cluster centroids of the trained model.
print(kproto.cluster_centroids_)
# Print training statistics
print(kproto.cost_)
print(kproto.n_iter_)

for s, c in zip(syms, clusters):
    print(f"Symbol: {s}, cluster:{c}")