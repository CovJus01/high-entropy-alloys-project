import PCA_tools as pca
import general_tools as tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Script for PCA analysis on the data


#Intial research on the dataset
dataset = pd.read_csv("../data/High_Entropy_Alloy_Parsed.csv")
fig_path = "../figures/PCA/"

X = dataset.drop(columns=["IDENTIFIER: Reference ID",
                          "FORMULA",
                          "Microstructure",
                          "Processing method",
                          "Type of test"])
labels = list(X)

X = X.to_numpy()
X = tools.preprocess(X)

#Perform PCA on the data
T,P,R2_list = pca.nipalspca(X,10)

#Create some score plots
for i in range(9):
    tools.score_plot(T[i], T[i+1],f"t{i+1}",f"t{i+2}", f"t{i+2} vs t{i+1}", f"{fig_path}t{i+2}vst{i+1}.png")
    tools.loadings_plot(labels, P[i], f"P{i+1}", f"{fig_path}p{i+1}.png")

for i in range(len(R2_list)):
    print(f"{i}: \t{R2_list[i]}")

