import PCA_tools as pca
import general_tools as tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Script for PCA analysis on the data


#Intial research on the dataset
dataset = pd.read_csv("../data/High_Entropy_Alloy_Parsed.csv")

print("\nData columns in our dataset\n")
for label in list(dataset):
    print(label)

X = dataset.drop(columns=["IDENTIFIER: Reference ID", "FORMULA","PROPERTY: Microstructure", "PROPERTY: Processing method", "PROPERTY: BCC/FCC/other", "PROPERTY: Type of test"]).to_numpy()

print(X)
t,p,R2 = pca.nipalspca(X,5)




