import PCA_tools as pca
import general_tools as tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Script for PLS 


# Modifiy your PATHS since I messed up mine. 
dataset = pd.read_csv(r"C:\Users\KarlRizk\OneDrive - McMaster University\Desktop\4H03_ProjectRepo\Repo Folder\high-entropy-alloys-project\data\High_Entropy_Alloy_Parsed.csv")
fig_path = r"../high-entropy-alloys-project/figures/PCA/"

X = dataset.drop(columns=["IDENTIFIER: Reference ID", "FORMULA","PROPERTY: Microstructure", "PROPERTY: Processing method", "PROPERTY: BCC/FCC/other", "PROPERTY: Type of test"])
labels = list(X)

X = X.to_numpy()
X = tools.preprocess(X)

xvar = X[:,[]]
#Perform PCA on the data
T,P,R2_list = pca.nipalspca(X,10)

# #Create some score plots
tools.score_plot(T[0], T[1], 5, 5, "t2 vs t1", f"{fig_path}T1vsT2.png")
tools.score_plot(T[2], T[3], 5, 5, "t4 vs t3", f"{fig_path}T4vsT3.png")
tools.score_plot(T[4], T[5], 5, 5, "t6 vs t5", f"{fig_path}T6vsT5.png")



tools.loadings_plot(labels[:44], P[0][:44], "P1", f"{fig_path}p1.png")
tools.loadings_plot(labels[44:], P[0][44:], "P1", f"{fig_path}p1(2).png")
for i in range(len(R2_list)):
    print(f"{i}: \t{R2_list[i]}")
