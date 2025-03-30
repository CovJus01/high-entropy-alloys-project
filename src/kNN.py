import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import general_tools as tools

#kNN implementation

def k_NN(X_train, t_train, k):

  print(X_train.shape)
  print(t_train.shape)
  N = t_train.shape[0]
  #Create an empty array to store the predictions
  predictions = np.empty(N)


  #Loop over X & t and perform k-NN
  for i in range(N):
    prediction =get_prediction(X_train, X_train[i],t_train, k)
    predictions[i] = prediction
  return predictions

def get_prediction(X_train, X, t_train, k):
    # Ignore training samples with NaN in their labels
    valid_indices = ~np.isnan(t_train)
    X_train_clean = X_train[valid_indices]
    t_train_clean = t_train[valid_indices]

    # Compute distances, ignoring NaN values in X
    diff = X_train_clean - X
    diff[np.isnan(diff)] = 0  # Treat NaN as 0 in difference calculation
    distances = np.linalg.norm(diff, axis=1)

    # Get indices of k nearest neighbors
    k_nearest_indices = np.argsort(distances)[:k]

    # Compute the mean of valid labels
    k_nearest_labels = t_train_clean[k_nearest_indices]
    valid_labels = k_nearest_labels[~np.isnan(k_nearest_labels)]  # Ignore NaNs in selected labels

    # Return mean prediction, or NaN if no valid neighbors
    return np.mean(valid_labels) if valid_labels.size > 0 else np.nan


dataset = pd.read_csv("../data/High_Entropy_Alloy_Parsed.csv")
fig_path = "../figures/kNN/"

# Setup X and t
X = dataset.drop(columns=["PROPERTY: Calculated Young modulus (GPa)","IDENTIFIER: Reference ID", "FORMULA","PROPERTY: Microstructure", "PROPERTY: Processing method", "PROPERTY: BCC/FCC/other", "PROPERTY: Type of test"])
t = dataset["PROPERTY: Calculated Young modulus (GPa)"].to_numpy()
labels = list(X)
X = X.to_numpy()
X = tools.preprocess(X)

#Attempt to approximate an unfilled dataentry
predictions = k_NN(X,t,15)

print(predictions)

