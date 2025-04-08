import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Load the dataset
data_path = "../high-entropy-alloys-project/data/High_Entropy_Alloy_Parsed.csv"
dataset = pd.read_csv(data_path)

# Drop non-feature columns
Unused_cols = [#"IDENTIFIER: Reference ID",
               "FORMULA",
               #"Processing method",
                #"Microstructure",
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

dataset = dataset.drop(columns=["Type of test", "Ag"])

# Extract features (everything from column 15 onwards)
X = dataset.iloc[:, 4:].to_numpy()

# Replace NaNs with column means (or choose a different strategy)
col_means = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_means, inds[1])

# Standardize features
X_scaled = StandardScaler().fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X_scaled)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=10, alpha=0.7)
plt.title("t-SNE Embedding of HEA Dataset")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)
plt.show()