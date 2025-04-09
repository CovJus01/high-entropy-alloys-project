# Implementation of the ANN with sub-models
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import general_tools as tools

# Function for masked MSE
def masked_mse(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    mask = tf.cast(mask, tf.float32)
    y_true_clean = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    sq_error = tf.square(y_pred - y_true_clean)
    masked_sq_error = sq_error * mask
    return tf.reduce_sum(masked_sq_error) / tf.reduce_sum(mask)

# A function for a masked MSE
def masked_mse_and_CE_loss(y_true, y_pred, num_reg):
    y_true_reg = y_true[:, :num_reg]
    y_pred_reg = y_pred[:, :num_reg]
    y_true_cat = y_true[:, num_reg:]
    y_pred_cat = y_pred[:, num_reg:]

    # Regression Error
    mask = tf.math.logical_not(tf.math.is_nan(y_true_reg))
    mask = tf.cast(mask, tf.float32)
    y_true_clean = tf.where(tf.math.is_nan(y_true_reg), tf.zeros_like(y_true_reg), y_true_reg)
    sq_error = tf.square(y_pred_reg - y_true_clean)
    masked_sq_error = sq_error * mask
    reg_error = tf.reduce_sum(masked_sq_error) / tf.reduce_sum(mask)

    # Classification error
    bce = tf.keras.losses.BinaryCrossentropy()
    cat_error = bce(y_true_cat, tf.keras.activations.sigmoid(y_pred_cat))

    return reg_error + cat_error

# Load and prepare data
dataset = pd.read_csv("../data/High_Entropy_Alloy_Parsed.csv")

Unused_cols = ["IDENTIFIER: Reference ID",
               "FORMULA",
               "Microstructure",
               "Processing method",
               "Type of test",
               "Ag"]

Y_cols = ["grain size",
          "Exp. Density",
          "Calculated Density",
          "HV",
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

X = dataset.drop(columns=Y_cols+Unused_cols)
Y = dataset[Y_cols]
y_labels = list(Y)

# Preprocess X
X = tools.preprocess(X.to_numpy())
X_mask = np.ma.masked_invalid(X[:,0])
X_mask = np.reshape(X_mask, (-1,1))
X = np.hstack((X, X_mask.mask))
X[np.isnan(X)] = 0

# Preprocess Y
Y_reg = tools.preprocess(Y.iloc[:,:10].to_numpy())
Y_cat = Y.iloc[:,10:].to_numpy()
Y = np.hstack((Y_reg,Y_cat))

# Shuffle data
np.random.seed(1234)
indices = np.random.permutation(len(X))
X = X[indices]
Y = Y[indices]

# Split data
X_train = X[:1000]
Y_train = Y[:1000]
X_val = X[1000:1400]
Y_val = Y[1000:1400]
X_test = X[1400:]
Y_test = Y[1400:]

# Get input and output widths
features = X.shape[1]
output_width = Y.shape[1]

# Define sub-models for less abundant properties
# We'll create specialized models for properties with more sparse data
# These are typically the later properties in the list

# Sub-model 1: Focus on mechanical properties (HV, YS, UTS, Elongation)
mech_props = [3, 4, 5, 6]  # indices of mechanical properties
submodel1 = tf.keras.Sequential([
    tf.keras.Input(shape=(features,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(mech_props), activation="linear")
])

# Sub-model 2: Focus on phase identification (B2, BCC, FCC, etc.)
phase_props = list(range(10, 18))  # indices of phase properties
submodel2 = tf.keras.Sequential([
    tf.keras.Input(shape=(features,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(phase_props), activation="linear")
])

# Sub-model 3: Focus on density and modulus properties
density_mod_props = [1, 2, 8, 9]  # indices of density/modulus properties
submodel3 = tf.keras.Sequential([
    tf.keras.Input(shape=(features,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(density_mod_props), activation="linear")
])

# Compile sub-models
submodel1.compile(optimizer='adam', loss=masked_mse)
submodel2.compile(optimizer='adam', loss='binary_crossentropy')
submodel3.compile(optimizer='adam', loss=masked_mse)

submodel_batchsize = 32
submodel_epochs = 250

# Train sub-models
print("Training sub-model 1 (mechanical properties)...")
submodel1.fit(
    X_train,
    Y_train[:, mech_props],
    batch_size=submodel_batchsize,
    epochs=submodel_epochs,
    validation_data=(X_val, Y_val[:, mech_props]),
    verbose=1
)

print("\nTraining sub-model 2 (phase identification)...")
submodel2.fit(
    X_train,
    Y_train[:, phase_props],
    batch_size=submodel_batchsize,
    epochs=submodel_epochs,
    validation_data=(X_val, Y_val[:, phase_props]),
    verbose=1
)

print("\nTraining sub-model 3 (density/modulus properties)...")
submodel3.fit(
    X_train,
    Y_train[:, density_mod_props],
    batch_size=submodel_batchsize,
    epochs=submodel_epochs,
    validation_data=(X_val, Y_val[:, density_mod_props]),
    verbose=1
)

# Create predictions from sub-models to use as features for final model
train_sub_preds = [
    submodel1.predict(X_train),
    submodel2.predict(X_train),
    submodel3.predict(X_train)
]
val_sub_preds = [
    submodel1.predict(X_val),
    submodel2.predict(X_val),
    submodel3.predict(X_val)
]
test_sub_preds = [
    submodel1.predict(X_test),
    submodel2.predict(X_test),
    submodel3.predict(X_test)
]

# Combine original features with sub-model predictions
X_train_combined = np.hstack([X_train] + train_sub_preds)
X_val_combined = np.hstack([X_val] + val_sub_preds)
X_test_combined = np.hstack([X_test] + test_sub_preds)

# Final model that combines all predictions
final_model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train_combined.shape[1],)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(output_width, activation="linear")
])

# Compile final model
final_model.compile(
    loss=lambda y_true, y_pred: masked_mse_and_CE_loss(y_true, y_pred, num_reg=10),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)

final_model.summary()

# Train final model
Epochs = 500
history = final_model.fit(
    X_train_combined,
    Y_train,
    batch_size=64,
    epochs=Epochs,
    validation_data=(X_val_combined, Y_val)
)

# Test the final model
predictions = final_model.predict(X_test_combined)

# Plot the results
fig, axs = plt.subplots(4,5, figsize=(15,10))
fig.suptitle(f"True vs Predicted values for 3 Primary Networks (Batch size = {submodel_batchsize}, Submodel Epochs = {submodel_epochs}) with {Epochs} Epochs for Final Network")
fig.tight_layout()

# Print regressive
for i in range(10):
    x,y = int(i/5),int(i%5)
    pred = predictions[:, i]
    true = Y_test[:, i]
    axs[x,y].plot(pred[~np.isnan(true)], true[~np.isnan(true)], 'o')
    axs[x,y].set_title(y_labels[i])

# Print categorical
for i in range(10, 18):
    x,y = int(i/5),int(i%5)
    pred = predictions[:, i]
    pred = tf.keras.activations.sigmoid(pred) > 0.5
    true = Y_test[:, i]
    pred = pred.numpy()

    # Calculate confusion matrix components
    TP = ((pred == 1) & (true == 1)).sum()
    TN = ((pred == 0) & (true == 0)).sum()
    FP = ((pred == 1) & (true == 0)).sum()
    FN = ((pred == 0) & (true == 1)).sum()
    confusion_list = [TP, TN, FP, FN]
    confusion_labels = ["TP", "TN", "FP", "FN"]
    axs[x,y].bar(confusion_labels, confusion_list)
    axs[x,y].set_title(y_labels[i])

# Clean up a bit
fig.delaxes(axs[3][3])
fig.delaxes(axs[3][4])
plt.show()
fig.savefig(f"../figures/ANNs/{submodel_batchsize}BS-{submodel_epochs}SubEps-{Epochs}Eps_results.png")
plt.close(fig)

# Plot the errors
fig = plt.figure()
plt.plot(range(Epochs), history.history["loss"], color="b", label="Training Loss")
plt.plot(range(Epochs), history.history["val_loss"], color="g", label="Validation Loss")
plt.title("Training Loss and Validation Loss")
fig.legend()
fig.show()



fig.savefig(f"../figures/ANNs/{submodel_batchsize}BS-{submodel_epochs}SubEps-{Epochs}Eps_loss.png")
plt.close(fig)

