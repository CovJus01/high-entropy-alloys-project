#Implementation of the ANN
#NEEDS REVISION TO ACTUALLY WORK WITH A PROPER DATASET
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import general_tools as tools

# A function for a masked MSE
def masked_mse_and_CE_loss(y_true, y_pred, num_reg):

    y_true_reg = y_true[:, :num_reg]
    y_pred_reg = y_pred[:, :num_reg]
    y_true_cat = y_true[:, num_reg:]
    y_pred_cat = y_pred[:, num_reg:]

    #Regression Error

    # Set value to true if the true value is valid
    mask = tf.math.logical_not(tf.math.is_nan(y_true_reg))
    mask = tf.cast(mask, tf.float32)

    # Setting true values to zero for the squared error calculation
    y_true_clean = tf.where(tf.math.is_nan(y_true_reg), tf.zeros_like(y_true_reg), y_true_reg)
    sq_error = tf.square(y_pred_reg - y_true_clean)

    #Ignoring invalid values
    masked_sq_error = sq_error * mask
    reg_error = tf.reduce_sum(masked_sq_error) / tf.reduce_sum(mask)

    #Classification error
    bce = tf.keras.losses.BinaryCrossentropy()
    cat_error = bce(y_true_cat, tf.keras.activations.sigmoid(y_pred_cat))


    return reg_error + cat_error




dataset = pd.read_csv("../data/High_Entropy_Alloy_Parsed.csv")
fig_path = "../figures/ANN/"


# Select the columns to be our inputs and our outputs
# X =  Different output properties
# Y =  Formula + Processing steps etc
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

X = dataset.drop(columns = Y_cols+Unused_cols)
Y = dataset[Y_cols]
y_labels = list(Y)
print(y_labels)
print(len(y_labels))
# Convert from a pandas dataframe into numpy values
X = tools.preprocess(X.to_numpy())
Y_reg = tools.preprocess(Y.iloc[:,:11].to_numpy())
Y_cat = Y.iloc[:,11:].to_numpy()
Y = np.hstack((Y_reg,Y_cat))

np.random.seed(1234)
indicies = np.random.permutation(len(X))
X = X[indicies]
Y = Y[indicies]
features = X.shape[1]
output_width = Y.shape[1]


# Split the dataset into the Training, Val and Test sets
X_train = X[:1000]
Y_train = Y[:1000]
X_val = X[1000:1400]
Y_val = Y[1000:1400]
X_test = X[1400:]
Y_test = Y[1400:]


# Define a basic model
model = tf.keras.Sequential([
    tf.keras.Input(shape = (features,),),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(units = output_width, activation="linear", kernel_initializer='he_normal')
])

# Compile using MSE
model.compile(
    loss = lambda y_true, y_pred: masked_mse_and_CE_loss(y_true, y_pred, num_reg = 11) ,
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
)

model.summary()


Epochs = 100
# Train the model using X_train and Y_train
history = model.fit(
    X_train,
    Y_train,
    batch_size=64,
    epochs=Epochs,
    validation_data=(X_val, Y_val)
)


# Test the model
predictions = model.predict(X_test)

# Plot the results
fig, axs = plt.subplots(4,5, figsize = (14,9))
fig.suptitle("True vs Predicted values")
fig.tight_layout()

#Print regressive
for i in range(11):
    x,y = int(i/5),int(i%5)
    pred = predictions[:, i]
    true = Y_test[:, i]
    axs[x,y].plot(pred[~np.isnan(true)], true[~np.isnan(true)], 'o')
    axs[x,y].set_title(y_labels[i])

#Print categorical
for i in range(11, 19):
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


plt.show()

# Plot the errors
plt.plot(range(Epochs),history.history["loss"], color = "b", label = "Training Loss")
plt.plot(range(Epochs),history.history["val_loss"], color = "g", label = "Validation Loss")
plt.show()
