#Implementation of the ANN
#NEEDS REVISION TO ACTUALLY WORK WITH A PROPER DATASET
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import general_tools as tools

# A function for a masked MSE
def masked_mse(y_true, y_pred):

    # Set value to true if the true value is valid
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    mask = tf.cast(mask, tf.float32)

    # Setting true values to zero for the squared error calculation
    y_true_clean = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    sq_error = tf.square(y_pred - y_true_clean)

    #Ignoring invalid values
    masked_sq_error = sq_error * mask

    # Return the Mean
    return tf.reduce_sum(masked_sq_error) / tf.reduce_sum(mask)



dataset = pd.read_csv("../data/High_Entropy_Alloy_Parsed.csv")
fig_path = "../figures/ANN/"
dataset = dataset.drop(columns = ["Type of test", "Ag"])

# Select the columns to be our inputs and our outputs
# X =  Different output properties
# Y =  Formula + Processing steps etc
X = dataset.iloc[:, 15:]
Y = dataset.iloc[:, 4:15]
y_labels = list(Y)

# Convert from a pandas dataframe into numpy values
X = tools.preprocess(X.to_numpy())
Y = tools.preprocess(Y.to_numpy())
np.random.seed(1234)
indicies = np.random.permutation(len(X))
X = X[indicies]
Y = Y[indicies]
features = X.shape[1]

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
    tf.keras.layers.Dense(units = 11, activation="linear", kernel_initializer='he_normal')
])

# Compile using MSE
model.compile(
    loss = masked_mse,
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
fig, axs = plt.subplots(3,4, figsize = (14,9))
fig.suptitle("True vs Predicted values")

for i in range(11):
    x,y = int(i/4),int(i%4)
    pred = predictions[:, i]
    true = Y_test[:, i]
    axs[x,y].plot(pred[~np.isnan(true)], true[~np.isnan(true)], 'o')
    axs[x,y].set_title(y_labels[i])


plt.show()

# Plot the errors
plt.plot(range(Epochs),history.history["loss"], color = "b", label = "Training Loss")
plt.plot(range(Epochs),history.history["val_loss"], color = "g", label = "Validation Loss")
plt.show()
