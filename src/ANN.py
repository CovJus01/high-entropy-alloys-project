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

# Convert from a pandas dataframe into numpy values
X = tools.preprocess(X.to_numpy())
Y = tools.preprocess(Y.to_numpy())
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
    tf.keras.layers.Dense(50, activation="relu", kernel_initializer='he_normal'),
    tf.keras.layers.Dense(units = 11, activation="linear", kernel_initializer='he_normal')
])

# Compile using MSE
model.compile(
    loss = masked_mse,
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
)

model.summary()


Epochs = 50
# Train the model using X_train and Y_train
history = model.fit(
    X_train,
    Y_train,
    batch_size=64,
    epochs=Epochs,
    validation_data=(X_val, Y_val)
)

Y_test[np.isnan(Y_test)] = 0.0
predictions = model.predict(X_test)
for i in range(11):
    plt.scatter(predictions[:, i], Y_test[:,i])
    plt.show()

plt.plot(range(Epochs),history.history["loss"], color = "b")
plt.plot(range(Epochs),history.history["val_loss"], color = "g")
plt.show()
