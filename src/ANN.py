#Implementation of the ANN
#NEEDS REVISION TO ACTUALLY WORK WITH A PROPER DATASET
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import general_tools as tools

dataset = pd.read_csv("../data/High_Entropy_Alloy_Parsed.csv")
fig_path = "../figures/ANN/"

X = dataset.iloc[:, 20:]
Y = dataset.iloc[:, 7]

X = tools.preprocess(X.to_numpy())
features = X.shape[1]
Y = tools.preprocess(Y.to_numpy())

X_train = X[:1000]
Y_train = Y[:1000]
X_val = X[1000:1400]
Y_val = Y[1000:1400]
X_test = X[1400:]
Y_test = Y[1400:]

model = tf.keras.Sequential([
    tf.keras.Input(shape = (features,),),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(units = 1, activation="linear")
])

model.compile(
    loss = tf.keras.losses.MeanSquaredError(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
)

model.fit(
    X,
    Y,
    epochs=50,
    batch_size = 100,
    validation_data=(X_val, Y_val)
)

predictions = model.predict(X_test)
plt.scatter(predictions, Y_test)
plt.show()
