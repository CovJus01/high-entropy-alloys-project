#Implementation of the ANN
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

X_data = pd.read_csv("pie_production.csv").to_numpy()
Y_data = pd.read_csv("quality.csv").to_numpy()
print(X_data.shape)
print(Y_data.shape)
features = X_data.shape[1]


model = tf.keras.Sequential([
    tf.keras.Input(shape = (features,),),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(units = 1, activation="linear")
])

model.compile(
    loss = tf.keras.losses.MeanSquaredError(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
    metrics = [tf.keras.metrics.BinaryAccuracy()],
)

model.fit(
    X_data,
    Y_data,
    epochs=50,
    batch_size = 100
)

predictions = model.predict(X_data)
plt.scatter(predictions, Y_data)
plt.show()
