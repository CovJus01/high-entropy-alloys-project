#Implementation of the ANN
#NEEDS REVISION TO ACTUALLY WORK WITH A PROPER DATASET
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import general_tools as tools

dataset = pd.read_csv("../data/High_Entropy_Alloy_Parsed.csv")
fig_path = "../figures/ANN/"

X = dataset.drop(columns=["PROPERTY: Calculated Young modulus (GPa)","IDENTIFIER: Reference ID", "FORMULA","PROPERTY: Microstructure", "PROPERTY: Processing method", "PROPERTY: BCC/FCC/other", "PROPERTY: Type of test"])
Y = dataset["PROPERTY: Calculated Young modulus (GPa)"]

X = tools.preprocess(X.to_numpy())
features = X.shape[1]
Y = tools.preprocess(Y.to_numpy())

model = tf.keras.Sequential([
    tf.keras.Input(shape = (features,),),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(units = 1, activation="linear")
])

model.compile(
    loss = tf.keras.losses.MeanSquaredError(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
    metrics = [tf.keras.metrics.Accuracy()],
)

model.fit(
    X,
    Y,
    epochs=50,
    batch_size = 100
)

predictions = model.predict(X)
plt.scatter(predictions, Y)
plt.show()
