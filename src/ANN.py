#Implementation of the ANN
#NEEDS REVISION TO ACTUALLY WORK WITH A PROPER DATASET
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import general_tools as tools
from matplotlib.lines import Line2D

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


# ********************** The Script begins Here **********************

# PREPROCESSING AND DATA HANDLING

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

# Center and scale X, then apply a mask for NaNs and append to the feature list
X = tools.preprocess(X.to_numpy())
X_mask = np.ma.masked_invalid(X[:,0])
X_mask = np.reshape(X_mask, (-1,1))
X = np.hstack((X, X_mask.mask))
X[np.isnan(X)] = 0

# Center and scale continuous values, stack them beside the categorical ones
Y_reg = tools.preprocess(Y.iloc[:,:10].to_numpy())
Y_cat = Y.iloc[:,10:].to_numpy()
Y = np.hstack((Y_reg,Y_cat))

#Shuffle our data to get different representation in the sets
np.random.seed(1234)
indicies = np.random.permutation(len(X))
X = X[indicies]
Y = Y[indicies]

#Get input and output widths
features = X.shape[1]
output_width = Y.shape[1]


# Split the dataset into the Training, Val and Test sets
X_train = X[:1000]
Y_train = Y[:1000]
X_val = X[1000:1400]
Y_val = Y[1000:1400]
X_test = X[1400:]
Y_test = Y[1400:]



# ANN SETUP AND TRAINING

models = {}
for nodes in [2,4,8,16,32,64,512]:
    for layers in [1,2,4,8]:

# Define a basic model
        model = tf.keras.Sequential([
            tf.keras.Input(shape = (features,),),
        ])

            #ADD Hidden Layers
        for layer in range(layers):
            model.add(tf.keras.layers.Dense(nodes, activation="relu"))

        #ADD Output Layer
        model.add(tf.keras.layers.Dense(units = output_width, activation="linear", kernel_initializer='he_normal'))


# Compile using custom error
        model.compile(
            loss = lambda y_true, y_pred: masked_mse_and_CE_loss(y_true, y_pred, num_reg = 10),
            optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
        )

        model.summary()

        Epochs = 100
        if(layers == 16):
            Epochs =500
# Train the model using X_train and Y_train
        history = model.fit(
            X_train,
            Y_train,
            batch_size=64,
            epochs=Epochs,
            validation_data=(X_val, Y_val)
        )


# MODEL ANALYSIS AND TESTING

# Test the model
        predictions = model.predict(X_test)

# Plot the results
        fig, axs = plt.subplots(4,5, figsize = (14,9))
        fig.suptitle("True vs Predicted values")
        fig.tight_layout()

#Plot regressive
        for i in range(10):
            x,y = int(i/5),int(i%5)
            pred = predictions[:, i]
            true = Y_test[:, i]
            axs[x,y].plot(pred[~np.isnan(true)], true[~np.isnan(true)], 'o')
            axs[x,y].set_title(y_labels[i])

#Plot categorical
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

        fig.delaxes(axs[3][4])
        fig.savefig(f"../figures/ANN/{nodes}n{layers}l_results.png")
        plt.close(fig)
# Plot the errors
        plt.plot(range(Epochs),history.history["loss"], color = "b", label = "Training Loss")
        plt.plot(range(Epochs),history.history["val_loss"], color = "g", label = "Validation Loss")

        #Uncomment for Error charts
        plt.close()
        #plt.show()
        models[f"{nodes}n{layers}l"] = {"num_nodes": nodes,
                                        "num_layers": layers,
                                        "train_error": history.history["loss"][-1],
                                        "val_error": history.history["val_loss"][-1]}


for model in models:
    curr_mod = models[model]
    plt.bar(f"{curr_mod['num_nodes']}n{curr_mod['num_layers']}l_t", curr_mod["train_error"], color= "blue")
    plt.bar(f"{curr_mod['num_nodes']}n{curr_mod['num_layers']}l_v", curr_mod["val_error"], color= "red")

plt.title("Model progression by Nodes and Depth")
plt.legend([Line2D([0], [0], color="blue", lw=4),
            Line2D([0], [0], color="red", lw=4)],["Training error", "Validation error"])
plt.xticks(rotation = 45, ha ='right')
plt.show()

