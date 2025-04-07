#General Purpose tools for data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Centering and Scaling for a given dataset X
def preprocess(X):
    #Normalize and Center the data
    X_c  = X - np.nanmean(X, axis=0)
    X_cs = X_c / np.nanstd(X_c, axis = 0)

    return X_cs

# Plotting a loadings and score plot for 2 components
def loadings_score_plot(t1, t2, p1, p2,  labels, x_lim, y_lim, title, path):

    #Scatter scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap('magma', len(labels)+2)
    ax.scatter(t1,t2)

    #Plot loadings
    for i in range(len(labels)):
      r = np.power(np.add(np.power(p1[i],2),np.power(p2[i],2)),0.5)
      ax.quiver(0,0,p1[i]/r,p2[i]/r, scale=3, width = 0.007, label = labels[i], color = cmap(i))

    #Adjust figure
    ax.set_xlim(-1*x_lim,x_lim)
    ax.set_xlabel("t1")
    ax.set_ylabel("t2")
    ax.set_ylim(-1*y_lim,y_lim)
    ax.legend(bbox_to_anchor=(1.15, 1))
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)

def score_plot(t1, t2, x_lim, y_lim, title, path):

    #Scatter scores
    fig,ax = plt.subplots(nrows =1, ncols=1)
    ax.set_xlim(-1*x_lim,x_lim)
    ax.set_xlabel("t1")
    ax.set_ylabel("t2")
    ax.set_ylim(-1*y_lim,y_lim)
    ax.set_title(title)
    ax.scatter(t1,t2)

    fig.savefig(path)
    #Adjust figure
    plt.close(fig)

def loadings_plot(labels, p, title, path):

    #Scatter scores
    fig,ax = plt.subplots(nrows =1, ncols=1, figsize=(30,15))
    cmap = plt.get_cmap('magma', len(labels))
    colors = cmap(np.linspace(0, 1, len(labels)))
    box = ax.get_position()
    ax.set_position([box.x0-0.05, box.y0, box.width, box.height])
    ax.set_title(title)
    ax.bar(range(len(labels)),p, label = labels, color = colors, edgecolor = "k")
    ax.legend(bbox_to_anchor=(1, 1))
    fig.savefig(path)
    #Adjust figure
    plt.close(fig)
