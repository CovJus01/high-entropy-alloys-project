# Implementation of tools for PCA analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Nipals PCA algorithm
def nipalspca(x, A):

    #This will store the components in a list to be accessed through loop
    components = [x] * (A +1)
    for i in range(A):

        step = 0
        #Step 1 - Select arbitrary t column
        t = components[i][:, 0]
        while(1):
            t_last = t
            #Step 2.1 - Regress the columns onto t to get p
            p = np.dot(1/np.dot(t.T, t),
                       np.dot(t.T, components[i]))

            #Step 2.2 - Normalize p to unit length
            p = p/np.linalg.norm(p)

            #Step 2.3 - Regress each row onto p.T  to get new t
            t = np.dot(1/np.dot(p.T, p),
                       np.dot(components[i], p))

            step+= 1
            delta_t = np.linalg.norm(t - t_last)
            #Check for convergence
            if(delta_t < 1e-8 or step > 500):
                break

        p = p.reshape((p.shape[0],1))
        t = t.reshape((t.shape[0],1))

        #Step 3 - Deflate X(a-1) to get X(a)
        components[i+1] = components[i] - np.matmul(t, p.T)


    R_sqr = 1 - (np.var(components[A])/np.var(components[0]))
    return t,p,R_sqr
