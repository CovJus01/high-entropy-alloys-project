# Implementation of tools for PCA analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def nipalspca(x, A):
    x = x.astype(float)  # Ensure input is float for numerical stability

    # Store deflated matrices
    components = [x.copy() for _ in range(A + 1)]

    T = [None] * A  # Score matrix
    P = [None] * A  # Loading matrix
    R2_list = [None] * A  # Variance explained

    for i in range(A):
        step = 0

        # Step 1: Initialize t as the column with max variance (ignoring NaNs)
        max_var_col = np.nanargmax(np.nanvar(components[i], axis=0))
        t = components[i][:, max_var_col].copy()

        while True:
            t_last = t.copy()

            # Step 2.1: Compute loadings (p), ignoring NaNs

            numerator = np.nansum(components[i] * t[:, np.newaxis], axis=0)
            denominator = np.nansum(t ** 2)
            p = numerator / denominator

            # Step 2.2: Normalize p (ignoring NaNs)
            p /= np.sqrt(np.nansum(p ** 2))

            # Step 2.3: Compute new scores (t), ignoring NaNs
            numerator = np.nansum(components[i] * p, axis=1)
            denominator = np.nansum(p ** 2)
            t = numerator / denominator

            step += 1
            T[i] = t
            P[i] = p

            # Convergence check
            if np.linalg.norm(np.nan_to_num(t - t_last)) < 1e-8 or step > 500:
                break

        # Step 3: Deflate X while handling NaNs
        t = t.reshape((-1, 1))
        p = p.reshape((-1, 1))
        components[i + 1] = components[i] - np.dot(np.nan_to_num(t), np.nan_to_num(p.T))

        # Compute R2 for this component (ignoring NaNs)
        R2_list[i] = 1 - (np.nanvar(components[i+1]) / np.nanvar(components[0]))

    return T, P, R2_list
