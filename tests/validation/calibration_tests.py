import numpy as np

def ece(bin_means_pred, bin_means_obs, bin_counts):
    diff = np.abs(bin_means_pred - bin_means_obs)
    w = bin_counts / np.maximum(1, np.sum(bin_counts))
    return float(np.sum(w * diff))
