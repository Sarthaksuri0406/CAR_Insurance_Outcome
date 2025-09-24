import numpy as np
import pandas as pd

def psi(expected, actual, bins=10):
    be = np.histogram(expected, bins=bins)[0] / len(expected)
    ba = np.histogram(actual, bins=bins)[0] / len(actual)
    be = np.where(be==0, 1e-6, be)
    ba = np.where(ba==0, 1e-6, ba)
    v = np.sum((be - ba) * np.log(be / ba))
    return v
