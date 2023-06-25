import numpy as np

def unit_step(y):
    return np.where(y > 0, 1, 0)