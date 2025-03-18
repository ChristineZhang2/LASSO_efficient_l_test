import numpy as np
from scipy.stats import t

def standardize(y):
    return (y - np.mean(y)) / np.std(y)

def normalize(v):
    return v / np.sqrt(np.sum(v**2))

def qhaar(q, n, lower_tail=True):
    if q >= 1:
        return 1 if lower_tail else 0
    if q <= -1:
        return 0 if lower_tail else 1
    if q == 0:
        return 0.5
    prob = 1 - t.cdf(np.sqrt((n - 1) / (1 / q**2 - 1)), df=n - 1)
    return prob if lower_tail else 1 - prob
