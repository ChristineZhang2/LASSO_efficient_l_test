import numpy as np
import cvxpy as cp
from adelie import LassoPath  # Importing Adelie for efficient LASSO

def beta_x(x, y, X, ind, lamb):
    n, p = X.shape
    Z = np.hstack((np.ones((n, 1)), np.delete(X, ind, axis=1)))
    beta = cp.Variable(p)
    obj = cp.sum_squares(y - x * X[:, ind] - Z @ beta) / (2 * n) + lamb * cp.norm(beta[1:], 1)
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()
    return np.round(beta.value, 9)

def beta_full(y, X, ind, lamb):
    n, p = X.shape
    Z = np.hstack((np.ones((n, 1)), X))
    beta = cp.Variable(p + 1)
    obj = cp.sum_squares(y - Z @ beta) / (2 * n) + lamb * cp.norm(beta[1:], 1)
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()
    return np.round(beta.value, 9)
