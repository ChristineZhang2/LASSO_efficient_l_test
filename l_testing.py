from adelie import LassoPath  # Importing Adelie for efficient LASSO
from utilities import standardize, qhaar

def l_test(y, X, ind, lambda_val=-1, lambda_cv=-1, adjusted=False, lasso_model=None):
    n, p = X.shape
    y_std = standardize(y)
    
    if lasso_model is None:
        lasso_model = LassoPath().fit(X, y)
    beta_hat = lasso_model.coef_
    
    x_val = abs(beta_hat[ind])
    if x_val == 0:
        return 1
    
    pval_right = 1 - qhaar(x_val, n - p, lower_tail=True)
    pval_left = qhaar(-x_val, n - p, lower_tail=True)
    return pval_left + pval_right
