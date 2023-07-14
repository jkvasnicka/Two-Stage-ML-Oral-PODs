'''Provide custom scoring functions to supplement sklearn.metrics.
'''

from sklearn import metrics
import numpy as np

#region: root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    '''Return the root-mean-squared error (RMSE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
        
    Returns
    -------
    float
    '''
    mse = metrics.mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)
#endregion