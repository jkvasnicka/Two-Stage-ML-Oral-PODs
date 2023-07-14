'''Provide support for evaluating machine-learning estimator predictions.
'''

from sklearn import metrics
import numpy as np 

from common import custom_metrics

# Define names of all regression metric functions in sklearn.metrics.
regression_metrics = [ 
    'explained_variance_score', 
    'max_error', 
    'mean_absolute_error', 
    'mean_squared_error', 
    'mean_squared_log_error', 
    'median_absolute_error', 
    'mean_absolute_percentage_error', 
    'mean_pinball_loss', 
    'r2_score', 
    'mean_tweedie_deviance', 
    'mean_poisson_deviance', 
    'mean_gamma_deviance', 
    'd2_tweedie_score'
]

#region: scores
def scores(y_true, y_pred, function_for_metric):
    '''Score the predicted values for each metric.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    function_for_metric : dict[k] --> function
       Scoring functions for model evaluation. These functions could be from 
       sklearn.metrics or custom.

    Returns
    ----------
    dict[k] --> float
        Mapping names of metrics to scores.
    '''
    return {metric : score(y_true, y_pred) 
            for metric, score in function_for_metric.items()}
#endregion

#region: try_scores
def try_scores(y_true, y_pred, function_for_metric):
    '''Score the predicted values for each metric.

    This function is like scores() but returns NaN where an error would 
    otherwise be raised. Can be used to determine which metrics to keep.
    '''
    # Initialize the container.
    score_for_metric = {}
    for metric, score in function_for_metric.items():
        try:
            score_for_metric[metric] = score(y_true, y_pred)
        except:
            # Score cannot be computed on these data.
            score_for_metric[metric] = np.nan
    return score_for_metric
#endregion

#region: get_scoring_functions
def get_scoring_functions(metrics_keys):
    '''Return a dict containing functions for model evaluation/scoring.

    Parameters
    ----------
    metrics_keys : list of str
        Names of metric functions. Must correspond to functions in 
        sklearn.metrics or custom.

    See Also
    --------
    '''
    # Initialize the container.
    function_for_metric = {}
    for metric in metrics_keys:
        try:
            function_for_metric[metric] = sklearn_metrics_get(metric)
        except AttributeError:
            function_for_metric[metric] = getattr(custom_metrics, metric)
    return function_for_metric
#endregion

# FIXME: This function may not be necessary.
#region: sklearn_metrics_get
def sklearn_metrics_get(attr_name):
    '''Get the specified attribute from sklearn.metrics.

    The attribute name could correspond a scoring function.
    '''
    return getattr(metrics, attr_name)
#endregion