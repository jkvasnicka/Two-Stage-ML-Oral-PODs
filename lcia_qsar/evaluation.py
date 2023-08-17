'''Provide support for evaluating machine-learning estimator predictions.
'''

#region: MetricWrapper
class MetricWrapper:
    '''
    A class for wrapping metrics.

    This class is designed to take a metric function as input and then 
    store any additional kwargs that this function may require. This is 
    particularly useful for metrics that require additional parameters 
    beyond the true and predicted values.

    Parameters
    ----------
    metric : callable
        The metric function to be wrapped. This should be a function that 
        takes as input two arrays: y_true and y_pred, which are the true 
        and predicted values respectively, and returns a float.
    **kwargs :
        Additional keyword arguments that will be passed to the metric 
        function when it is called.

    Attributes
    ----------
    metric : callable
        The metric function to be used.
    kwargs : dict
        The keyword arguments for the metric function.
    '''

    def __init__(self, metric, **kwargs):
        self.metric = metric
        self.kwargs = kwargs

    def __call__(self, y_true, y_pred):
        '''
        Call the wrapped metric function with the stored kwargs.

        This method makes an instance of the MetricWrapper callable, 
        so it can be used just like the original metric function, 
        but with the stored kwargs automatically included.

        Parameters
        ----------
        y_true : array-like
            The true values.
        y_pred : array-like
            The predicted values.

        Returns
        -------
        float
            The result of the metric function.
        '''
        return self.metric(y_true, y_pred, **self.kwargs)
#endregion

#region: score
def score(y_true, y_pred, function_for_metric):
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