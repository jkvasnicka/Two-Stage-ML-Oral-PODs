'''
This module provides a `MetricsManager` class to manage and evaluate various 
metrics according to the configuration settings. It also includes a 
`MetricWrapper` class to handle metrics that require additional parameters 
beyond the true and predicted values.

Classes
-------
- MetricsManager : A class to manage and evaluate metrics.
- MetricWrapper : A class for wrapping metrics with additional parameters.
'''

import importlib

class MetricsManager:
    '''
    A class to manage and evaluate metrics.

    This class takes a dictionary of metric settings, loads the corresponding
    metric functions, and provides a method to score predictions according 
    to these metrics.

    Example
    -------
    metrics_settings = {
        "mean_squared_error": {
            "module": "sklearn.metrics"
        }
    }
    metrics_manager = MetricsManager(metrics_settings)
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    scores = metrics_manager.score(y_true, y_pred)
    '''
    def __init__(self, metrics_settings):
        '''
        Initialize the MetricsManager with given settings.

        Parameters
        ----------
        metrics_settings : dict
            Dictionary mapping metric names to their corresponding settings.
            The settings include module, class (optional), and kwargs (optional).
        '''
        self.function_for_metric = {}  # initialize

        for name, config in metrics_settings.items():
            module = importlib.import_module(config['module'])
            class_name = config.get('class', name)
            kwargs = config.get('kwargs', {})
            
            metric = getattr(module, class_name)
            metric_instance = MetricWrapper(metric, **kwargs)

            self.function_for_metric[name] = metric_instance

    #region: score
    def score(self, y_true, y_pred):
        '''
        Score the predicted values for each metric.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Ground truth (correct) target values.
        y_pred : array-like of shape (n_samples,)
            Estimated target values.

        Returns
        ----------
        dict
            Mapping names of metrics to scores as floats.
        '''
        return {metric : score(y_true, y_pred) 
                for metric, score in self.function_for_metric.items()}
    #endregion

#region: MetricWrapper
class MetricWrapper:
    '''
    A class for wrapping metrics.

    Wraps individual metric functions, handling any additional parameters they
    may require. This makes it easy to use metrics that require parameters
    beyond the true and predicted values.

    Attributes
    ----------
    metric : callable
        The metric function to be used.
    kwargs : dict
        The keyword arguments for the metric function.

    Methods
    -------
    __call__(y_true, y_pred)
        Makes an instance callable, like the original metric function.
    '''
    def __init__(self, metric, **kwargs):
        '''
        Initialize the MetricWrapper.

        Parameters
        ----------
        metric : callable
            The metric function to be wrapped. Should take y_true, y_pred as input.
        **kwargs : dict, optional
            Additional keyword arguments that will be passed to the metric function.
        '''
        self.metric = metric
        self.kwargs = kwargs

    def __call__(self, y_true, y_pred):
        '''
        Call the wrapped metric function with the stored kwargs.

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

        Example
        -------
        >>> from sklearn.metrics import mean_squared_error
        >>> wrapped_mse = MetricWrapper(mean_squared_error)
        >>> y_true = [1, 2, 3]
        >>> y_pred = [1, 2, 2]
        >>> wrapped_mse(y_true, y_pred)
        0.3333
        '''
        return self.metric(y_true, y_pred, **self.kwargs)
#endregion