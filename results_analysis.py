'''
This module is responsible for analyzing the results of machine learning 
models. It includes functionalities for in-sample and out-of-sample 
predictions, feature importance analysis, and other result-related tasks.

Classes
-------
ResultsAnalyzer 
    Analyzes the results of machine learning models and provides methods for 
    predictions, feature importance, and more.
'''

import pandas as pd 
import numpy as np

from workflow_base import SupervisedLearningWorkflow

#region: ResultsAnalyzer.__init__
class ResultsAnalyzer:
    '''
    A class to analyze the results of machine learning models.

    This class provides methods to obtain in-sample and out-of-sample 
    predictions, determine important features, and perform other 
    result-related analyses.

    Attributes
    ----------
    results_manager : A `ResultsManager` instance for managing results data.
    data_manager : A `DataManager` instance for managing data.
    config : UnifiedConfiguration object

    Methods
    -------
    get_in_sample_prediction(model_key, inverse_transform=False) 
        Get in-sample predictions.
    predict_out_of_sample(model_key, inverse_transform=False) 
        Predict out-of-sample data.
    get_important_features(model_key) 
        Get important features for the model.
    get_important_features_replicates(model_key) 
        Get important features for each replicate.
    split_replicates(dataframe, stride) 
        Split a replicates DataFrame into individual DataFrames.
    '''
    def __init__(self, results_manager, data_manager, config):
        '''
        Initialize the ResultsAnalyzer class.

        Parameters
        ----------
        results_manager : A `ResultsManager` instance
        data_manager : A `DataManager` instance
        config : UnifiedConfiguration object
        '''
        self.results_manager = results_manager
        self.data_manager = data_manager
        self.config = config
#endregion

    #region: get_in_sample_prediction
    def get_in_sample_prediction(self, model_key, inverse_transform=False):
        '''
        Get in-sample predictions for the given model key.

        Parameters
        ----------
        model_key : Tuple
            Key identifying the model for which predictions are required.
        inverse_transform : bool, optional
            If True, applies the inverse transform to the predictions 
            (default is False).

        Returns
        -------
        y_pred : pandas.Series
            Predicted target values.
        X : pandas.DataFrame
            Features used for prediction.
        y_true : pandas.Series
            True target values.
        '''
        model_key_names = self.results_manager.read_model_key_names()
        key_for = dict(zip(model_key_names, model_key))
        # Load only the intersection of samples
        X, y_true = self.data_manager.load_features_and_target(**key_for)

        y_pred, X = self._get_prediction(model_key, X, inverse_transform)

        return y_pred, X, y_true
    #endregion

    #region: predict_out_of_sample
    def predict_out_of_sample(self, model_key, inverse_transform=False):
        '''
        Predict out-of-sample data for the given model key.

        Parameters
        ----------
        model_key : Tuple
            Key identifying the model for which predictions are required.
        inverse_transform : bool, optional
            If True, applies the inverse transform to the predictions 
            (default is False).

        Returns
        -------
        y_pred : pandas.Series
            Predicted target values.
        X : pandas.DataFrame
            Features used for prediction.
        '''
        model_key_names = self.results_manager.read_model_key_names()
        key_for = dict(zip(model_key_names, model_key))
        # Load the entire file
        X = self.data_manager.load_features(**key_for)

        y_pred, X = self._get_prediction(model_key, X, inverse_transform)

        return y_pred, X
    #endregion

    #region: _get_prediction
    def _get_prediction(self, model_key, X, inverse_transform=False):
        '''
        Get predictions for the given input.

        Parameters
        ----------
        model_key : Tuple
            Key identifying the model for which predictions are required.
        X : pandas.DataFrame
            Features used for prediction.
        inverse_transform : bool, optional
            If True, applies the inverse transform to the predictions 
            (default is False).

        Returns
        -------
        y_pred : pandas.Series
            Predicted target values.
        X : pandas.DataFrame
            Features used for prediction with fitted columns.
        '''
        estimator = self.results_manager.read_estimator(model_key)
        X = X[estimator.feature_names_in_]
        y_pred = pd.Series(estimator.predict(X), index=X.index)
        if inverse_transform:
            y_pred = 10**y_pred
        return y_pred, X
    #endregion

    #region: prediction_interval
    @staticmethod
    def prediction_interval(prediction, error, z_score=1.645):
        '''
        Calculate the prediction interval.

        Parameters
        ----------
        prediction : pd.Series
            Predicted values in log10 scale.
        error : float
            Measure of uncertainty, such as the Root Mean Squared Error.

        Returns
        -------
        lower_bound : pd.Series
            Lower bound of the prediction interval.
        upper_bound : pd.Series
            Upper bound of the prediction interval.
        '''
        lower_bound = prediction - z_score*error
        upper_bound = prediction + z_score*error
        return lower_bound, upper_bound
    #endregion

    #region: load_exposure_data
    def load_exposure_data(self, index_col='DTXSID', log10_transform=True):
        '''
        Load exposure data and optionally apply a log10 transformation.

        Parameters
        ----------
        index_col : str, optional
            Column name to set as the DataFrame index, default is 'DTXSID'.
        log10_transform : bool, optional
            If True, applies a log10 transformation to the exposure data, 
            default is True.

        Returns
        -------
        exposure_df : pandas.DataFrame
            DataFrame containing exposure predictions, with columns sorted.
        '''
        # TODO: Move to config for flexibility?
        sorted_columns = [
            '95th percentile (mg/kg/day)',
            '50th percentile (mg/kg/day)',
            '5th percentile (mg/kg/day)'
        ]

        exposure_df = (
            pd.read_csv(
                self.config.path.seem3_exposure_file,
                encoding='latin-1',
                index_col=index_col)
            [sorted_columns]
        )

        if log10_transform:
            exposure_df = np.log10(exposure_df)

        return exposure_df
    #endregion

    #region: margins_of_exposure
    @staticmethod
    def margins_of_exposure(pods, exposures, log10_units=True):
        '''
        Compute the margins of exposure (MOE) between predicted effects (pods) and 
        exposures. The function aligns the indices of pods and exposures, keeping 
        only their intersection, and computes the MOE accordingly.

        Parameters
        ----------
        pods : pandas.Series
            Predicted points of departure, indexed by chemical identifier.
        exposures : pandas.DataFrame or pandas.Series
            Exposure data, indexed by chemical identifier. If a DataFrame, each 
            column represents a different exposure estimate.
        log10_units : bool, optional
            If True, computes the MOE in log10 units (default is True). If False,
            returns the MOE in original units.

        Returns
        -------
        pandas.DataFrame or pandas.Series
            Margins of exposure, with the same shape as exposures, containing the 
            MOE values for the aligned indices.

        Notes
        -----
        The function aligns the indices of pods and exposures using an inner join,
        meaning that only the overlapping indices are included in the result.
        '''
        pods_aligned, exposures_aligned = pods.align(exposures, join='inner')

        operation = np.subtract if log10_units else np.divide

        if isinstance(exposures_aligned, pd.DataFrame):
            moe = exposures_aligned.apply(
                lambda col: operation(pods_aligned, col), 
                axis=0
                )
        else:
            moe = operation(pods_aligned, exposures_aligned)

        return moe
    #endregion

    #region: get_important_features
    def get_important_features(self, model_key):
        '''
        Get important features for the model identified by the given key.

        Parameters
        ----------
        model_key : Tuple
            Key identifying the model.

        Returns
        -------
        feature_names : list
            List of important feature names.
        '''
        result_df = self.results_manager.read_result(model_key, 'importances')

        # Get the parameters to reproduce the feature selection
        kwargs = self.config.model.kwargs_build_model
        args = (
            kwargs['criterion_metric'],
            kwargs['n_features']
        )
        
        feature_names = (
            SupervisedLearningWorkflow.select_features(result_df, *args)
        )
        return feature_names
    #endregion

    #region: get_important_features_replicates
    def get_important_features_replicates(self, model_key):
        '''
        Get important features for each replicate of the model identified by 
        the given key.

        Parameters
        ----------
        model_key : Tuple
            Key identifying the model.

        Returns
        -------
        feature_names_for_replicate : dict
            Dictionary mapping replicate index to the list of important 
            features.
        '''
        result_df = self.results_manager.read_result(
            model_key, 
            'importances_replicates'
            )

        # Get the parameters to reproduce the feature selection
        kwargs = self.config.model.kwargs_build_model
        stride = (
            kwargs['n_splits_select']
            * kwargs['n_repeats_select'] 
            * kwargs['n_repeats_perm']
        )
        args = (
            kwargs['criterion_metric'],
            kwargs['n_features']
        )

        list_of_df = ResultsAnalyzer.split_replicates(result_df, stride)

        feature_names_for_replicate = {
            i : SupervisedLearningWorkflow.select_features(result_df, *args) 
            for i, result_df in enumerate(list_of_df)
            }
        return feature_names_for_replicate
    #endregion

    #region: split_replicates
    @staticmethod
    def split_replicates(dataframe, stride):
        '''
        Split a replicates DataFrame into individual DataFrames.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            DataFrame containing replicates data.
        stride : int
            Stride to use for splitting the DataFrame.

        Returns
        -------
        list_of_df : list
            List of DataFrames, each containing a subset of replicates.
        '''
        list_of_df = []
        length = len(dataframe)
        start = 0

        while start < length:
            end = start + stride
            subset = dataframe.iloc[start:end]
            list_of_df.append(subset)
            start = end
            
        return list_of_df
    #endregion

    #region: list_model_keys
    def list_model_keys(
            self, 
            inclusion_string=None, 
            exclusion_string=None
            ):
        '''Refer to `ResultsManager.list_model_keys` for documentation'''
        return self.results_manager.list_model_keys(
            inclusion_string, 
            exclusion_string
            )
    #endregion

    #region: group_model_keys
    def group_model_keys(
            self, 
            exclusion_key_names, 
            string_to_exclude=None, 
            model_keys=None
            ):
        '''Refer to `ResultsManager.group_model_keys` for documentation'''
        return self.results_manager.group_model_keys(
            exclusion_key_names, 
            string_to_exclude, 
            model_keys
            )
    #endregion

    #region: read_model_key_names
    def read_model_key_names(self):
        '''Refer to `ResultsManager.read_model_key_names` for documentation'''
        return self.results_manager.read_model_key_names()
#endregion

    #region: read_result
    def read_result(self, model_key, result_type):
        '''Refer to `ResultsManager` for documentation'''
        return self.results_manager.read_result(model_key, result_type)
#endregion

    #region: combine_results   
    def combine_results(self, model_keys, result_type):
        '''Refer to `ResultsManager.combine_results` for documentation'''
        return self.results_manager.combine_results(model_keys, result_type)
    #endregion

    #region: load_features_and_target    
    def load_features_and_target(self, *args, **kwargs):
        '''
        Refer to `DataManager.load_features_and_target` for documentation
        '''
        return self.data_manager.load_features_and_target(*args, **kwargs)
    #endregion

    #region: load_features
    def load_features(self, *args, **kwargs):
        '''Refer to `DataManager.load_features` for documentation'''
        return self.data_manager.load_features(*args, **kwargs)
    #endregion

    #region: load_target
    def load_target(self, *args, **kwargs):
        '''Refer to `DataManager.load_target` for documentation'''
        return self.data_manager.load_target(*args, **kwargs)
    #endregion