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

from feature_selection import FeatureSelector

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
    def __init__(self, results_manager, data_manager, exposure_file):
        '''
        Initialize the ResultsAnalyzer class.

        Parameters
        ----------
        results_manager : A `ResultsManager` instance
        data_manager : A `DataManager` instance
        exposure_file : str
            Path to the exposure data file.
        '''
        self.results_manager = results_manager
        self.data_manager = data_manager
        self._seem3_exposure_file = exposure_file
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

    #region: get_typical_pod_error
    def get_typical_pod_error(
            self, 
            model_key,
            metric='root_mean_squared_error'
            ):
        '''
        Helper function to get the typical POD prediction error.

        This error can be used to derive the POD prediction interval.

        Returns
        -------
        float
            The median RMSE from cross validation.

        See Also
        --------
        ResultsAnalyzer.prediction_interval()
        '''
        typical_rmse = (
            self.read_result(model_key, 'performances')[metric]
            .quantile()
        )
        return typical_rmse
    #endregion

    #region: pod_and_prediction_interval
    def pod_and_prediction_interval(
            self, 
            model_key, 
            inverse_transform=False, 
            normalize=False
            ):
        '''
        Compute Points of Departure (PODs) with uncertainty estimates.

        Parameters
        ----------
        model_key : tuple
            Key identifying the model to be analyzed.
        inverse_transform : bool, optional
            If True, applies the inverse transform to the predictions 
            (default is False).
        normalize : bool, optional
            If True, return cumulative frequencies (proportions) instead of 
            counts.

        Returns
        -------
        dict of pandas.Series
            - `pod` : sorted Points of Departure.
            - `cum_count`: Cumulative counts for the sorted PODs.
            - `lb`: Lower bound of the 90% prediction interval.
            - `ub`: Upper bound of the 90% prediction interval.
        '''
        y_pred, *_ = self.predict_out_of_sample(model_key)
        sorted_pods, cumulative_data = self.generate_cdf_data(
            y_pred, 
            normalize=normalize
            )
        
        rmse = self.get_typical_pod_error(model_key)  # log10-units
        lb, ub = self.prediction_interval(sorted_pods, rmse)
            
        if inverse_transform:
            sorted_pods, lb, ub = ResultsAnalyzer._inverse_log10(
                sorted_pods, lb, ub
                )

        pod_data = {
            'pod' : sorted_pods,
            'lb' : lb,
            'ub' : ub
            }
        ResultsAnalyzer._insert_cumulative_data(
            pod_data, 
            cumulative_data, 
            normalize
            )
        
        return pod_data
    #endregion

    #region: moe_and_prediction_intervals
    def moe_and_prediction_intervals(
            self, 
            model_key,
            inverse_transform=False, 
            normalize=False            
            ):
        '''
        Compute Margins of Exposure (MOEs) with uncertainty estimates.

        Two primary sources of uncertainty are addressed:
            1. Predicted PODs (hazard uncertainty) represented by a 90% 
               prediction interval.
            2. Exposure estimates, reflected by examining MOEs at different 
               exposure percentiles.
        
        Parameters
        ----------
        model_key : tuple
            Key identifying the model to be analyzed.
        inverse_transform : bool, optional
            If True, applies the inverse transform to the predictions 
            (default is False).
        normalize : bool, optional
            If True, return cumulative frequencies (proportions) instead of 
            counts.
        
        Returns
        -------
        dict of pandas.DataFrame
            A dictionary where keys are exposure percentiles. 
            Each corresponding value is a DataFrame containing:
            - `moe`: Sorted Margins of Exposure.
            - `cum_count`: Cumulative counts for the sorted MOEs.
            - `lb`: Lower bound of the 90% prediction interval.
            - `ub`: Upper bound of the 90% prediction interval.
        '''
        y_pred, *_ = self.predict_out_of_sample(model_key)
        
        exposure_df = self.load_exposure_data()
        moes = self.margins_of_exposure(y_pred, exposure_df)

        rmse = self.get_typical_pod_error(model_key)  # log10-units
        
        results_for_percentile = {}  # initialize
        
        for percentile in exposure_df.columns:
            
            sorted_moes, cumulative_data = self.generate_cdf_data(
                moes[percentile],
                normalize=normalize
                )
            
            lb, ub = self.prediction_interval(sorted_moes, rmse)
            
            if inverse_transform:
                sorted_moes, lb, ub = ResultsAnalyzer._inverse_log10(
                    sorted_moes, lb, ub
                    )

            moe_data = {
                    'moe': sorted_moes,
                    'lb': lb,
                    'ub': ub
                    }
            ResultsAnalyzer._insert_cumulative_data(
                moe_data, 
                cumulative_data, 
                normalize
                )
            results_for_percentile[percentile] = pd.DataFrame(moe_data)
            
        return results_for_percentile
    #endregion

    #region: generate_cdf_data
    @staticmethod
    def generate_cdf_data(data_series, normalize=False):
        '''
        Generate sorted values and their cumulative counts (or frequencies) for 
        CDF plotting.

        Parameters
        ----------
        data_series : pd.Series
            The data series to generate CDF data from.
        normalize : bool, optional
            If True, return cumulative frequencies (proportions) instead of 
            counts.

        Returns
        -------
        sorted_values : pd.Series
            Sorted values from the input data series.
        cumulative_data : np.ndarray
            Cumulative counts or frequencies for the sorted values.
        '''
        data_series = data_series.dropna()
        
        sorted_values = data_series.sort_values()
        cumulative_counts = np.arange(1, len(sorted_values) + 1)
        
        if normalize:
            cumulative_data = cumulative_counts / len(sorted_values)
        else:
            cumulative_data = cumulative_counts
            
        return sorted_values, cumulative_data
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
                self._seem3_exposure_file,
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

        ## Reproduce the feature selection using the stored settings

        config = self.results_manager.read_configuration()
        args = (
            config['feature_selection']['criterion_metric'],
            config['feature_selection']['n_features']
        )
        feature_names = (
            FeatureSelector.select_features(result_df, *args)
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

        ## Reproduce the feature selection using the stored settings
        
        config = self.results_manager.read_configuration()

        stride = (
            config['feature_selection']['n_splits_select']
            * config['feature_selection']['n_repeats_select'] 
            * config['feature_selection']['n_repeats_perm']
        )
        args = (
            config['feature_selection']['criterion_metric'],
            config['feature_selection']['n_features']
        )

        list_of_df = ResultsAnalyzer.split_replicates(result_df, stride)
        feature_names_for_replicate = {
            i : FeatureSelector.select_features(result_df, *args) 
            for i, result_df in enumerate(list_of_df)
            }
        
        return feature_names_for_replicate
    #endregion

    #region: get_pod_comparison_data
    def get_pod_comparison_data(self, model_key):
        '''
        Retrieve Point of Departure (POD) comparison data.
        
        Parameters
        ----------
        model_key : tuple
            The model key for which to retrieve the POD data.
            
        Returns
        -------
        dict
            A dictionary containing POD data for the given model key. The 
            dictionary has keys 'Regulatory', 'ToxValDB', and 'QSAR', each 
            mapping to a corresponding data series.

        See Also
        --------
        plot.cumulative_pod_distributions()
        '''        
        y_regulatory_df = self.load_regulatory_pods()
        model_key_names = self.read_model_key_names()
        key_for = dict(zip(model_key_names, model_key))
        effect = key_for['target_effect']
        results = self.get_in_sample_prediction(model_key)
        
        y_for_label = {
            'Regulatory': y_regulatory_df[effect].dropna(),
            'ToxValDB': results[0],
            'QSAR': results[-1]
        }
        
        return y_for_label
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

    #region: read_model_keys
    def read_model_keys(
            self, 
            inclusion_string=None, 
            exclusion_string=None
            ):
        '''Refer to `ResultsManager.read_model_keys` for documentation'''
        return self.results_manager.read_model_keys(
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
    def combine_results(self, result_type, model_keys=None):
        '''Refer to `ResultsManager.combine_results` for documentation'''
        return self.results_manager.combine_results(
            result_type, 
            model_keys=model_keys
            )
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

    #region: load_regulatory_pods
    def load_regulatory_pods(self):
        '''Refer to `DataManager.load_regulatory_pods` for documentation'''
        return self.data_manager.load_regulatory_pods()
    #endregion

    #region: load_oral_equivalent_doses
    def load_oral_equivalent_doses(self):
        '''
        Refer to `DataManager.load_oral_equivalent_doses` for documentation
        '''
        return self.data_manager.load_oral_equivalent_doses()
    #endregion

    #region: _inverse_log10
    @staticmethod
    def _inverse_log10(sorted_values, lb, ub):
        '''
        Helper function to transform data from log10-units to natural units.
        '''
        return (
            10**sorted_values, 
            10**lb, 
            10**ub
        )
    #endregion

    #region: _insert_cumulative_data
    @staticmethod
    def _insert_cumulative_data(data_dict, cumulative_data, normalize):
        '''
        Helper function to insert cumulative data into a dictionary. 

        The key is determined based on whether the data were normalized.
        '''
        if normalize:
            data_dict['cum_freq'] = cumulative_data
        else:
            data_dict['cum_count'] = cumulative_data
    #endregion