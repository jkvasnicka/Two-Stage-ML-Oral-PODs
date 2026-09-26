'''
This module contains the `ResultsAnalyzer` class, which is responsible for 
analyzing the results of the modeling workflows. It includes functionalities 
for in-sample and out-of-sample predictions, feature importance analysis, and 
other result-related tasks.
'''

import pandas as pd 
import numpy as np
import itertools

from feature_selection import FeatureSelector

# NOTE: For backwards compatibility
from plotting import sensitivity_analysis  


# Bounds are in log10 MOE; the upper bound belongs to each category.
MOE_CATEGORIES = {
    'Low Concern': (2., np.inf),
    'Moderate Concern': (0., 2.),
    'High Concern': (-np.inf, 0.),
}


def classify_moe(moes):
    '''Classify a log10 MOE Series; nonfinite values remain unclassified.'''
    labels = ['High Concern', 'Moderate Concern', 'Low Concern']
    bounds = [-np.inf] + [MOE_CATEGORIES[label][1] for label in labels]
    return pd.cut(moes.where(np.isfinite(moes)), bounds, labels=labels)


def pod_moe_results(analyzer, model_keys, exposure, pod_transform=None):
    '''Calculate endpoint application PODs and MOEs from fitted estimators.

    Parameters
    ----------
    analyzer : ResultsAnalyzer
        Supplies predictions, model-key metadata, and CV RMSE.
    model_keys : iterable of tuple
        One model per endpoint. Training chemicals are excluded.
    exposure : pandas.DataFrame
        Log10 intake estimates, shared unchanged between compared routes.
    pod_transform : callable, optional
        Converts the native log10 POD result frame to exposure-compatible
        dose units, including bounds, preserving index and CDF columns.
        Omit when native POD units already match exposure units.

    Returns
    -------
    dict
        Endpoint keys mapped to ``pod`` and ``moe`` result frames. PODs
        retain all application chemicals; MOEs retain exposure matches.
    '''
    effect_index = analyzer.read_model_key_names().index('target_effect')
    results = {}
    for key in analyzer.validate_model_keys(model_keys):
        effect = key[effect_index]
        if effect in results:
            raise ValueError('Specify only one model per endpoint.')
        pods = analyzer.pod_and_prediction_interval(key)
        if pod_transform is not None:
            pods = pod_transform(pods)
        moes = analyzer.moe_from_pods(
            pods['pod'], exposure, analyzer.get_typical_pod_error(key),
        )
        results[effect] = {'pod': pods, 'moe': moes}
    return results


def summarize_moe(results, exposure, label_for_effect):
    '''Return baseline manuscript counts and typical exposure uncertainty.

    Parameters
    ----------
    results : dict
        Endpoint results from ``pod_moe_results()`` for either route.
    exposure : pandas.DataFrame
        The unchanged log10 exposure inputs used to calculate the MOEs.
    label_for_effect : dict
        Manuscript endpoint labels.

    Returns
    -------
    summary : pandas.DataFrame
        Application/exposure denominators and cumulative concern counts
        using the lower hazard bound at upper exposure.
    exposure_uncertainty : float
        Median log10 exposure interval width over the union of endpoint
        application populations. Display precision is left to the caller.
    '''
    rows = {}
    application = pd.Index([])
    upper = '95th percentile (mg/kg/day)'
    lower = '5th percentile (mg/kg/day)'
    for effect, model in results.items():
        pods, moes = model['pod'], model['moe'][upper]['lb']
        application = application.union(pods.index)
        rows[label_for_effect[effect]] = {
            'Application chemicals': len(pods),
            'Chemicals with exposure': len(moes),
            'MOE <= 1': int(moes.le(0).sum()),
            'MOE <= 100': int(moes.le(2).sum()),
        }
    exposure = exposure.loc[exposure.index.intersection(application)]
    uncertainty = (exposure[upper] - exposure[lower]).median()
    return pd.DataFrame.from_dict(rows, orient='index'), uncertainty


def route_allocation_results(oral_results, inhalation_results):
    '''Pair application MOEs and classify conservative screening estimates.

    Parameters
    ----------
    oral_results, inhalation_results : dict
        Endpoint results from ``pod_moe_results()``. Both routes must use
        the same unchanged exposure inputs and compatible dose units.
        Training exclusions are inherited from each route's predictions.

    Returns
    -------
    paired : dict of pandas.DataFrame
        Eligible DTXSIDs by endpoint, with (route, exposure, statistic)
        columns. Retains MOE points and bounds at all three exposures and
        lower-hazard/upper-exposure concern labels. Cumulative counts are omitted
        because the population changes after pairing.
    '''
    if oral_results.keys() != inhalation_results.keys():
        raise ValueError('Oral and inhalation endpoints must match.')
    exposures = [f'{p}th percentile (mg/kg/day)' for p in (50, 5, 95)]
    upper = exposures[-1]
    paired = {}
    for effect in oral_results:
        routes = {}
        for route, results in (
                ('oral', oral_results[effect]['moe']),
                ('inhalation', inhalation_results[effect]['moe'])):
            frames = {}
            for exposure in exposures:
                frame = results[exposure].loc[:, ['moe', 'lb', 'ub']]
                if not frame.index.is_unique or frame.index.hasnans:
                    raise ValueError('MOE DTXSIDs must be unique/nonmissing.')
                frames[exposure] = frame
            routes[route] = pd.concat(frames, axis=1, join='inner')
        table = pd.concat(routes, axis=1, join='inner')
        table = table.loc[np.isfinite(table).all(axis=1)].copy()
        for route in routes:
            table[(route, upper, 'concern')] = classify_moe(
                table[(route, upper, 'lb')]
            )
        paired[effect] = table
    return paired


def route_allocation_table(oral_results, inhalation_results, label_for_effect):
    '''Format endpoint matrices with margins and whole-population percentages.

    Parameters
    ----------
    oral_results, inhalation_results : dict
        Endpoint application results using identical exposure inputs; see
        ``route_allocation_results()`` for pairing and eligibility.
    label_for_effect : dict
        Manuscript labels for endpoint keys.

    Returns
    -------
    pandas.DataFrame
        Counts and percentages to two decimal places; small nonzero values
        are shown as <0.01%. Empty populations
        show ``NA`` percentages rather than implying an observed fraction.
    '''
    paired = route_allocation_results(oral_results, inhalation_results)
    panels = {}
    for effect, frame in paired.items():
        matrix = _route_concern_counts(frame)
        n = len(frame)
        totals = matrix.copy()
        totals['Total'] = matrix.sum(axis=1)
        totals.loc['Total'] = totals.sum(axis=0)
        panels[label_for_effect[effect]] = totals.applymap(
            lambda count: _format_concern_count(count, n)
        )
    return pd.concat(panels).rename_axis(
        index=['Endpoint', 'Oral concern'],
        columns='Inhalation concern, n (%)',
    )


def _route_concern_counts(table):
    '''Count all nine screening-concern combinations, including zero cells.'''
    upper = '95th percentile (mg/kg/day)'
    order = ['High Concern', 'Moderate Concern', 'Low Concern']
    return pd.crosstab(
        table[('oral', upper, 'concern')],
        table[('inhalation', upper, 'concern')],
    ).reindex(index=order, columns=order, fill_value=0).fillna(0).astype(int)


def _format_concern_count(count, total):
    '''Format an exact count without rounding a small nonzero percent to zero.'''
    if not total:
        return f'{count:,} (NA)'
    percent = 100 * count / total
    label = '<0.01' if 0 < percent < 0.01 else f'{percent:.2f}'
    return f'{count:,} ({label})'


def pod_to_effect_level(pod, factor=3.49):
    '''
    Convert POD values to 10%-incidence population effect levels.

    The default factor of 3.49 is the best-estimate (P50) human variability
    factor used by Aurisano et al. to convert a population-median effect level
    (50% incidence) to a 10%-incidence human population effect level. Thus,
    the conversion yields an oral human effect dose (HD_M^10%) or inhalation
    human effect concentration (HC_M^10%), according to the POD units.

    Sources
    -------
    Aurisano et al. (2023), doi:10.1289/EHP11524.
    Aurisano et al. (2024), doi:10.1021/acs.est.4c00207.

    Parameters
    ----------
    pod : scalar or array-like
        Point-of-departure values. NumPy arrays and pandas objects retain
        their input shape and labels.
    factor : float, optional
        Positive human variability factor. Default is 3.49. Supply another
        value to override the literature-based default.

    Returns
    -------
    scalar or array-like
        Effect-level values of the same general type as ``pod``.
    '''
    if factor <= 0:
        raise ValueError("'factor' must be greater than zero.")
    return pod / factor

#region: ResultsAnalyzer.__init__
class ResultsAnalyzer:
    '''
    A class to analyze the results of machine learning models.

    This class provides methods to obtain in-sample and out-of-sample 
    predictions, determine important features, and perform other 
    result-related analyses.
    '''
    def __init__(self, results_manager, data_manager, plot_settings):
        '''
        Initialize the ResultsAnalyzer class.

        Parameters
        ----------
        results_manager : A `ResultsManager` instance
        data_manager : A `DataManager` instance
        plot_settings : SimpleNamespace
            Configuration settings related to plotting.
        '''
        self.results_manager = results_manager
        self.data_manager = data_manager
        self.plot_settings = plot_settings
#endregion

    # FIXME: Appears that inverse_transform only applied to y_pred, not y_true?
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

    # TODO: Add optional inverse_transform?
    #region: get_out_sample_prediction
    def get_out_sample_prediction(self, model_key, aggregation='mean'):
        '''
        Get out-of-sample aggregated predictions for the given model key.

        Aggregates predictions across cross-validation replicates for each
        chemical, providing a single prediction per chemical based on the 
        specified aggregation method (e.g., mean).

        Parameters
        ----------
        model_key : Tuple
            Key identifying the model for which predictions are required.
        aggregation : str
            Type of aggregation to apply ('mean', 'median', etc.). Must be a 
            valid method of pd.core.groupby.GroupBy.

        Returns
        -------
        y_pred_agg : pandas.Series
            Aggregated predicted target values.
        y_true : pandas.Series
            True target values.
        '''
        # Load the observed values for comparison
        model_key_names = self.results_manager.read_model_key_names()
        key_for = dict(zip(model_key_names, model_key))
        y_true = self.data_manager.load_target(**key_for)

        # Get the out-of-sample predictions
        predictions = self.results_manager.read_result(model_key, 'predictions').squeeze()

        if not hasattr(pd.core.groupby.GroupBy, aggregation):
            raise ValueError(f"Aggregation method '{aggregation}' is not valid.")

        # Perform the aggregation
        group = predictions.groupby(level=0)
        y_pred_agg = getattr(group, aggregation)()

        prediction_chemicals = list(y_pred_agg.index.unique(level=0))
        return y_pred_agg, y_true[prediction_chemicals]
    #endregion

    #region: predict
    def predict(
            self, 
            model_key, 
            inverse_transform=False, 
            exclude_training=False
            ):
        '''
        Make prediction for the given model key.

        Parameters
        ----------
        model_key : Tuple
            Key identifying the model for which predictions are required.
        inverse_transform : bool, optional
            If True, applies the inverse transform to the predictions 
            (default is False).
        exclude_training : bool, optional
            If True, excludes chemicals used for model training. Default is 
            False; predictions are made for all chemicals with features.

        Returns
        -------
        y_pred : pandas.Series
            Predicted target values.
        X : pandas.DataFrame
            Features used for prediction.
        '''
        model_key_names = self.results_manager.read_model_key_names()
        key_for = dict(zip(model_key_names, model_key))

        X = self.data_manager.load_features(
            **key_for, 
            exclude_training=exclude_training
            )

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
            normalize=False, 
            exclude_training=True
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
        exclude_training : bool, optional
            If True, excludes chemicals used for model training. Default is 
            True, because these chemicals already have labeled data.

        Returns
        -------
        pandas.DataFrame
            - `pod` : sorted Points of Departure.
            - `cum_count`: Cumulative counts for the sorted PODs.
            - `lb`: Lower bound of the 90% prediction interval.
            - `ub`: Upper bound of the 90% prediction interval.
        '''
        y_pred, *_ = self.predict(model_key, exclude_training=exclude_training)
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
        
        return pd.DataFrame(pod_data)
    #endregion

    #region: moe_and_prediction_intervals
    def moe_and_prediction_intervals(
            self, 
            model_key,
            inverse_transform=False, 
            normalize=False,
            exclude_training=True           
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
        exclude_training : bool, optional
            If True, excludes chemicals used for model training. Default is 
            True, because these chemicals already have labeled data.
        
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
        y_pred, *_ = self.predict(model_key, exclude_training=exclude_training)
        
        exposure_df = self.data_manager.load_exposure_data()
        rmse = self.get_typical_pod_error(model_key)
        return self.moe_from_pods(
            y_pred, exposure_df, rmse, inverse_transform, normalize,
        )
    #endregion

    @staticmethod
    def moe_from_pods(
            predictions, exposure_df, rmse,
            inverse_transform=False, normalize=False):
        '''Summarize unit-compatible log10 POD/exposure results.

        Parameters
        ----------
        predictions : pandas.Series
            Log10 PODs in exposure-compatible dose units, indexed by DTXSID.
        exposure_df : pandas.DataFrame
            Log10 intake estimates, one column per uncertainty percentile.
        rmse : float
            Model-specific median CV RMSE in log10 units.
        inverse_transform, normalize : bool, optional
            Return linear values or cumulative proportions, respectively.

        Returns
        -------
        dict of pandas.DataFrame
            Per-exposure MOE points, hazard bounds, and cumulative counts
            (or proportions), using the original oral calculation.
        '''
        moes = ResultsAnalyzer.margins_of_exposure(predictions, exposure_df)
        results_for_percentile = {}  # initialize

        for percentile in exposure_df.columns:

            sorted_moes, cumulative_data = ResultsAnalyzer.generate_cdf_data(
                moes[percentile],
                normalize=normalize
                )

            lb, ub = ResultsAnalyzer.prediction_interval(sorted_moes, rmse)

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
            dictionary has keys 'Authoritative', 'ToxValDB', and 'QSAR', each 
            mapping to a corresponding data series.

        See Also
        --------
        plot.cumulative_pod_distributions()
        '''        
        model_key_names = self.read_model_key_names()
        key_for = dict(zip(model_key_names, model_key))
        
        y_auth = (
            self.load_authoritative_pods()
            [key_for['target_effect']]
            .dropna()
        )
        _, y_true = self.data_manager.load_features_and_target(**key_for)
        y_pred, _ = self.predict(model_key)
        
        y_for_label = {
            self.plot_settings.authoritative_label: y_auth,
            self.plot_settings.surrogate_label: y_true,
            self.plot_settings.qsar_label: y_pred
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

    #region: summarize_model_performances
    def summarize_model_performances(self, model_keys=None, quantiles=None):
        '''
        Generate a statistical summary table comparing performance scores of each 
        model.

        Parameters
        ----------
        model_keys : list of tuple, optional
            List of model keys for which to retrieve the results. If None, 
            then all model keys will be used.
        quantiles : float or array-like, optional
            Value between 0 <= q <= 1, the quantile(s) to compute. Defaults to 90%
            and 95% confidence intervals.
            
        Returns
        -------
        pandas.DataFrame
            The index has three levels: effect, model_name, metric. The columns are
            the quantiles.
        '''
        if not quantiles:
            quantiles = [0.025, 0.05, 0.5, 0.95, 0.975]  # by default

        performances = self.results_manager.combine_results(
            'performances', 
            model_keys=model_keys
        )
        # Get the results for the metrics of interest
        metrics = list(self.plot_settings.label_for_metric)
        performances = performances.loc[
            :, performances.columns.get_level_values('metric').isin(metrics)
        ]
        # Use the nice labels in the config file
        performances = performances.rename(
            self.plot_settings.label_for_metric, 
            level='metric', 
            axis=1
        )

        model_key_names = self.results_manager.read_model_key_names()

        performances_for = {}  # initialize

        for effect, effect_label in self.plot_settings.label_for_effect.items():
            performances_for[effect_label] = sensitivity_analysis.prepare_data_for_plotting(
                performances, 
                effect, 
                self.data_manager, 
                model_key_names, 
                self.plot_settings
            )

        # Create a statistical summary table
        performance_summary = (
            pd.concat(performances_for, axis=1)
            .quantile(quantiles)
            .T
        )
        # Name the first index level
        performance_summary.index.names = ['effect'] + performance_summary.index.names[1:]

        return performance_summary
    #endregion

    #region: model_performance_table
    def model_performance_table(
            self,
            model_keys=None,
            quantiles=(0.05, 0.5, 0.95),
            decimals=2,
            ):
        '''Return a formatted manuscript performance table.

        Parameters
        ----------
        model_keys : list of tuple, optional
            Model keys to include.
        quantiles : sequence of float, optional
            Lower, median, and upper quantiles to display.
        decimals : int, optional
            Number of decimal places to display.

        Returns
        -------
        pandas.DataFrame
            Four-column table with effect section rows and one row per model.
        '''
        quantiles = tuple(quantiles)
        if len(quantiles) != 3:
            raise ValueError(
                'quantiles must contain lower, median, and upper values.'
            )

        lower, median, upper = quantiles
        summary = self.summarize_model_performances(
            model_keys=model_keys,
            quantiles=quantiles,
        )

        def format_value(value):
            rounded = round(float(value), decimals)
            if rounded == 0:
                rounded = 0
            return f'{rounded:.{decimals}f}'

        def format_interval(values):
            return (
                f'{format_value(values[median])} '
                f'[{format_value(values[lower])}–'
                f'{format_value(values[upper])}]'
            )

        row_order = summary.index.droplevel('metric')
        row_order = row_order[~row_order.duplicated()]
        table = (
            summary.apply(format_interval, axis=1)
            .unstack('metric')
            .reindex(row_order)
        )
        table = table.reindex(
            columns=list(self.plot_settings.label_for_metric.values())
        )
        table = table.rename(columns={'$R^2$': 'R²'})

        section_label_for_effect = {
            'General Noncancer': 'General non-cancer effects',
            'Reproductive/Developmental': (
                'Reproductive/developmental effects'),
        }
        rows = []
        for effect in table.index.get_level_values('effect').unique():
            section_label = section_label_for_effect.get(
                effect, f'{effect}')
            rows.append([section_label, '', '', ''])
            for model_name, values in table.loc[effect].iterrows():
                if ' (final) ' in model_name:
                    model_name = model_name.replace(
                        ' (final)', ' with feature selection')
                elif model_name.startswith('RDKit Features'):
                    model_name = f'**{model_name}'
                else:
                    model_name = f'*{model_name}'
                rows.append([model_name, *values])

        return pd.DataFrame(
            rows,
            columns=['QSAR Model (n)', 'RMSE', 'MedAE', 'R²'],
        )
    #endregion

    #region: describe
    def describe(self, model_key, result_type, percentiles=None):
        '''
        Describe model performances by generating a summary of selected 
        metrics.

        Parameters
        ----------
        model_key : str
            Identifier for the model to summarize.
        result_type : str
            Specifies the type of results to describe.
        percentiles : list of float, optional
            The percentiles to include in the output. Uses pandas default.

        Returns
        -------
        pandas.DataFrame
        '''
        if 'importances' in result_type:
            metrics = list(self.plot_settings.label_for_scoring)
        else: 
            metrics = list(self.plot_settings.label_for_metric)

        performances = self.read_result(model_key, result_type)

        if 'root_mean_squared_error' in metrics:
            metrics.append('gsd')
            metrics.append('gsd_squared')
            rmse = performances['root_mean_squared_error']
            gsd, gsd_squared = self._calculate_gsd_with_confidence(rmse)
            performances['gsd'] = gsd
            performances['gsd_squared'] = gsd_squared

        return performances.describe(percentiles=percentiles)[metrics]
    #endregion

    #region: _calculate_gsd_with_confidence
    @staticmethod
    def _calculate_gsd_with_confidence(rmse, z_score=1.96):
        '''
        Calculate the geometric standard deviation (GSD) and its 
        confidence-adjusted value.

        Parameters
        ----------
        rmse : float
            The root mean squared error from which the GSD is derived.
        z_score : float, optional
            The z-score corresponding to the desired confidence interval. 
            Defaults to 1.96, which corresponds to approximately a 95% 
            confidence interval.
        '''
        gsd = 10 ** rmse  # in natural units
        gsd_adjusted = gsd ** z_score
        return gsd, gsd_adjusted
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
            ignore_components,
            exclusion_string=None,
            model_keys=None,
            filter_single_key_groups=True
        ):
        '''
        Group model keys by forming a new grouping key for each model key,
        achieved by ignoring specified components. This groups model keys 
        that share the same modified grouping key, enabling analysis based on 
        grouped similarities.

        Parameters
        ----------
        ignore_components : str or list of str
            The component name or names of the model keys to be ignored when 
            forming the grouping key.
        exclusion_string : str, optional
            Specifies a substring to filter out keys containing it. If None, 
            no filtering is performed.
        model_keys : list of tuples, optional
            The set of model keys to be grouped. Each tuple represents a 
            complete model key. If None, keys are fetched from the 
            ResultsManager object.
        filter_single_key_groups : bool, optional
            If True, groups with only one model key are excluded from the 
            output.

        Returns
        -------
        grouped_model_keys : list of tuples
            Each tuple consists of a grouping key and a list of model keys 
            sharing this grouping key.
        '''
        if model_keys is None:
            # Use all available model keys.
            model_keys = self.read_model_keys()

        model_keys = ResultsAnalyzer.validate_model_keys(model_keys)

        if isinstance(ignore_components, str):
            ignore_components = [ignore_components]

        if exclusion_string:
            # Filter out model keys containing the specified substring
            model_keys = [
                k for k in model_keys if exclusion_string not in k
                ]

        # Get indices of components to ignore based on their names
        exclusion_key_indices = [
            self.read_model_key_names().index(key)
            for key in ignore_components
            ]

        def create_grouping_key(model_key):
            return tuple(item for idx, item in enumerate(model_key)
                        if idx not in exclusion_key_indices)

        # Sort model keys by their new grouping keys
        sorted_model_keys = sorted(model_keys, key=create_grouping_key)

        # Group the sorted model keys by their new grouping keys
        grouped_model_keys = [
            (grouping_key, list(group))
            for grouping_key, group in itertools.groupby(
            sorted_model_keys, key=create_grouping_key)
        ]

        if filter_single_key_groups:
            # Remove groups containing only one model key
            grouped_model_keys = [
                (grouping_key, group)
                for grouping_key, group in grouped_model_keys
                if len(group) > 1
            ]

        return grouped_model_keys
    #endregion

    #region: validate_model_keys
    @staticmethod
    def validate_model_keys(model_keys):
        '''
        Validate and convert model_keys to a list of tuples if necessary. 

        This function allows model keys to be stored in JSON files as lists and 
        converted into tuples post-loading.

        Parameters
        ----------
        model_keys : list of tuples or list of lists
            If the model keys are provided as lists, they will be converted to 
            tuples.

        Returns
        -------
        model_keys : list of tuples
        '''
        if all(isinstance(model_key, list) for model_key in model_keys):
            model_keys = [tuple(model_key) for model_key in model_keys]
        return model_keys
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

    #region: load_authoritative_pods
    def load_authoritative_pods(self):
        '''Refer to `DataManager.load_authoritative_pods` for documentation'''
        return self.data_manager.load_authoritative_pods()
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
