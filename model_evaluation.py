'''
This module defines the `ModelEvaluator` class, responsible for evaluating 
machine learning models using repeated k-fold cross-validation. The class 
provides the flexibility to evaluate models both with and without feature 
selection, using a set of predefined metrics. It leverages parallel processing 
to efficiently handle large datasets.
'''

import pandas as pd
from sklearn.model_selection import RepeatedKFold
from joblib import Parallel, delayed

#region: ModelEvaluator.__init__
class ModelEvaluator:
    '''
    Class responsible for evaluating machine learning models using different 
    evaluation criteria, possibly with or without feature selection.
    '''
    def __init__(
            self, evaluation_settings, metrics_manager, feature_selector=None, 
            n_jobs=None):
        '''
        Initialize the ModelEvaluator.

        Parameters
        ----------
        evaluation_settings : SimpleNamespace
            The configuration settings for the model evaluation.
        metrics_manager : MetricsManager
            The manager that handles different evaluation metrics.
        feature_selector : FeatureSelector, optional
            An instance of a FeatureSelector to handle feature selection 
            (default is None).
        n_jobs : int, optional
            See joblib.Parallel for reference.
        '''
        self.evaluation_settings = evaluation_settings
        self.metrics_manager = metrics_manager
        self.feature_selector = feature_selector
        self._n_jobs = n_jobs
#endregion

    #region: cross_validate_model
    def cross_validate_model(self, estimator, X, y, select_features=False):
        '''
        Evaluate the model with or without feature selection.

        Parameters
        ----------
        estimator : object
            The machine learning estimator to be evaluated.
        X : pandas.DataFrame
            The complete set of features data.
        y : pandas.Series
            The complete set of target data.
        select_features : bool, optional
            Whether to include feature selection in the evaluation (default is 
            False).

        Returns
        -------
        dict
            Evaluation results.
        '''
        if select_features:
            return self._cross_validate_with_selection(estimator, X, y)
        else:
            return self._cross_validate_without_selection(estimator, X, y)
    #endregion

    #region: _cross_validate_with_selection
    def _cross_validate_with_selection(self, estimator, X, y):
        '''
        Evaluate the model with nested feature selection.

        A repeated k-fold cross validation is used to evaluate performance. A
        feature selection scheme is nested within.

        Parameters
        ----------
        estimator : object
            The machine learning estimator to be evaluated.
        X : pandas.DataFrame
            The complete set of features data.
        y : pandas.Series
            The complete set of target data.

        Returns
        -------
        dict
            Evaluation results.
        '''
        estimator, performances, importances_replicates, predictions = (
            self._evaluate_with_repeated_kfold_and_selection(estimator, X, y)
        )

        evaluation_results = {
            'performances': performances,
            'importances_replicates': importances_replicates,
            'predictions': predictions
        }
        return evaluation_results
    #endregion

    #region: _evaluate_with_repeated_kfold_and_selection
    def _evaluate_with_repeated_kfold_and_selection(self, estimator, X, y):
        '''
        Execute a repeated k-fold cross-validation with nested feature selection.

        Parameters
        ----------
        estimator : object
            The machine learning estimator to be evaluated.
        X : pandas.DataFrame
            The complete set of features data.
        y : pandas.Series
            The complete set of target data.

        Returns
        -------
        estimator : object
            The fitted estimator with selected features.
        performances : pandas.DataFrame
            A DataFrame containing performance results across different folds.
        importances_replicates : pandas.DataFrame
            A DataFrame containing feature importances across different folds.
        predictions : pandas.DataFrame
            A DataFrame containing out-of-sample predictions for each chemical,
            with a MultiIndex tracking cross-validation folds and replicates.

        Notes
        -----
        This method involves nested cross-validation where feature selection is 
        performed within each fold of the cross-validation process.
        '''
        performances, importances_replicates, predictions_data = [], [], []

        rkf_cv = RepeatedKFold(
            n_splits=self.evaluation_settings.n_splits_cv, 
            n_repeats=self.evaluation_settings.n_repeats_cv, 
            random_state=self.evaluation_settings.random_state_cv
        )

        for replicate_num, (train_ix, test_ix) in enumerate(rkf_cv.split(X)):
            X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
            y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

            estimator, important_features, importances = (
                self.feature_selector.nested_feature_selection(
                estimator, X_train, y_train)
            )

            importances_replicates.append(importances)

            estimator.fit(X_train[important_features], y_train)

            y_pred = estimator.predict(X_test[important_features])
            performances.append(self.metrics_manager.score(y_test, y_pred))

            for ix, pred in zip(X_test.index, y_pred):
                predictions_data.append((ix, replicate_num, pred))

        predictions = _create_predictions_dataframe(predictions_data, X)

        performances = pd.DataFrame(performances)
        performances.columns.names = ['metric']
        importances_replicates = pd.concat(importances_replicates)

        return estimator, performances, importances_replicates, predictions
    #endregion

    #region: _cross_validate_without_selection
    def _cross_validate_without_selection(self, estimator, X, y):
        '''
        Evaluate the model without feature selection.

        A repeated k-fold cross validation is used to evaluate performance.

        Parameters
        ----------
        estimator : object
            The estimator to evaluate.
        X : pandas.DataFrame
            The features data.
        y : pandas.Series
            The target data.

        Returns
        -------
        dict
            Evaluation results.
        '''
        estimator, performances, predictions = (
            self._evaluate_with_repeated_kfold(estimator, X, y)
        )

        # Fit the model to all data.
        estimator.fit(X, y)

        evaluation_results = {
            'performances': performances,
            'predictions': predictions
        }
        return evaluation_results
    #endregion

    #region: _evaluate_with_repeated_kfold
    def _evaluate_with_repeated_kfold(self, estimator, X, y):
        '''
        Execute a repeated k-fold cross-validation on the entire dataset.

        Parameters
        ----------
        estimator : object
            The machine learning estimator to be evaluated.
        X : pandas.DataFrame
            The complete set of features data.
        y : pandas.Series
            The complete set of target data.

        Returns
        -------
        estimator : object
            The fitted estimator.
        performances : pandas.DataFrame
            A DataFrame containing performance results across different folds.
        predictions : pandas.DataFrame
            A DataFrame containing out-of-sample predictions for each chemical,
            with a MultiIndex tracking cross-validation folds and replicates.

        Notes
        -----
        This method facilitates parallelized execution of cross-validation 
        using joblib's Parallel and delayed functions.
        '''
        rkf = RepeatedKFold(
            n_splits=self.evaluation_settings.n_splits_cv, 
            n_repeats=self.evaluation_settings.n_repeats_cv, 
            random_state=self.evaluation_settings.random_state_cv
        )

        results = Parallel(n_jobs=self._n_jobs)(
            delayed(self._split_fit_predict_and_score)(
                estimator, X, y, train_ix, test_ix
            ) for train_ix, test_ix in rkf.split(X)
        )

        # Unpack the results from parallel executions
        performances, predictions_data = [], []
        for replicate_num, (score, pred_data) in enumerate(results):
            performances.append(score)
            for ix, pred in pred_data:
                predictions_data.append((ix, replicate_num, pred))

        predictions = _create_predictions_dataframe(predictions_data, X)

        performances = pd.DataFrame(performances)
        performances.columns.names = ['metric']

        return estimator, performances, predictions
    #endregion

    #region: _split_fit_predict_and_score
    def _split_fit_predict_and_score(self, estimator, X, y, train_ix, test_ix):
        '''
        Perform a single split of the data, fit the estimator, predict on the 
        test set, and score the performance. 
        
        This function is designed to be used within a cross-validation loop.

        Parameters
        ----------
        estimator : object
            The machine learning estimator to be trained and evaluated.
        X : pandas.DataFrame
            The complete set of features data.
        y : pandas.Series
            The complete set of target data.
        train_ix : array-like
            Indices for the training set in the current fold.
        test_ix : array-like
            Indices for the test set in the current fold.

        Returns
        -------
        tuple
            A tuple containing the performance score of the estimator on the 
            test data and a list of tuples. Each tuple in the list consists of 
            a test index and the corresponding prediction, facilitating 
            tracking of predictions across folds and replicates in 
            cross-validation.

        Notes
        -----
        This method is a utility function for parallelized execution of 
        cross-validation steps, used in conjunction with joblib's Parallel and 
        delayed functions.
        '''
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        score = self.metrics_manager.score(y_test, y_pred)

        return score, list(zip(X_test.index, y_pred))
    #endregion

#region: _create_predictions_dataframe
def _create_predictions_dataframe(predictions_data, X):
    '''
    Create a DataFrame containing test-set (out-of-sample) predictions for 
    each chemical.
    
    A MultiIndex is used to track cross-validation folds/replicates for each
    chemical.

    Parameters
    ----------
    predictions_data : list of tuples
        List of tuples (chemical/index, replicate number, prediction).
    X : pandas.DataFrame
        The features data used in cross-validation.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing individual predictions with a MultiIndex.

    Notes
    -----
    While this DataFrame has only a single column (prediction), this object 
    facilitates writing results to disk via DataFrame.to_parquet(), etc.

    Examples
    --------
    # Aggregate predictions across cross-validation repeats for each chemical
    mean_predictions = predictions.groupby(level=0).mean()
    '''
    index_name = X.index.name if X.index.name else 'chemical'
    predictions_index = pd.MultiIndex.from_tuples(
        [(chemical, replicate) for chemical, replicate, _ in predictions_data],
        names=[index_name, 'replicate']
    )
    predictions_values = [pred for _, _, pred in predictions_data]
    return pd.DataFrame(
        predictions_values, 
        index=predictions_index, 
        columns=['prediction']
        )
#endregion