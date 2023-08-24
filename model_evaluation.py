'''
This module defines the ModelEvaluator class, responsible for evaluating 
machine learning models using repeated k-fold cross-validation. The class 
provides the flexibility to evaluate models both with and without feature 
selection, using a set of predefined metrics. It leverages parallel processing 
to efficiently handle large datasets and complex models.

Note
----
This class should be used in conjunction with pre-configured model settings, 
metrics manager, and an optional feature selector to fully utilize its 
capabilities.
'''

import pandas as pd
from sklearn.model_selection import RepeatedKFold
from joblib import Parallel, delayed

#region: ModelEvaluator.__init__
class ModelEvaluator:
    '''
    Class responsible for evaluating machine learning models using different 
    evaluation criteria, possibly with or without feature selection.

    Attributes
    ----------
    evaluation_settings : SimpleNamespace
        The configuration settings for the model evaluation.
    metrics_manager : object
        The manager that handles different evaluation metrics.
    feature_selector : FeatureSelector, optional
        An instance of a FeatureSelector to handle feature selection (default 
        is None).
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
            The estimator to evaluate.
        X : pandas.DataFrame
            The features data.
        y : pandas.Series
            The target data.
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
        Private method to evaluate the model with feature selection.

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
            Evaluation results including performances and 
            importances_replicates.
        '''
        estimator, performances, importances_replicates = (
            self._evaluate_with_repeated_kfold_and_selection(estimator, X, y)
        )

        # TODO: Create a Results class?
        evaluation_results = {
            'performances' : performances, 
            'importances_replicates' : importances_replicates
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
            The estimator to evaluate.
        X : pandas.DataFrame
            The features data.
        y : pandas.Series
            The target data.

        Returns
        -------
        estimator : object
            The fitted estimator with selected features.
        performances : pandas.DataFrame
            The performance results across different folds.
        importances_replicates : pandas.DataFrame
            The feature importances across different folds.
        '''
        # Initialize containers for the results.
        performances, importances_replicates = [], []

        # Initialize the outer cross-validation loop for model evaluation.
        rkf_cv = RepeatedKFold(
            n_splits=self.evaluation_settings.n_splits_cv, 
            n_repeats=self.evaluation_settings.n_repeats_cv, 
            random_state=self.evaluation_settings.random_state_cv
            )

        for train_ix, test_ix in rkf_cv.split(X):
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
        
        performances = pd.DataFrame(performances)
        performances.columns.names = ['metric']
        importances_replicates = pd.concat(importances_replicates)

        return estimator, performances, importances_replicates
    #endregion

    #region: _cross_validate_without_selection
    def _cross_validate_without_selection(self, estimator, X, y):
        '''
        Private method to evaluate the model without feature selection.

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
            Evaluation results including performances.
        '''
        estimator, performances = (
            self._evaluate_with_repeated_kfold(estimator, X, y)
        )

        # Fit the model to all data.
        estimator.fit(X, y)

        # TODO: Create a Results class?
        evaluation_results = {
            'performances' : performances
        }
        return evaluation_results
    #endregion

    #region: _evaluate_with_repeated_kfold
    def _evaluate_with_repeated_kfold(self, estimator, X, y):
        '''
        Execute a repeated k-fold cross-validation with all data.

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
        estimator : object
            The fitted estimator.
        performances : pandas.DataFrame
            The performance results across different folds.
        '''
        rkf = RepeatedKFold(
            n_splits=self.evaluation_settings.n_splits_cv, 
            n_repeats=self.evaluation_settings.n_repeats_cv, 
            random_state=self.evaluation_settings.random_state_cv
            )

        performances = Parallel(n_jobs=self._n_jobs)(
            delayed(self._split_fit_predict_and_score)(
                estimator, 
                X, 
                y, 
                train_ix, 
                test_ix, 
                ) for train_ix, test_ix in rkf.split(X)
            )
        performances = pd.DataFrame(performances)
        performances.columns.names = ['metric']

        return estimator, performances
    #endregion

    #region: _split_fit_predict_and_score
    def _split_fit_predict_and_score(
            self, estimator, X, y, train_ix, test_ix):
        '''
        Split the data, fit and evaluate the estimator for one fold.

        Enables parallelized cross-validation via joblib.

        Parameters
        ----------
        estimator : object
            The estimator to evaluate.
        X : pandas.DataFrame
            The features data.
        y : pandas.Series
            The target data.
        train_ix : array-like
            The indices for the training data.
        test_ix : array-like
            The indices for the test data.

        Returns
        -------
        object
            The score of the estimator on the test data.
        '''
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

        estimator.fit(X_train, y_train)

        y_pred = estimator.predict(X_test)
        return self.metrics_manager.score(y_test, y_pred)
    #endregion