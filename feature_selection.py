'''
This module defines the `FeatureSelector` class, a utility for selecting 
important features in a dataset based on permutation importances. It includes 
methods to compute feature importances through repeated k-fold 
cross-validation and to select features according to specific criteria and 
metrics.

Example
-------
feature_selector = FeatureSelector(model_settings)
estimator, important_features, importances = (
    feature_selector.nested_feature_selection(estimator, X_train, y_train)
    )
'''

import pandas as pd
import numpy as np 
from sklearn.model_selection import RepeatedKFold
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed

#region: FeatureSelector.__init__
class FeatureSelector:
    '''
    A class to select important features using permutation importances and 
    repeated k-fold cross-validation.

    Attributes
    ----------
    model_settings : SimpleNamespace
        A configuration object containing settings related to the model, such 
        as cross-validation splits, repeated folds, scoring metrics, etc.

    Methods
    -------
    nested_feature_selection(estimator, X_train, y_train)
        Perform feature selection using nested cross-validation.
    permutation_importances(estimator, X_train, y_train)
        Compute permutation importances for feature evaluation using repeated 
        k-fold cross-validation.
    permutation_importance_wrapper(
        estimator, X_train, y_train, train_ix, test_ix
        )
        Execute the permutation_importance algorithm for one fold.
    select_features(importances, criterion_metric, n_features)
        Get important features based on a single criterion and metric.
    '''
    def __init__(self, model_settings):
        '''
        Initialize the FeatureSelector class with model settings.

        Parameters
        ----------
        model_settings : SimpleNamespace
            A configuration object containing settings related to the model.
        '''
        self.model_settings = model_settings
#endregion

    #region: nested_feature_selection
    def nested_feature_selection(self, estimator, X_train, y_train):
        '''
        Perform feature selection using nested cross-validation.

        Parameters
        ----------
        estimator : object
            A scikit-learn estimator object with fit and predict methods.
        X_train : pandas.DataFrame
            Training feature data.
        y_train : pandas.Series
            Training target data.

        Returns
        -------
        estimator : object
            Fitted estimator after feature selection.
        important_features : list of str
            List of selected important feature names.
        importances : pandas.DataFrame
            Dataframe containing feature importances.
        '''
        estimator, importances = self.permutation_importances(estimator, X_train, y_train)        
        important_features = FeatureSelector.select_features(
            importances, 
            self.model_settings.criterion_metric, 
            self.model_settings.n_features
            )

        return estimator, important_features, importances
    #endregion

    #region: permutation_importances
    def permutation_importances(self, estimator, X_train, y_train):
        '''
        Compute permutation importances for feature evaluation. 
        
        Wraps the function sklearn.inspection.permutation_importance() within 
        a repeated k-fold cross-validation scheme.

        Parameters
        ----------
        estimator : object
            A scikit-learn estimator object with fit and predict methods.
        X_train : pandas.DataFrame
            Training feature data.
        y_train : pandas.Series
            Training target data.

        Returns
        -------
        estimator : object
            Fitted estimator after permutation importances.
        importances_for_metric : pandas.DataFrame
            Dataframe containing feature importances, with metrics as columns 
            and features as rows.
        '''
        # Initialize the inner cross-validation loop for feature selection.
        rkf_inner = RepeatedKFold(
            n_splits=self.model_settings.n_splits_select, 
            n_repeats=self.model_settings.n_repeats_select, 
            random_state=self.model_settings.random_state_select
            )

        dicts_of_bunch_objs = Parallel(n_jobs=self.model_settings.n_jobs)(
            delayed(self.permutation_importance_wrapper)(
                estimator,
                X_train, 
                y_train, 
                train_ix, 
                test_ix
                ) for train_ix, test_ix in rkf_inner.split(X_train)
            )
        
        # Initialize a container for the final results.
        importances_for_metric = {}
        # Unpack the raw importance scores from the Bunch objects.
        for metric in self.model_settings.feature_importance_scorings:
            importances = np.concatenate(
                [d[metric].importances for d in dicts_of_bunch_objs], 
                axis=1).T
            importances_for_metric[metric] = pd.DataFrame(
                importances, columns=list(X_train))
        importances_for_metric = pd.concat(importances_for_metric, axis=1)
        importances_for_metric.columns.names = ['metric', 'feature']

        return estimator, importances_for_metric
    #endregion

    #region: permutation_importance_wrapper
    def permutation_importance_wrapper(
            self, estimator, X_train, y_train, train_ix, test_ix):
        '''
        Execute the permutation_importance algorithm for one fold.

        Parameters
        ----------
        estimator : object
            A scikit-learn estimator object with fit and predict methods.
        X_train : pandas.DataFrame
            Training feature data.
        y_train : pandas.Series
            Training target data.
        train_ix : array-like
            Indices for the training set in the current fold.
        test_ix : array-like
            Indices for the test set in the current fold.

        Returns
        -------
        result : object
            A Bunch object containing the permutation importances.
        '''
        X_train_inner = X_train.iloc[train_ix, :]
        X_test_inner = X_train.iloc[test_ix, :]
        y_train_inner = y_train.iloc[train_ix]
        y_test_inner = y_train.iloc[test_ix]

        estimator.fit(X_train_inner, y_train_inner)

        return permutation_importance(
            estimator, 
            X_test_inner, 
            y_test_inner, 
            scoring=self.model_settings.feature_importance_scorings, 
            n_repeats=self.model_settings.n_repeats_perm, 
            n_jobs=1,  # to avoid "oversubscription" of CPU resources
            random_state=self.model_settings.random_state_perm
            )
    #endregion

    #region: select_features
    @staticmethod
    def select_features(importances, criterion_metric, n_features):
        '''
        Get important features based on a single criterion and metric.

        Parameters
        ----------
        importances : pandas.DataFrame
            Dataframe containing feature importances, with metrics as columns 
            and features as rows.
        criterion_metric : str
            A metric name to be used as a criterion for feature selection.
        n_features : int
            Number of features to be selected.

        Returns
        -------
        list of str
            Names of important features corresponding to the selected columns.
        '''
        metric_importances = importances[criterion_metric]
        features_greatest_to_least = list(
            metric_importances.quantile()
            .sort_values(ascending=False)
            .index
        )
        return list(features_greatest_to_least)[:n_features]
    #endregion