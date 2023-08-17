'''Provide support for managing supervised learning workflows.

NOTE: Setting different values for the 'n_jobs' parameter can lead to 
different results. When n_jobs=-1, the state of the random number 
generator (e.g., of an estimator) may be distributed differently 
across folds/processes compared to the sequential case (n_jobs=None).
https://joblib.readthedocs.io/en/latest/auto_examples/parallel_random_state.html
'''

import pandas as pd
import numpy as np 
from sklearn.model_selection import RepeatedKFold
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed

import evaluation

# TODO: Change name, ModelEvaluator? Use a config file for the arguments.
#region: SupervisedLearningWorkflow.__init__
class SupervisedLearningWorkflow:
    '''The base class. At minimum, it needs X & y.

    'estimator' can be a Pipeline. Setting estimator can be delayed, e.g., 
    until self._repeated_kfolds().
    '''
    def __init__(self, X, y, estimator=None):
        self.X = X
        self.y = y
        self.estimator = estimator
#endregion

    #region: build_model_with_selection
    def build_model_with_selection(
            self, scoring,  function_for_metric, criterion_metric=None, n_features=10,
            n_splits_cv=5, n_repeats_cv=50, random_state_cv=None,
            n_splits_select=5, n_repeats_select=50, random_state_select=None, 
            n_repeats_perm=5, random_state_perm=None, n_jobs=None, **kwargs):
        '''Build a model, with feature selection, for out-of-sample 
        prediction.

        Evaluate its generalization error via a nested cross validation.
        '''
        ## Estimate the model's generalization error via "pseudo models."
        performances, importances_replicates = (
             self._repeated_kfold_with_nested_selection(
                scoring,  
                function_for_metric, 
                criterion_metric=criterion_metric,
                n_features=n_features,
                n_splits_cv=n_splits_cv, 
                n_repeats_cv=n_repeats_cv, 
                random_state_cv=random_state_cv,
                n_splits_select=n_splits_select, 
                n_repeats_select=n_repeats_select, 
                random_state_select=random_state_select, 
                n_repeats_perm=n_repeats_perm, 
                random_state_perm=random_state_perm, 
                n_jobs=n_jobs
             )
        )

        ## Build the model for out-of-sample prediction.
        X, y = self.X, self.y  # all data
        important_features, importances = (
            self._nested_feature_selection(
                X, 
                y,
                scoring, 
                criterion_metric=criterion_metric,
                n_features=n_features, 
                n_splits_select=n_splits_select, 
                n_repeats_select=n_repeats_select, 
                random_state_select=random_state_select, 
                n_repeats_perm=n_repeats_perm, 
                random_state_perm=random_state_perm,
                n_jobs=n_jobs
                )
        )
        self.estimator.fit(X[important_features], y)
        
        return {
            'performances' : performances, 
            'importances_replicates' : importances_replicates, 
            'importances' : importances
            }
    #endregion

    #region: _repeated_kfold_with_nested_selection
    def _repeated_kfold_with_nested_selection(
            self, scoring,  function_for_metric, criterion_metric=None, n_features=10,
            n_splits_cv=5, n_repeats_cv=50, random_state_cv=None,
            n_splits_select=5, n_repeats_select=50, random_state_select=None, 
            n_repeats_perm=5, random_state_perm=None, n_jobs=None):
        '''
        '''
        # Start with all data.
        X, y = self.X, self.y

        # Initialize containers for the results.
        performances, importances_replicates = [], []

        # Initialize the outer cross-validation loop for model evaluation.
        rkf_cv = RepeatedKFold(
            n_splits=n_splits_cv, n_repeats=n_repeats_cv, 
            random_state=random_state_cv)

        for train_ix, test_ix in rkf_cv.split(X):
            X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
            y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

            important_features, importances = (
                self._nested_feature_selection(
                    X_train, 
                    y_train, 
                    scoring, 
                    criterion_metric=criterion_metric, 
                    n_features=n_features, 
                    n_splits_select=n_splits_select, 
                    n_repeats_select=n_repeats_select, 
                    random_state_select=random_state_select, 
                    n_repeats_perm=n_repeats_perm, 
                    random_state_perm=random_state_perm,
                    n_jobs=n_jobs
                    )
            )
            importances_replicates.append(importances)

            self.estimator.fit(X_train[important_features], y_train)

            y_pred = self.estimator.predict(X_test[important_features])
            performances.append(evaluation.score(y_test, y_pred, function_for_metric)) 
        
        performances = pd.DataFrame(performances)
        performances.columns.names = ['metric']
        importances_replicates = pd.concat(importances_replicates)

        return performances, importances_replicates
    #endregion

    #region: _nested_feature_selection
    def _nested_feature_selection(
            self, X_train, y_train, scoring, criterion_metric=None, n_features=10, 
            n_splits_select=5, n_repeats_select=50, random_state_select=None, 
            n_repeats_perm=5, random_state_perm=None, n_jobs=None):
        '''
        '''
        importances = self._repeated_kfold_permutation_importance(
            X_train, 
            y_train,
            scoring, 
            n_splits_select=n_splits_select, 
            n_repeats_select=n_repeats_select, 
            random_state_select=random_state_select, 
            n_repeats_perm=n_repeats_perm, 
            random_state_perm=random_state_perm,
            n_jobs=n_jobs
        )        
        important_features = SupervisedLearningWorkflow.select_features(
            importances, criterion_metric, n_features)

        return important_features, importances
    #endregion

    #region: _repeated_kfold_permutation_importance
    def _repeated_kfold_permutation_importance(
            self, X_train, y_train, scoring, n_splits_select=5, n_repeats_select=50, 
            random_state_select=None, n_repeats_perm=5, random_state_perm=None,
            n_jobs=None):
        '''Compute permutation importances for feature evaluation. 

        Wrap the function, sklearn.inspection.permutation_importance(), within a 
        repeated k-fold cross-validation scheme.

        Parameters
        ----------
        scoring : list of str
            Scorers to use. Can be a single scorer but must be contained in a list.
            Each scorer must be in sklearn.metrics.get_scorer_names().
        For the remaining parameters, see learn._repeated_kfold(). Parameters with 
        suffix, 'cv', correspond to the cross-validation splitter, whereas those
        with suffix, 'perm', corrspond to permutation_importance().
        
        Returns
        -------
        dict of pandas.DataFrame
            Axis 0 = execution number, Axis 1 = feature.
        '''
        # Initialize the inner cross-validation loop for feature selection.
        rkf_inner = RepeatedKFold(
            n_splits=n_splits_select, 
            n_repeats=n_repeats_select, 
            random_state=random_state_select
            )

        dicts_of_bunch_objs = Parallel(n_jobs=n_jobs)(
            delayed(self._permutation_importance)(
                X_train, 
                y_train, 
                train_ix, 
                test_ix, 
                scoring=scoring, 
                n_repeats_perm=n_repeats_perm, 
                random_state_perm=random_state_perm
                ) for train_ix, test_ix in rkf_inner.split(X_train)
            )
        
        # Initialize a container for the final results.
        importances_for_metric = {}
        # Unpack the raw importance scores from the Bunch objects.
        for metric in scoring:
            importances = np.concatenate(
                [d[metric].importances for d in dicts_of_bunch_objs], 
                axis=1).T
            importances_for_metric[metric] = pd.DataFrame(
                importances, columns=list(X_train))
        importances_for_metric = pd.concat(importances_for_metric, axis=1)
        importances_for_metric.columns.names = ['metric', 'feature']

        return importances_for_metric
    #endregion

    #region: _permutation_importance
    def _permutation_importance(
            self, X_train, y_train, train_ix, test_ix, scoring=None, 
            n_repeats_perm=5, random_state_perm=None):
        '''Execute the permutation_importance algorithm for one fold.

        Enables parallelized cross validation via joblib.
        '''
        X_train_inner = X_train.iloc[train_ix, :]
        X_test_inner = X_train.iloc[test_ix, :]
        y_train_inner = y_train.iloc[train_ix]
        y_test_inner = y_train.iloc[test_ix]

        self.estimator.fit(X_train_inner, y_train_inner)

        return permutation_importance(
            self.estimator, 
            X_test_inner, 
            y_test_inner, 
            scoring=scoring, 
            n_repeats=n_repeats_perm, 
            n_jobs=1,  # to avoid "oversubscription" of CPU resources
            random_state=random_state_perm
            )
    #endregion

    #region: build_model_without_selection
    def build_model_without_selection(
            self, function_for_metric, n_splits_cv=5, n_repeats_cv=50, 
            random_state_cv=None, n_jobs=None, **kwargs):
        '''Build a model, without feature selection, for out-of-sample 
        prediction.

        Evaluate its generalization error via a cross validation.
        '''
        performances = self._repeated_kfold_all_data(
            function_for_metric, 
            n_splits_cv=n_splits_cv, 
            n_repeats_cv=n_repeats_cv, 
            random_state_cv=random_state_cv, 
            n_jobs=n_jobs
        )

        # Fit the model to all data.
        self.estimator.fit(self.X, self.y)

        return performances
    #endregion

    #region: _repeated_kfold_all_data
    def _repeated_kfold_all_data(
            self, function_for_metric, n_splits_cv=5, n_repeats_cv=50, 
            random_state_cv=None, n_jobs=None):
        '''Execute a repeated k-fold cross validation with all data. 
        '''
        X, y = self.X, self.y  # all data

        rkf = RepeatedKFold(
            n_splits=n_splits_cv, n_repeats=n_repeats_cv, random_state=random_state_cv)

        performances = Parallel(n_jobs=n_jobs)(
            delayed(self._split_fit_predict_and_score)(
                X, 
                y, 
                train_ix, 
                test_ix, 
                function_for_metric
                ) for train_ix, test_ix in rkf.split(X)
            )
        performances = pd.DataFrame(performances)
        performances.columns.names = ['metric']

        return performances
    #endregion

    #region: _split_fit_predict_and_score
    def _split_fit_predict_and_score(
                self, X, y, train_ix, test_ix, function_for_metric):
            '''Split the data, fit and evaluate the estimator for one fold.

            Enables parallelized cross validation via joblib.
            '''
            X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
            y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

            self.estimator.fit(X_train, y_train)

            y_pred = self.estimator.predict(X_test)
            return evaluation.score(y_test, y_pred, function_for_metric)
    #endregion

    # TODO: May be obsolete?
    #region: X_fitted
    @property
    def X_fitted(self):
        '''Return the features (X) seen during fit().

        Works only when X has feature names that are all strings.
        '''
        return self.X[self.estimator.feature_names_in_]
    #endregion

    #region: select_features
    @staticmethod
    def select_features(importances, criterion_metric, n_features):
        '''Get important features based on a single criterion and metric.

        Returns
        -------
        list of str
            Names of important features corresponding to the columns.
        '''
        metric_importances = importances[criterion_metric]
        features_greatest_to_least = list(
            metric_importances.quantile()
            .sort_values(ascending=False)
            .index
        )
        return list(features_greatest_to_least)[:n_features]
    #endregion