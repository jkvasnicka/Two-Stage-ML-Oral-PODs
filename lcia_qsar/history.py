'''Contains the mix-in class, LciaQsarHistory.
'''

import pandas as pd
import os.path
import itertools

#region: LciaQsarHistory.__init__
class LciaQsarHistory:
    '''Mix-in class to navigate the history of workflow instructions.

    There are two general types of histories: 
        1. Fitted estimators
        2. Scores
    '''
    def __init__(self):
        # Initialize a container for the full histories.
        self._history_for = {}
#endregion

    #region: write_score_histories_to_csv
    def write_score_histories_to_csv(self, path_subdir=str()):
        '''For now, this method is used to monitor results via Git throughout
        refactoring.
        '''
        def are_dataframes(history):
            return all(isinstance(v, pd.DataFrame) for v in history) is True
        score_history_keys = [k for k, history in self._history_for.items() 
                              if are_dataframes(history)]

        if score_history_keys:
            if path_subdir:
                if not os.path.exists(path_subdir):
                    os.makedirs(path_subdir)

        for k in score_history_keys: 
            file_path = os.path.join(path_subdir, k+'.csv')
            history = self.concatenate_history(k)
            history.to_csv(file_path)
    #endregion

    #region: concatenate_history
    def concatenate_history(self, history_key):
        '''Concatenate the score histories across all models.
        '''
        # FIXME: For backwards compatability,
        model_keys = self.model_keys
        if 'importances' in history_key:
            model_keys = [k for k in model_keys if 'with_selection' in k]
        
        history_for_model = dict(
            zip(model_keys, self._history_for[history_key]))
        history_concat = pd.concat(history_for_model, axis=1)
        # Update/expand the column level names using the current history.
        previous_names = getattr(self, history_key).columns.names
        new_names = self.instruction_names + ['estimator']
        history_concat.columns.names = new_names + previous_names
        return history_concat
    #endregion

    #region: model_keys
    @property
    def model_keys(self):
        '''Return list of keys for all models in the history.
        '''
        if not hasattr(self, '_model_keys'):
            # Set the attribute (lazy evaluation).
            cartesian_product = itertools.product(
                self.config.model.instruction_keys, self.config.estimator_names)
            self._model_keys = [(*instruction_key, est_name) 
                    for instruction_key, est_name in cartesian_product]
            
        return self._model_keys
    #endregion

    #region: selection_model_keys
    @property
    def selection_model_keys(self):
        '''Return a subset of keys for models with feature selection'''
        return [k for k in self.model_keys if 'with_selection' in k]
    #endregion

    #region _get_current_history
    def _get_current_history(self, history_key):
        '''
        '''
        return self._history_for[history_key][-1]
    #endregion

    #region _set_current_history
    def _set_current_history(self, history_key, value):
        '''
        '''
        if history_key not in self._history_for:
            # Initialize the history (lazy evaluation).
            self._history_for[history_key] = []

        self._history_for[history_key].append(value)
    #endregion

    #region estimator
    @property
    def estimator(self):
        return self._get_current_history('estimator')

    @estimator.setter
    def estimator(self, current_estimator):
        self._set_current_history('estimator', current_estimator)
    #endregion

    #region: get_estimator
    def get_estimator(self, model_key):
        '''
        '''
        iloc = self.iloc_for_model_key[model_key]
        return self._history_for['estimator'][iloc]
    #endregion

    #region: iloc_for_model_key
    @property
    def iloc_for_model_key(self):
        return {i : k for k, i in enumerate(self.model_keys)}
    #endregion

    #region performances
    @property
    def performances(self):
        return self._get_current_history('performances')

    @performances.setter
    def performances(self, current_performances):
        self._set_current_history('performances', current_performances)
    #endregion

    #region importances
    @property
    def importances(self):
        return self._get_current_history('importances')

    @importances.setter
    def importances(self, current_importances):
        self._set_current_history('importances', current_importances)
    #endregion

    #region: get_importances
    def get_importances(self, model_key):
        '''
        '''
        iloc = self.iloc_for_selection_key[model_key]
        return self._history_for['importances'][iloc]
    #endregion

    #region: get_important_features
    def get_important_features(self, model_key, select_features, args=()):
        '''
        '''
        importances = self.get_importances(model_key)
        return select_features(importances, *args)
    #endregion

    #region importances_replicates
    @property
    def importances_replicates(self):
        return self._get_current_history('importances_replicates')

    @importances_replicates.setter
    def importances_replicates(self, current_importances_replicates):
        self._set_current_history(
            'importances_replicates', current_importances_replicates)
    #endregion

    #region: get_importances_replicates
    def get_importances_replicates(self, model_key):
        '''
        '''
        iloc = self.iloc_for_selection_key[model_key]
        return self._history_for['importances_replicates'][iloc]
    #endregion

    #region: get_important_features_replicates
    def get_important_features_replicates(
            self, model_key, stride, select_features, args=()):
        '''
        '''
        importances_replicates = LciaQsarHistory.split_replicates(
            self.get_importances_replicates(model_key), stride)
        
        return {i : select_features(importances, *args) 
                for i, importances in enumerate(importances_replicates)}
    #endregion

    #region: split_replicates
    @staticmethod
    def split_replicates(dataframe, stride):
        '''Split a replicates DataFrame into a list of DataFrame objects, one
        for each replicate.

        In general, the stride should be equal to the product of,
            'n_splits_select',
            'n_repeats_select', 
            'n_repeats_perm',
        and the len(DataFrame) should always a multiple of the stride.
        '''
        df_list = []
        length = len(dataframe)
        start = 0

        while start < length:
            end = start + stride
            subset = dataframe.iloc[start:end]
            df_list.append(subset)
            start = end
            
        return df_list
    #endregion

    #region: iloc_for_selection_key
    @property
    def iloc_for_selection_key(self):
        return {i : k for k, i in enumerate(self.selection_model_keys)}
    #endregion