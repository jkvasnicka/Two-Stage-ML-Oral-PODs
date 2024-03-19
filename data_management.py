'''
This module contains the `DataManager` class which manages the loading and 
handling of features and target data.
'''

import pandas as pd 

#region: DataManager.__init__
class DataManager:
    '''
    Manages the loading and handling of features and target data.
    '''
    def __init__(self, data_settings, path_settings):
        '''
        Initialize the DataManager with configuration settings.

        Parameters
        ----------
        data_settings : SimpleNamespace
            Data configuration settings. 
        path_settings : SimpleNamespace
            Path configuration settings.
        '''
        # TODO: Should be private attributes? Check other classes too.
        self.data_settings = data_settings
        self.path_settings = path_settings
#endregion

    #region: load_features_and_target
    def load_features_and_target(
            self, 
            *, 
            target_effect, 
            features_source, 
            ld50_type, 
            data_condition, 
            **kwargs
            ):
        '''
        Load both features (X) and target (y) based on the provided 
        parameters. Ensure that X and y share a common index.

        Parameters
        ----------
        target_effect : str
            The target effect to be considered.
        features_source : str
            The source of the features data.
        ld50_type : str
            Type of LD50 data to be used.
        data_condition : str
            Condition for handling data (e.g., dropping missing values).

        **kwargs
            Collects any unneeded key-value pairs from model_keys.

        Returns
        -------
        tuple
            A tuple containing the loaded features (X) and target (y) as 
            pandas objects with a common index.
        '''
        X = self.load_features(
            features_source=features_source, 
            ld50_type=ld50_type, 
            data_condition=data_condition
            )

        y = self.load_target(target_effect=target_effect)

        # Use the intersection of chemicals.
        return DataManager.with_common_index(X, y)
    #endregion

    #region: load_features
    def load_features(
            self, 
            *, 
            features_source, 
            ld50_type, 
            data_condition, 
            exclude_training=False,
            target_effect=None,
            **kwargs
            ):
        '''
        Load the features (X) based on the provided parameters and 
        configuration.

        Parameters
        ----------
        features_source : str
            The source of the features data.
        ld50_type : str
            Type of LD50 data to be used.
        data_condition : str
            Condition for handling data (e.g., dropping missing values).
        exclude_training : bool, optional
            If True, 'target_effect' must also be passed and chemicals used 
            for model training will be excluded. Default is False; features 
            are loaded for all chemicals.
        target_effect : str, optional
            The target effect to be considered. Needed if 'exclude_training'.
        **kwargs
            Collects any unneeded key-value pairs.

        Returns
        -------
        pandas.DataFrame
            The loaded features (X) as a DataFrame.
        '''
        features_path = (
            self.path_settings.file_for_features_source[features_source]
        )
        X = pd.read_parquet(features_path)

        if exclude_training:
            y = self.load_target(target_effect=target_effect)
            training_chemicals = set(y.index.intersection(X.index))
            X = X.drop(training_chemicals)

        if self.data_settings.use_experimental_for_ld50[ld50_type]:
            ld50s_experimental = (
                pd.read_csv(
                    self.path_settings.ld50_experimental_file, 
                    index_col=0)
                    .squeeze()
                    )
            X = DataManager._swap_column(
                X, 
                self.data_settings.ld50_pred_column_for_source[features_source], 
                ld50s_experimental
                )
            
        if self.data_settings.drop_missing_for_condition[data_condition]:
            # Use only samples with complete data.
            X = X.dropna(how='any')
        
        return X
    #endregion

    #region: load_target
    def load_target(self, *, target_effect, **kwargs):
        '''
        Load the target variable (y) for the specified target effect.

        Parameters
        ----------
        target_effect : str
            The target effect to be considered.

        **kwargs
            Collects any unneeded key-value pairs.

        Returns
        -------
        pandas.Series
            The loaded target variable (y) as a Series.
        '''
        ys = pd.read_csv(self.path_settings.surrogate_pods_file, index_col=0)
        return ys[target_effect].squeeze().dropna()
    #endregion

    #region: _swap_column
    @staticmethod
    def _swap_column(X, column_old, column_new):
        '''
        Swap an existing column in the DataFrame with a new column.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to be swapped.
        column_old : str
            Name of the existing column to be replaced.
        column_new : pandas.Series
            The new column to replace the existing one.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with the swapped column.
        '''
        X = X.drop(column_old, axis=1)
        return X.merge(
            column_new, 
            left_index=True, 
            right_index=True, 
            how='left'
            )
    #endregion

    #region: with_common_index
    @staticmethod
    def with_common_index(*pandas_objects):
        '''
        Align the provided pandas objects based on a common index.

        Parameters
        ----------
        *pandas_objects : pandas.DataFrame or pandas.Series
            One or more pandas objects to be aligned by index.

        Returns
        -------
        list
            List of pandas objects (DataFrame or Series) aligned by a common 
            index.
        '''
        original_obj_for_key = {k : v for k, v in enumerate(pandas_objects)}
        multiindex_frame = pd.concat(
            original_obj_for_key, join='inner', axis=1
            )

        # Initialize a container.
        common_objects = []
        for k, original_obj in original_obj_for_key.items():
            new_obj = multiindex_frame[k]
            if isinstance(original_obj, pd.Series):
                new_obj = new_obj.squeeze()
            common_objects.append(new_obj)
        return common_objects
    #endregion

    #region: load_authoritative_pods
    def load_authoritative_pods(self):
        '''
        Load the authoritative points of departure (log10-units). 

        Returns
        -------
        pandas.DataFrame
            One column for each target effect, and chemicals along the index.
        '''
        pods_for_effect = pd.read_csv(
            self.path_settings.authoritative_pods_file, 
            index_col=0
            )
        return pods_for_effect
    #endregion

    #region: load_oral_equivalent_doses
    def load_oral_equivalent_doses(self):
        '''
        Load the oral equivalent doses from a CSV file.

        Returns
        -------
        pandas.Series
        '''
        oeds = (
            pd.read_csv(
                self.path_settings.toxcast_oeds_file,
                index_col=0
            )
            .squeeze()
        )
        return oeds
    #endregion

    #region: load_exposure_data
    def load_exposure_data(self):
        '''
        Load the SEEEM3 exposure data from a Parquet file.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing exposure predictions.
        '''
        return pd.read_parquet(self.path_settings.seem3_exposure_file)
    #endregion

    # TODO: This is only used to get DTXSIDs for OPERA, etc. Move upstream.
    #region: load_application_chemicals
    def load_application_chemicals(self):
        '''
        Get chemical identifiers (DTXSID) for model application.

        These identifiers are derived from the intersection of the 
        "Merged NORMAN Suspect List: SusDat" and SEEM3 chemicals, excluding
        any chemicals used for model training.

        Returns
        -------
        list of str

        References
        ----------
        https://doi.org/10.5281/zenodo.6853705
        '''
        norman_chemicals = set(
            pd.read_csv(self.path_settings.norman_chemicals_file)
            .squeeze()
            )
        exposure_chemicals = set(self.load_exposure_data().index)
        training_chemicals = set(self.load_training_chemicals())

        return list(
            norman_chemicals
            .intersection(exposure_chemicals)
            .difference(training_chemicals)
        )
    #endregion
    
    # TODO: This is only used to get DTXSIDs for OPERA, etc. Move upstream.
    #region: load_training_chemicals
    def load_training_chemicals(self):
        '''
        Helper function to get the identifiers for chemicals used for model 
        training.

        These chemicals should be excluded from "out-of-sample" predictions.
        For all other purposes, training data should be loaded via 
        `DataManager.load_features_and_target()`.
        '''        
        return list(
            pd.read_csv(
                self.path_settings.surrogate_pods_file, 
                index_col=0
            )
            .index
        )
    #endregion

    # TODO: This is only used to get DTXSIDs for OPERA, etc. Move upstream.
    #region: load_all_chemicals
    def load_all_chemicals(self):
        '''
        Get all chemical identifiers (DTXSID). 

        "All chemicals" in this context is the union of the training set 
        and out-of-sample chemicals.

        Returns
        -------
        list
        '''
        training_chemicals = set(self.load_training_chemicals())
        app_chemicals = set(self.load_application_chemicals())

        return list(app_chemicals.union(training_chemicals))
    #endregion