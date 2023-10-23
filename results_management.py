'''
The `results_management` module provides the `ResultsManager` class, 
responsible for managing the writing, reading, and handling of results 
and fitted estimators. It includes methods for saving and retrieving results 
in Parquet format, handling fitted estimator objects with Joblib, managing 
metadata such as headers, and providing utility functions for listing and 
accessing model keys.

Classes:
    ResultsManager: Manages the reading and writing of results and fitted 
    estimators.
'''

import os
import pandas as pd
import json
import joblib
import itertools

#region: ResultsManager
class ResultsManager:
    '''
    Manage the reading and writing of results and fitted estimators.

    The `ResultsManager` class provides methods to handle the storage and 
    retrieval of various types of results including model performances, 
    importances, fitted estimators, and metadata such as headers and model key 
    names. It supports reading and writing files for results and estimators.

    Parameters
    ----------
    output_dir : str
        Path to the directory where results and estimators will be saved.

    Attributes
    ----------
    output_dir : str
        Path to the directory where results and estimators are saved.
    _metadata_path : str
        Path to the metadata JSON file within the output directory.

    Methods
    -------
    write_estimator(estimator, model_key)
        Write the fitted estimator to a Joblib file.
    read_estimator(model_key)
        Read the fitted estimator from a Joblib file.
    write_result(result_df, model_key, result_type)
        Write the result DataFrame to the specified file.
    read_result(model_key, result_type)
        Read the result file and returns a DataFrame.
    ... (other methods)
    '''
    def __init__(self, output_dir, results_file_type='csv', model_key_creator=None):
        '''
        Initialize the ResultsManager with the specified output directory.

        This manager handles the storage and retrieval of model results and 
        metadata. If a ModelKeyCreator is provided, it will be used to 
        facilitate model key creation during the initial writing of results.

        Parameters
        ----------
        output_dir : str
            Path to the directory where results and estimators will be saved.
        results_file_type : str, optional
            Must be 'csv' or 'parquet'. Default 'csv'.
        model_key_creator : ModelKeyCreator, optional
            An instance of ModelKeyCreator to assist in creating model keys. 
            Primarily used during the initial writing stage of results. 
            If not provided, model key creation is assumed to have been 
            handled externally.
        '''
        self.output_dir = output_dir

        if results_file_type not in ('csv', 'parquet'):
            raise ValueError(
                "'results_file_type' must be either 'csv' or 'parquet'")
        self._results_file_type = results_file_type
        
        if model_key_creator:

            model_key_names = model_key_creator.create_model_key_names()
            self.write_model_key_names(model_key_names)

            mapping = model_key_creator.create_identifier_key_mapping()
            self.write_identifier_key_mapping(mapping)
#endregion

    #region: write_results
    def write_results(self, model_key, results):
        '''
        Write the results to the appropriate files.

        Parameters
        ----------
        model_key : tuple of str
            Model key identifying the result.
        results : dict 
            Contains the results to be written.
        '''
        for result_type, result_data in results.items():
            if isinstance(result_data, pd.DataFrame):
                self.write_result(result_data, model_key, result_type)
            elif hasattr(result_data, 'fit'):
                self.write_estimator(result_data, model_key)
    #endregion

    #region: write_estimator
    def write_estimator(self, estimator, model_key):
        '''
        Write the fitted estimator to a Joblib file.

        Parameters
        ----------
        estimator : object
            Fitted estimator object to be saved.
        model_key : tuple of str
            Model key identifying the estimator.

        Notes
        -----
        The Joblib file is saved in a subdirectory named after the model_key,
        within the output directory.
        '''
        path = self._build_estimator_path(model_key)
        joblib.dump(estimator, path)
    #endregion

    #region: read_estimator
    def read_estimator(self, model_key):
        '''
        Read the fitted estimator from a Joblib file.

        Parameters
        ----------
        model_key : tuple of str
            Model key identifying the estimator.

        Returns
        -------
        object
            Fitted estimator object.

        Notes
        -----
        The Joblib file is read from a subdirectory named after the model_key,
        within the output directory.
        '''
        path = self._build_estimator_path(model_key)
        return joblib.load(path)
    #endregion

    #region: _build_estimator_path
    def _build_estimator_path(self, model_key):
        '''
        Build the full path for the specified estimator file.

        Parameters
        ----------
        model_key : tuple of str
            Model key identifying the estimator.

        Returns
        -------
        str
            Full path to the estimator file.

        Notes
        -----
        This method builds the path for the estimator using the "joblib" 
        extension and ensures that the corresponding directory exists.
        '''
        return self._build_path(model_key, 'estimator', 'joblib')
    #endregion
 
    #region: write_result
    def write_result(self, result_df, model_key, result_type):
        '''
        Write a result DataFrame to the specified file type.

        Parameters
        ----------
        result_df : pd.DataFrame
            DataFrame containing the results to be saved.
        model_key : tuple of str
            Model key identifying the result.
        result_type : str
            Type of result (e.g., "performances", "importances").

        Notes
        -----
        The result file is saved in a subdirectory named after the model_key,
        within the output directory.
        '''
        # Create directory structure
        path = self._build_path(
            model_key, result_type, self._results_file_type)

        # Write DataFrame to disk
        if self._results_file_type == 'csv':
            result_df.to_csv(path)
            self.write_level_names(result_df.columns.names, result_type)
        elif self._results_file_type == 'parquet':
            result_df.to_parquet(path)
    #endregion

    #region: read_result
    def read_result(self, model_key, result_type):
        '''
        Read the result file and return a DataFrame.

        Parameters
        ----------
        model_key : tuple of str
            Model key identifying the result.
        result_type : str
            Type of result (e.g., "performances", "importances").

        Returns
        -------
        pd.DataFrame
            DataFrame containing the results.

        Notes
        -----
        The result file is read from a subdirectory named after the model_key,
        within the output directory.
        '''
        path = self._build_path(
            model_key, result_type, self._results_file_type)

        if self._results_file_type == 'csv':
            level_names = self.read_level_names()[result_type]
            header_indices = list(range(len(level_names)))
            result_df = pd.read_csv(path, header=header_indices, index_col=0)
            result_df.columns.names = level_names
        elif self._results_file_type == 'parquet':
            result_df = pd.read_parquet(path)

        return result_df
    #endregion

    #region: combine_results
    def combine_results(self, result_type, model_keys=None):
        '''
        Combine results from multiple model keys into a single DataFrame with 
        MultiIndex columns.

        Parameters
        ----------
        result_type : str
            The type of result to retrieve, e.g., 'performances', 
            'importances', etc. Must match the name used when saving the 
            results.
        model_keys : list of tuple, optional
            List of model keys for which to retrieve the results. If None, 
            then all model keys will be used.

        Returns
        -------
        combined_df : pandas.DataFrame
            Combined DataFrame with MultiIndex columns. The first level of the 
            MultiIndex contains the model keys, and the remaining levels are 
            taken from the original columns of the individual result DataFrames.
        '''
        if not model_keys:
            model_keys = self.read_model_keys()  # all of them

        combined_data = {
            key: self.read_result(key, result_type) 
            for key in model_keys
            }
        combined_df = pd.concat(combined_data, axis=1)

        model_key_names = self.read_model_key_names()
        first_df = next(iter(combined_data.values()))
        combined_df.columns.names = (
            model_key_names 
            + list(first_df.columns.names)
        )

        return combined_df
    #endregion

    #region: _build_path
    def _build_path(self, model_key, result_type, file_type):
        '''
        Build the full path for the specified file.

        Parameters
        ----------
        model_key : tuple of str
            Model key identifying the result.
        result_type : str
            Type of result (e.g., "performances", "importances").
        file_type : str
            File extension (e.g., "csv", "parquet", "joblib").

        Returns
        -------
        str
            Full path to the file.

        Notes
        -----
        This method builds the path based on the given parameters and ensures
        that the corresponding directory exists.
        '''
        directory_name = self.model_key_to_identifier(model_key)
        directory = os.path.join(self.output_dir, directory_name)
        self._ensure_directory(directory)
        
        return os.path.join(directory, f'{result_type}.{file_type}')
    #endregion
    
    #region: identifier_to_model_key
    def identifier_to_model_key(self, identifier):
        '''
        Retrieve the model key corresponding to a given identifier.

        Parameters
        ----------
        identifier : str
            The simple identifier used to represent a model key.

        Returns
        -------
        tuple
            The model key corresponding to the provided identifier. 
            If the identifier is not found, returns None.
        '''
        mapping = self.read_identifier_key_mapping()
        return mapping.get(identifier)
    #endregion

    #region: model_key_to_identifier
    def model_key_to_identifier(self, model_key):
        '''
        Retrieve the identifier corresponding to a given model key.

        Parameters
        ----------
        model_key : tuple
            The model key for which to find the corresponding identifier.

        Returns
        -------
        str
            The identifier representing the provided model key. 
            Raises a KeyError if the model key is not found.
        '''
        mapping = self.read_identifier_key_mapping()
        inverse_mapping = {v: k for k, v in mapping.items()}
        return inverse_mapping[model_key]
    #endregion

    #region: _ensure_directory
    def _ensure_directory(self, path):
        '''
        Ensure that the specified directory exists.

        Parameters
        ----------
        path : str
            Path to the directory.

        Notes
        -----
        If the directory does not exist, it is created.
        '''
        if not os.path.exists(path):
            os.makedirs(path)
    #endregion

    #region: write_level_names
    def write_level_names(self, level_names, result_type):
        '''
        Write the level names to the metadata file.

        Parameters
        ----------
        level_names : list of str or pd.core.indexes.frozen.FrozenList
            List of level names for the result DataFrame.
        result_type : str
            Type of result (e.g., "performances", "importances").

        Notes
        -----
        The level names are stored as metadata within the output directory.
        If the input is of type FrozenList, it is converted to a list.
        '''
        if isinstance(level_names, pd.core.indexes.frozen.FrozenList):
            level_names = list(level_names)  # ensures compatibility with JSON

        all_metadata = self.read_all_metadata()
        k = 'level_names_for_result'
        if k not in all_metadata:
            all_metadata[k] = {}  # initialize
        
        all_metadata[k][result_type] = level_names 
        self.write_all_metadata(all_metadata)
    #endregion

    #region: read_level_names
    def read_level_names(self):
        '''
        Read the level names from the metadata file.

        Returns
        -------
        dict
            Dictionary containing the level names for different result types.

        Notes
        -----
        The level names are read from the metadata stored within the output 
        directory.
        '''
        all_metadata = self.read_all_metadata()
        return all_metadata['level_names_for_result']
    #endregion

    #region: get_level_indices
    def get_level_indices(level_names, multi_index):
        '''
        Retrieve the indices of the given level names in a MultiIndex.

        Parameters
        ----------
        level_names : list of str
            Names of the levels for which indices are to be found.
        multi_index : pd.MultiIndex
            MultiIndex object containing the levels.

        Returns
        -------
        list of int
            Indices of the given level names within the MultiIndex.
        '''
        return [multi_index.names.index(name) for name in level_names]
    #endregion

    #region: read_model_keys
    def read_model_keys(self, inclusion_string=None, exclusion_string=None):
        '''
        List all model keys present in the metadata file.
        
        The keys are optionally filtered by inclusion or exclusion criteria.

        Parameters
        ----------
        inclusion_string : str, optional
            String to use for including model keys. If a model key contains this 
            string, it will be included in the final output. If None, no inclusion 
            filtering is applied.
        exclusion_string : str, optional
            String to use for excluding model keys. If a model key contains this 
            string, it will be excluded from the final output. If None, no exclusion 
            filtering is applied.

        Returns
        -------
        list of tuple
            List of model keys, where each model key is represented as a tuple of 
            strings.
        '''
        model_keys = [
            tuple(v) for v in self.read_identifier_key_mapping().values()
            ]

        if inclusion_string:
            model_keys = [k for k in model_keys if inclusion_string in k]
        if exclusion_string:
            model_keys = [k for k in model_keys if exclusion_string not in k]

        return model_keys   
    #endregion

    #region: write_model_key_names
    def write_model_key_names(self, model_key_names):
        '''
        Write the model key names to the metadata file.

        Parameters
        ----------
        model_key_names : list of str
            Names corresponding to the elements in the model keys.

        Notes
        -----
        The model key names are stored as metadata within the output directory.
        '''
        all_metadata = self.read_all_metadata()
        all_metadata['model_key_names'] = model_key_names
        self.write_all_metadata(all_metadata)
    #endregion

    #region: read_model_key_names
    def read_model_key_names(self):
        '''
        Read the model key names from the metadata.

        Returns
        -------
        list of str
            List of model key names.
        '''
        all_metadata = self.read_all_metadata()
        return all_metadata['model_key_names']
    #endregion

    #region: get_name_key_mapping
    def get_name_key_mapping(self, model_key):
        '''
        Map the elements of a model key to their corresponding names.

        Parameters
        ----------
        model_key : tuple of str
            Model key whose elements are to be mapped.

        Returns
        -------
        dict
            Dictionary mapping model key names to their corresponding elements 
            in the given model key.
        '''
        model_key_names = self.read_model_key_names()
        return dict(zip(model_key_names, model_key))
    #endregion

    #region: write_identifier_key_mapping
    def write_identifier_key_mapping(self, model_key_for_id):
        '''
        Write the mapping between model identifiers and their corresponding 
        model keys to the metadata file.

        Parameters
        ----------
        model_key_for_id : dict
            A dictionary with identifiers (str) as keys and model keys (list) 
            as values.

        Notes
        -----
        This method updates the metadata with the provided mapping and 
        writes the updated metadata back to the file.
        '''
        all_metadata = self.read_all_metadata()
        all_metadata['model_key_for_id'] = model_key_for_id
        self.write_all_metadata(all_metadata)
    #endregion

    #region: read_identifier_key_mapping
    def read_identifier_key_mapping(self):
        '''
        Retrieve the mapping between model identifiers and their corresponding 
        model keys.

        Returns
        -------
        dict
            A dictionary where keys are identifiers (str) and values are model 
            keys (tuple).

        Notes
        -----
        The method reads the metadata file, extracts the mapping, 
        and converts model keys from lists (as stored in JSON) to tuples.
        '''
        all_metadata = self.read_all_metadata()
        mapping = all_metadata['model_key_for_id']
        return {id : tuple(model_key) for id, model_key in mapping.items()}
    #endregion

    #region: write_configuration
    def write_configuration(self, configuration):
        '''
        Write all configuration settings to a JSON file.

        The configuration is stored as a nested dictionary mapping each 
        configuration category name to the corresponding settings.

        Parameters
        ----------
        configuration : UnifiedConfiguration
            Contains all configuration settings.
        '''
        all_metadata = self.read_all_metadata()
        all_metadata['configuration'] = configuration.to_dict()
        self.write_all_metadata(all_metadata)
    #endregion

    #region: read_configuration
    def read_configuration(self):
        '''
        Retrieve the configuration metadata.

        Returns
        -------
        dict
            Configuration stored as a nested dictionary mapping each 
            configuration category name to the corresponding settings.
        '''
        all_metadata = self.read_all_metadata()
        return all_metadata['configuration']
    #endregion

    #region: write_all_metadata
    def write_all_metadata(self, metadata):
        '''
        Write all metadata to a JSON file.

        Parameters
        ----------
        metadata : dict
            Dictionary containing the metadata to be written.

        Notes
        -----
        If the directory does not exist, it will be created.
        '''
        # Ensure the directory exists
        directory = os.path.dirname(self._metadata_path)
        os.makedirs(directory, exist_ok=True)

        with open(self._metadata_path, 'w') as file:
            json.dump(metadata, file)
    #endregion

    #region: read_all_metadata
    def read_all_metadata(self):
        '''
        Read all metadata from a JSON file.

        Returns
        -------
        dict
            Dictionary containing the metadata read from the file. Returns an 
            empty dictionary if the file does not exist.
        '''
        if not os.path.exists(self._metadata_path):
            return {}
        
        with open(self._metadata_path, 'r') as file:
            return json.load(file)
    #endregion

    #region: _metadata_path
    @property
    def _metadata_path(self):
        '''
        Get the path to the metadata file within the output directory.

        Returns
        -------
        str
            Path to the metadata file.
        '''
        return os.path.join(self.output_dir, 'metadata.json')
    #endregion

    #region: group_model_keys
    def group_model_keys(
            self,
            exclusion_key_names , 
            string_to_exclude=None,
            model_keys=None,
        ):
        '''
        Group model keys by grouping keys. A grouping key is formed by taking a model 
        key and excluding one or more of its elements.

        Parameters
        ----------
        exclusion_key_names : str or list of str
            Names of keys (which should be in the model key names stored in the class) to 
            exclude when forming the grouping key.
        string_to_exclude : str, optional
            String to exclude model keys containing it. If a model key contains this 
            string, it will be excluded from the final output. If None, no model keys 
            are excluded based on this criterion.
        model_keys : list of tuples, optional
            Model keys to be grouped. Each tuple represents a model key. If None, 
            all model keys available in the ResultsManager object will be used.

        Returns
        -------
        grouped_model_keys : list of 2-tuple
            Contains each grouping key and its corresponding group of model keys.

        Note
        ----
        This method assumes that the model keys and key names are stored within the
        ResultsManager object.
        '''
        if model_keys is None:
            # Use all model keys.
            model_keys = self.read_model_keys()

        if isinstance(exclusion_key_names , str):
            exclusion_key_names  = [exclusion_key_names]

        if string_to_exclude:
            # Exclude model keys that contain string_to_exclude
            model_keys = [k for k in model_keys if string_to_exclude not in k]

        # Get the indices of the keys to exclude in the key names
        exclusion_key_indices = [
            self.read_model_key_names().index(key) 
            for key in exclusion_key_names
            ]

        def create_grouping_key(model_key):
            return tuple(item for idx, item in enumerate(model_key) 
                        if idx not in exclusion_key_indices)

        # Sort model keys by grouping key
        # This is necessary because itertools.groupby() groups only consecutive 
        # elements with the same key
        sorted_model_keys = sorted(model_keys, key=create_grouping_key)

        # Group the sorted keys by grouping key
        grouped_model_keys = [
            (grouping_key, list(group))
            for grouping_key, group in itertools.groupby(
            sorted_model_keys, key=create_grouping_key)
        ]

        return grouped_model_keys
    #endregion