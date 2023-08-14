'''
The `results_management` module provides the `ResultsManager` class, 
responsible for managing the writing, reading, and handling of results 
and fitted estimators. It includes methods for saving and retrieving results 
in CSV format, handling fitted estimator objects with Joblib, managing 
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

#region: ResultsManager
class ResultsManager:
    '''
    Manage the reading and writing of results and fitted estimators.

    The `ResultsManager` class provides methods to handle the storage and 
    retrieval of various types of results including model performances, 
    importances, fitted estimators, and metadata such as headers and model key 
    names. It supports reading and writing CSV files for results and Joblib 
    files for estimators.

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
        Writes the fitted estimator to a Joblib file.
    read_estimator(model_key)
        Reads the fitted estimator from a Joblib file.
    write_result(result_df, model_key, result_type)
        Writes the result DataFrame to a CSV file.
    read_result(model_key, result_type)
        Reads the result CSV file and returns a DataFrame.
    ... (other methods)
    '''
    def __init__(self, output_dir):
        '''
        Initialize the ResultsManager with the specified output directory.

        Parameters
        ----------
        output_dir : str
            Path to the directory where results and estimators will be saved.
        '''
        self.output_dir = output_dir
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
        Write a result DataFrame to a CSV file.

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
        The CSV file is saved in a subdirectory named after the model_key,
        within the output directory.
        '''
        # Create directory structure
        path = self._build_path(model_key, result_type, 'csv')
        # Write DataFrame to disk
        result_df.to_csv(path)
        self.write_level_names(result_df.columns.names, result_type)
    #endregion

    #region: read_result
    def read_result(self, model_key, result_type):
        '''
        Read the result CSV file and return a DataFrame.

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
        The CSV file is read from a subdirectory named after the model_key,
        within the output directory.
        '''
        path = self._build_path(model_key, result_type, 'csv')
        level_names = self.read_level_names()[result_type]
        header_indices = list(range(len(level_names)))
        result_df = pd.read_csv(path, header=header_indices, index_col=0)
        result_df.columns.names = level_names
        return result_df
    #endregion

    #region: _build_path
    def _build_path(self, model_key, result_type, extension):
        '''
        Build the full path for the specified file.

        Parameters
        ----------
        model_key : tuple of str
            Model key identifying the result.
        result_type : str
            Type of result (e.g., "performances", "importances").
        extension : str
            File extension (e.g., "csv", "joblib").

        Returns
        -------
        str
            Full path to the file.

        Notes
        -----
        This method builds the path based on the given parameters and ensures
        that the corresponding directory exists.
        '''
        directory_name = self._convert_model_key(model_key=model_key)
        directory = os.path.join(self.output_dir, directory_name)
        self._ensure_directory(directory)
        
        return os.path.join(directory, f'{result_type}.{extension}')
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

    #region: list_model_keys
    def list_model_keys(self):
        '''
        List all model keys present in the output directory.

        Returns
        -------
        list of tuple
            List of model keys, where each model key is represented as a tuple of 
            strings.
        '''
        model_keys = []

        for entry in os.listdir(self.output_dir):
            entry_path = os.path.join(self.output_dir, entry)

            if os.path.isdir(entry_path):
                model_key = self._convert_model_key(directory_name=entry)
                model_keys.append(model_key)

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

    #region: get_model_key_mapping
    def get_model_key_mapping(self, model_key):
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
        return dict(zip(self.model_key_names, model_key))
    #endregion

    #region: _convert_model_key
    def _convert_model_key(self, model_key=None, directory_name=None):
        '''
        Convert between model key and directory name representations.

        Parameters
        ----------
        model_key : tuple of str, optional
            Model key to be converted into directory name.
        directory_name : str, optional
            Directory name to be converted into model key.

        Returns
        -------
        str or tuple of str
            Converted directory name (if model_key is provided) or model key 
            (if directory_name is provided).

        Notes
        -----
        Either model_key or directory_name must be provided, but not both.
        '''
        if model_key:
            # Build directory name
            return '-'.join(model_key)
        elif directory_name:
            # Collapse directory name into model key
            return tuple(directory_name.split('-'))
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