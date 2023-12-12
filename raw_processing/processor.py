'''
This module processes raw datasets to extract, clean, and structure relevant 
features and targets for the LCIA-QSAR modeling workflow.

This module contains the RawDataProcessor class, which provides various 
methods to handle raw input files, convert them to a structured format, 
and save the resulting data to disk. The processed data are suitable for 
subsequent modeling processes in the LCIA-QSAR workflow.

The raw data processing involves several tasks including extraction of 
chemical identifiers, parsing and cleaning of data files for feature 
and target variable extraction, and saving them in a structured format.

Example
-------
    from raw_processing import RawDataProcessor
    processor = RawDataProcessor(
        raw_data_settings, data_settings, path_settings
        )
    processor._surrogate_pods_from_raw()
    processor._comptox_features_from_raw()
    ... (and other processing methods)
'''

import pandas as pd
import os.path

from . import opera 
from . import comptox
from . import other_sources
from . import rdkit_utilities

# TODO:
'''
Need to make directories if not exist. This class could handle all writing 
logic, and the other modules could simply handle the processing from raw.
'''

#region: RawDataProcessor.__init__
class RawDataProcessor:
    '''
    Processe raw datasets to extract and structure relevant features and 
    targets for the LCIA-QSAR modeling workflow.

    The RawDataProcessor class provides methods to handle raw input files, 
    process the raw data, and save the resulting structured data to disk 
    in a format suitable for subsequent modeling.
    '''
    def __init__(self, raw_data_settings, data_settings, path_settings):
        '''
        Initialize the RawDataProcessor with configuration settings.

        Parameters
        ----------
        raw_data_settings : SimpleNamespace
            Configuration settings specific to raw data processing.
        data_settings : SimpleNamespace
            Workflow data configuration settings. 
        path_settings : SimpleNamespace
            Configuration settings for file paths.
        '''
        self._raw_data_settings = raw_data_settings
        self._data_settings = data_settings
        self._path_settings = path_settings 
        # NOTE: Only DTXSID identifier has been tested
        self._index_col = 'DTXSID'

        # Map data types to their respective processing function
        self._dispatcher = {
            'dsstox_sdf_data' : self._dsstox_sdf_data_from_raw,
            'opera_features' : self._opera_features_from_raw,
            'comptox_features' : self._comptox_features_from_raw,
            'surrogate_pods' : self._surrogate_pods_from_raw,
            'authoritative_pods' : self._authoritative_pods_from_raw,
            'experimental_ld50s' : self._experimental_ld50s_from_raw,
            'seem3_exposure_data' : self._seem3_exposure_data_from_raw,
            'toxcast_oeds' : self._oral_equivalent_doses_from_raw
        }
#endregion

    #region: process_from_raw
    def process_from_raw(self, data_type):
        '''
        '''
        return self._dispatcher[data_type]()
    #endregion

    #region: _dsstox_sdf_data_from_raw
    def _dsstox_sdf_data_from_raw(self):
        '''
        Extract and process all SDF V2000 files in the DSSTox database.

        This method reads SDF files containing chemical identifiers and other
        data from the DSSTox database. The data are distributed across 
        multiple SDF files where each file contains several thousand unique 
        chemicals. The resulting DataFrame is written to a Parquet file.

        Returns
        -------
        pandas.DataFrame
            SDF data for all batches of chemicals.

        References
        ----------
        https://www.epa.gov/comptox-tools/comptox-chemicals-dashboard-release-notes#latest%20version
        '''
        sdf_data = rdkit_utilities.sdf_to_dataframe(
            self._path_settings.dsstox_sdf_dir,
            write_path=self._build_path_dsstox_compiled()
            )

        # Write the DTXSID column to a text file for OPERA 2.9
        dtxsid_column = self._raw_data_settings.dsstox_sdf_dtxsid_column
        dtxsid_file = self._build_path_dsstox_identifiers()
        sdf_data[dtxsid_column].to_csv(dtxsid_file, header=False, index=False)

        return sdf_data
    #endregion

    #region: _build_path_dsstox_compiled
    def _build_path_dsstox_compiled(self):
        '''
        Helper function to build a path to the output file containing all 
        DSSTox data.

        The file name is derived from the directory name and extension.
        '''
        sdf_directory = self._path_settings.dsstox_sdf_dir
        directory_name = os.path.split(sdf_directory)[-1]
        file_name = f'{directory_name}.parquet'
        return os.path.join(sdf_directory, file_name)
    #endregion

    #region: _build_path_dsstox_identifiers
    def _build_path_dsstox_identifiers(self):
        '''
        Helper function to build a path to the output file containing the
        DSSTox identifiers (DTXSID).

        The file name is derived from the column name and extension.
        '''
        dtxsid_column = self._raw_data_settings.dsstox_sdf_dtxsid_column
        return os.path.join(
            self._path_settings.dsstox_sdf_dir, 
            f'{dtxsid_column}.txt'
            )
    #endregion

    #region: _surrogate_pods_from_raw
    def _surrogate_pods_from_raw(self):
        '''
        Extract and process surrogate toxicity values from raw data.

        This method reads raw surrogate toxicity values from an Excel file, 
        processes the data by selecting specific toxicity metrics, and optionally 
        applies a log10 transformation. The processed data is then saved to 
        disk.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing processed surrogate toxicity values.
        '''
        surrogate_pods = other_sources.surrogate_toxicity_values_from_excel(
            self._path_settings.raw_surrogate_pods_file, 
            self._raw_data_settings.tox_metric, 
            self._index_col.lower(), 
            self._raw_data_settings.surrogate_tox_data_kwargs,
            log10=self._raw_data_settings.do_log10_target,
            effect_mapper=self._raw_data_settings.effect_mapper,
            write_path=self._path_settings.surrogate_pods_file
        )

        return surrogate_pods
    #endregion

    # TODO: Remove "opera_file_namer". Use file extension (CSV) instead.
    #region: _opera_features_from_raw
    def _opera_features_from_raw(self):
        '''
        Extract and process OPERA features from raw data batches.

        This method reads and processes raw feature data from OPERA. It also 
        saves the data to a Parquet file on disk.

        Additionally, applicability domain (AD) flags are extracted and saved.

        Returns
        -------
        tuple of pandas.DataFrame
            A tuple containing two DataFrames:
            1. AD flags for each chemical.
            2. Processed OPERA features.
        '''
        prefix = self._raw_data_settings.opera_file_prefix
        extension = self._raw_data_settings.opera_file_extension
        opera_file_namer = lambda name: prefix + name + extension

        AD_flags, X_opera = opera.process_all_batches(
            self._path_settings.raw_opera_features_dir, 
            self._path_settings.opera_mapper_file,
            opera_file_namer,
            self._raw_data_settings.logging_file_name, 
            index_name=self._index_col, 
            discrete_columns=self._data_settings.discrete_columns_for_source['opera'],
            discrete_suffix=self._data_settings.discrete_column_suffix,
            log10_pat=self._raw_data_settings.opera_log10_pat
        )
        # Drop any chemicals missing all features (e.g., inorganics)
        X_opera = X_opera.dropna(how='all')
        AD_flags = AD_flags.loc[X_opera.index]

        features_write_path=self._path_settings.file_for_features_source['opera']
        X_opera.to_parquet(features_write_path, compression='gzip')

        flags_write_path=self._path_settings.opera_AD_file
        AD_flags.to_parquet(flags_write_path, compression='gzip')

        return AD_flags, X_opera
    #endregion

    #region: _comptox_features_from_raw
    def _comptox_features_from_raw(self):
        '''
        Extract and process CompTox features from raw data.

        This method reads raw feature data from CompTox and processes them by 
        excluding certain chemicals based on OPERA data. The processed data 
        are then saved to a CSV file on disk.

        Returns
        -------
        pandas.DataFrame
            The processed CompTox features.
        '''
        # TODO: Is there a better way to identify these chemicals?
        chemicals_to_exclude = opera.chemicals_to_exclude_from_qsar(
            self._path_settings.chemical_identifiers_file, 
            self._path_settings.opera_structures_file
        )

        return comptox.opera_test_predictions_from_csv(
            self._path_settings.raw_comptox_features_file, 
            self._index_col, 
            chemicals_to_exclude=chemicals_to_exclude,
            columns_to_exclude=self._raw_data_settings.comptox_columns_to_exclude,
            log10_pat=self._raw_data_settings.comptox_log10_pat, 
            write_path=self._path_settings.file_for_features_source['comptox']
        )
    #endregion

    #region: _experimental_ld50s_from_raw
    def _experimental_ld50s_from_raw(self):
        '''
        Extract and process experimental LD50 values from raw data.

        This method reads raw LD50 experimental data and processes them by 
        filtering based on CompTox identifiers. The processed data are then 
        saved to a CSV file on disk.

        Returns
        -------
        pandas.DataFrame
            The processed experimental LD50 values.
        '''
        return other_sources.experimental_ld50s_from_excel(
            self._path_settings.raw_ld50_experimental_file, 
            self._raw_data_settings.ld50_exp_column, 
            id_for_casrn=self._map_casrn_to_dtxsid(), 
            id_name=self._index_col,
            write_path=self._path_settings.ld50_experimental_file
        )
    #endregion
        
    #region: _authoritative_pods_from_raw
    def _authoritative_pods_from_raw(self):
        '''
        Extract and process authoritative Points of Departure from raw data.

        This method reads raw authoritative PODs and processes them by mapping 
        CASRN to the specified chemical identifier. The processed data are then 
        saved to a CSV file on disk.

        Returns
        -------
        pandas.DataFrame
            The processed authoritative Points of Departure values.
        '''
        return other_sources.authoritative_toxicity_values_from_excel(
            self._path_settings.raw_authoritative_pods_file, 
            self._raw_data_settings.auth_data_kwargs,
            self._raw_data_settings.auth_file_ilocs_for_effect, 
            id_for_casrn=self._map_casrn_to_dtxsid(), 
            id_name=self._index_col, 
            write_path=self._path_settings.authoritative_pods_file
        )
    #endregion

    #region: _map_casrn_to_dtxsid
    def _map_casrn_to_dtxsid(self):
        '''
        Map CASRN to DTXSID using the DSSTox dataset.

        Returns
        -------
        dict
            A dictionary mapping CASRN to DTXSID.
        '''
        casrn_column = self._raw_data_settings.dsstox_sdf_casrn_column
        dtxsid_column = self._raw_data_settings.dsstox_sdf_dtxsid_column

        return (
            pd.read_parquet(self._build_path_dsstox_compiled())
            .set_index(casrn_column)
            [dtxsid_column]
            .to_dict()
        )
    #endregion

    #region: _seem3_exposure_data_from_raw
    def _seem3_exposure_data_from_raw(self):
        '''
        Extract and process SEEM3 exposure data from raw data.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing exposure predictions.
        '''
        return other_sources.seem3_exposure_data_from_excel(
            self._path_settings.raw_seem3_exposure_file,
            self._raw_data_settings.seem3_data_kwargs,
            self._index_col,
            write_path=self._path_settings.seem3_exposure_file
        )
    #endregion

    #region: _oral_equivalent_doses_from_raw
    def _oral_equivalent_doses_from_raw(self):
        '''
        Extract and process oral equivalent doses from raw data.

        Returns
        -------
        pandas.Series
        '''
        return other_sources.oral_equivalent_doses_from_excel(
            self._path_settings.raw_toxcast_oeds_file,
            self._raw_data_settings.oed_data_kwargs,
            self._index_col,
            write_path=self._path_settings.toxcast_oeds_file
        )
    #endregion