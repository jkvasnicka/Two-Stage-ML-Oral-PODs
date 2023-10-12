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
    processor.targets_from_raw()
    processor.comptox_features_from_raw()
    ... (and other processing methods)
'''

import pandas as pd
import re

from . import pattern
from . import parse
from . import opera 
from . import comptox

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
#endregion

    #region: get_labeled_identifiers
    def get_labeled_identifiers(self, sheet_name='ORAL', do_write=True):
        '''
        Extract chemical identifiers with labels for modeling.

        Parameters
        ----------
        sheet_name : str, optional
            Name of the sheet in the Excel file from which identifiers 
            are to be extracted. Default is 'ORAL'.
        do_write : bool, optional
            Whether to write the extracted identifiers to a TXT file. Default 
            is True.

        Returns
        -------
        pandas.Series
            A series containing the extracted chemical identifiers.
        '''
        # NOTE: Only DTXSID has been verified to work
        identifier_type = 'dtxsid'

        # Define key-word arguments for pandas.read_excel().
        kwargs = {
            'sheet_name': sheet_name,
            'header': [0, 1]
            }

        identifiers = (
            pd.read_excel(
                self._path_settings.raw_surrogate_pods_file, 
                **kwargs)
            .droplevel(axis=1, level=0)
            [['dtxsid', 'casrn']]
            )
        
        pattern_for_col = {
            'dtxsid': re.compile(pattern.dtxsid(as_group=True)),
            'casrn': re.compile(pattern.casrn(as_group=True))
        }
        for col, pat in pattern_for_col.items():
            identifiers.loc[:, col] = (
                identifiers[col].str.extract(pat, expand=False))

        identifiers = identifiers[identifier_type].dropna()

        if do_write:
            # Write contents to TXT file for batch download in OPERA.
            identifiers.to_csv(
                self._path_settings.chemical_id_dev_file, 
                header=None, 
                index=None, 
                sep=' '
            )

        return identifiers
    #endregion

    #region: get_all_identifiers
    def get_all_identifiers(self, do_write=True):
        '''
        Extract all chemical identifiers for modeling.

        Parameters
        ----------
        do_write : bool, optional
            Whether to write the extracted identifiers to a TXT file. Default 
            is True.

        Returns
        -------
        pandas.Series
            A series containing all the extracted chemical identifiers.
        '''
        # NOTE: Only DTXSID has been verified to work
        identifiers = (
            pd.read_csv(
                self._path_settings.seem3_exposure_file,
                encoding='latin-1')
            ['DTXSID']
        )

        if do_write:
            identifiers.to_csv(
                self._path_settings.chemical_id_app_file,    
                index=False, 
                header=False,
                sep=' '
            )

        return identifiers
    #endregion

    #region: targets_from_raw
    def targets_from_raw(self):
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
        surrogate_pods = parse.surrogate_toxicity_values_from_excel(
            self._path_settings.raw_surrogate_pods_file, 
            self._raw_data_settings.sheet_name,
            self._raw_data_settings.tox_metric, 
            self._index_col.lower(), 
            log10=self._raw_data_settings.do_log10_target,
            effect_mapper=self._raw_data_settings.effect_mapper,
            write_path=self._path_settings.surrogate_pods_file
        )

        return surrogate_pods
    #endregion

    # FIXME: For backwards compatibility, training data separate
    #region: opera_features_from_raw
    def opera_features_from_raw(self):
        '''
        Extract and process OPERA features from raw data batches.

        This method reads and processes raw feature data from OPERA, including 
        training and application batches. It combines the processed feature 
        data, removes duplicates based on chemical intersections, and saves the 
        combined data to a CSV file on disk.

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

        # Load the training data
        AD_flags_train, opera_features_train = opera.parse_data_with_applicability_domains(
            self._path_settings.raw_opera_features_dir, 
            self._path_settings.opera_mapper_file, 
            opera_file_namer, 
            index_name=self._index_col, 
            discrete_columns=self._data_settings.discrete_columns_for_source['opera'],
            discrete_suffix=self._data_settings.discrete_column_suffix,
            log10_pat=self._raw_data_settings.opera_log10_pat
        )

        AD_flags_app, opera_features_app = opera.process_all_batches(
            self._path_settings.opera_application_batches_dir, 
            self._path_settings.opera_mapper_file,
            opera_file_namer,
            self._raw_data_settings.structures_file_name, 
            self._raw_data_settings.logging_file_name, 
            index_name=self._index_col, 
            discrete_columns=self._data_settings.discrete_columns_for_source['opera'],
            discrete_suffix=self._data_settings.discrete_column_suffix,
            log10_pat=self._raw_data_settings.opera_log10_pat
        )

        # Drop duplicates. 
        chem_intersection = list(
            opera_features_train.index.intersection(opera_features_app.index))
        AD_flags_app = AD_flags_app.drop(chem_intersection)
        opera_features_app = opera_features_app.drop(chem_intersection)

        data_write_path=self._path_settings.file_for_features_source['opera']
        flags_write_path=self._path_settings.opera_AD_file

        opera_features = pd.concat([opera_features_train, opera_features_app])
        opera_features.to_csv(data_write_path)

        AD_flags = pd.concat([AD_flags_train, AD_flags_app])
        AD_flags.to_csv(flags_write_path)

        return AD_flags, opera_features
    #endregion

    #region: comptox_features_from_raw
    def comptox_features_from_raw(self):
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
        chemicals_to_exclude = opera.chemicals_to_exclude_from_qsar(
            self._path_settings.chemical_id_dev_file, 
            self._path_settings.chemical_structures_dev_file
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

    #region: experimental_ld50s_from_raw
    def experimental_ld50s_from_raw(self):
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
        chem_identifiers = self.load_comptox_identifiers()

        return parse.experimental_ld50s_from_excel(
            self._path_settings.raw_ld50_experimental_file, 
            chem_identifiers, 
            self._index_col, 
            ld50_columns=self._raw_data_settings.ld50_columns, 
            write_path=self._path_settings.ld50_experimental_file
        )
    #endregion

    #region: regulatory_pods_from_raw
    def regulatory_pods_from_raw(self):
        '''
        Extract and process regulatory Points of Departure from raw data.

        This method reads raw regulatory PODs and processes them by mapping 
        CASRN to the specified chemical identifier. The processed data are then 
        saved to a CSV file on disk.

        Returns
        -------
        pandas.DataFrame
            The processed regulatory Points of Departure values.

        '''
        chem_identifiers = self.load_comptox_identifiers()

        # FIXME: For backwards compatibility
        # Map CASRN to index_col for replacing the original index.
        chem_id_for_casrn = (
            chem_identifiers
            .reset_index()
            .set_index('CASRN')[self._index_col]
            .to_dict()
        )

        return parse.regulatory_toxicity_values_from_csv(
            self._path_settings.raw_regulatory_pods_file, 
            self._raw_data_settings.reg_file_ilocs_for_effect, 
            chem_id_for_casrn=chem_id_for_casrn, 
            new_chem_id=self._index_col, 
            write_path=self._path_settings.regulatory_pods_file
        )
    #endregion

    # FIXME: For backwards compatibility.
    #region: load_comptox_identifiers
    def load_comptox_identifiers(self):
        '''
        Load chemical identifiers from the CompTox database.

        This method retrieves the stored chemical identifiers from a CSV file 
        on disk.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing chemical identifiers.

        '''
        return pd.read_csv(
            self._path_settings.comptox_identifiers_file, 
            index_col=self._index_col
        )
    #endregion

    #region: oral_equivalent_doses_from_raw
    def oral_equivalent_doses_from_raw(self):
        '''
        Extract and process oral equivalent doses from raw ToxCast data.

        This method retrieves oral equivalent doses (OEDs) from the raw ToxCast
        dataset, performs necessary transformations (e.g., taking the logarithm), 
        and then writes a CSV file to disk.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing processed oral equivalent doses.
        '''
        return parse.toxcast_expocast_from_csv(
            self._path_settings.raw_toxcast_oeds_file, 
            self._index_col, 
            data_columns=self._raw_data_settings.oed_columns,
            log10=self._raw_data_settings.do_log10_target,
            write_path=self._path_settings.toxcast_oeds_file
        )            
    #endregion