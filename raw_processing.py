'''
'''

import pandas as pd
import re

import pattern

#region: RawDataProcessor.__init__
class RawDataProcessor:
    '''
    '''
    def __init__(self, path_settings):
        self._path_settings = path_settings 
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
