'''
This module contains utility functions that leverage the RDKit package.
'''

from rdkit.Chem import PandasTools
import pandas as pd
import os

#region: sdf_to_dataframe
def sdf_to_dataframe(sdf_directory, do_write=True):
    '''
    Parse one or more SDF V2000 files into a pandas.DataFrame.

    The DataFrame's index corresponds to the chemicals, and the columns are
    the SDF data fields.

    Parameters
    ----------
    sdf_directory : str
        Path to the directory containing the SDF files.
    do_write : bool, optional
        Whether to write the DataFrame to a Parquet file. Default True.

    Returns
    -------
    pandas.DataFrame
    '''
    if not os.path.exists(sdf_directory):
        raise ValueError(f'Directory {sdf_directory} does not exist.')
    
    sdf_data = []  # initialize
    # Use the directory name for the output file name
    directory_name = os.path.split(sdf_directory)[-1]

    for dirpath, _, filenames in os.walk(sdf_directory):
        sdf_files = [f for f in filenames if f.endswith('.sdf')]

        for sdf_file in sdf_files:
            sdf_path = os.path.join(dirpath, sdf_file)
            sdf_data_subset = PandasTools.LoadSDF(
                sdf_path, 
                smilesName='Canonical_SMILES',
                molColName=None  # to enable binary storage
            )
            sdf_data.append(sdf_data_subset)

    if not sdf_data:
        raise ValueError('No SDF files found in the specified directory.')
    
    sdf_data = pd.concat(sdf_data, ignore_index=True)

    if do_write:
        file_name = f'{directory_name}.parquet'
        write_path = os.path.join(sdf_directory, file_name)
        sdf_data.to_parquet(write_path, index=False, compression='gzip')

    return sdf_data
#endregion