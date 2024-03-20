'''
This module contains utility functions that leverage the RDKit package.
'''

from rdkit.Chem import PandasTools
from rdkit.Chem import MolFromSmiles, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
import os

from . import utilities

#region: sdf_to_dataframe
def sdf_to_dataframe(sdf_directory, write_path=None):
    '''
    Parse one or more SDF V2000 files into a pandas.DataFrame.

    The DataFrame's index corresponds to the chemicals, and the columns are
    the SDF data fields.

    Parameters
    ----------
    sdf_directory : str
        Path to the directory containing the SDF files.
    write_path : str, optional
        Path to the output file. If present, will write the DataFrame to disk.

    Returns
    -------
    pandas.DataFrame
    '''
    if not os.path.exists(sdf_directory):
        raise ValueError(f'Directory {sdf_directory} does not exist.')
    
    sdf_data = []  # initialize
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

    if write_path:
        utilities.ensure_directory_exists(write_path)
        sdf_data.to_parquet(write_path, index=False, compression='gzip')

    return sdf_data
#endregion

# TODO: Inverse log-transform for consistency with other features sets?
#region: get_2d_descriptors
def get_2d_descriptors(
        smiles_for_chem, 
        index_name,
        discrete_suffix=None,
        write_path=None
        ):
    '''
    Get all two-dimensional molecular descriptors from RDKit.

    Parameters
    ----------
    smiles_for_chem : dict
        Mapping of DTXSID to SMILES string.
    index_name : str
        Used to name the index, e.g., 'DTXSID'.
    discrete_suffix : str (optional)
        Will be appended to the end of each string in discrete columns.
    write_path : str (optional)
        Path to write the return as a parquet file.
    
    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    RDKit also provides some more recent "2D-autocorrelation" descriptors, 
    but the functions may have not been fully debugged.
    '''
    mol_for_chem = {
        chem: MolFromSmiles(smiles) 
        for chem, smiles in smiles_for_chem.items()}

    # Filter out None values which correspond to parsing errors.
    mol_for_chem = {
        chem: mol for chem, mol in mol_for_chem.items() 
        if mol is not None}

    desc_list = [tup[0] for tup in Descriptors._descList]
    calc = (
        MoleculeDescriptors
        .MolecularDescriptorCalculator(desc_list))

    descriptors = pd.DataFrame.from_dict(
        {chem: calc.CalcDescriptors(mol) 
        for chem, mol in mol_for_chem.items()}, 
        orient='index',
        columns=calc.GetDescriptorNames())
    descriptors.index.name = index_name

    if discrete_suffix:
        discrete_columns = list(descriptors.select_dtypes('int'))
        descriptors = utilities.tag_discrete_columns(
            descriptors, 
            discrete_columns, 
            discrete_suffix
        )
        
    if write_path is not None:
        utilities.ensure_directory_exists(write_path)
        descriptors.to_parquet(write_path)

    return descriptors
#endregion