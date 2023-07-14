'''Various helper functions for using the RDKit library.
'''

import pandas as pd
from rdkit.Chem import MolFromSmiles, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

#region: molecular_descriptors_2d
def molecular_descriptors_2d(
        smiles_for_chem, chemicals_to_exclude=None, write_path=None):
    '''Get all two-dimensional molecular descriptors from RDKit.

    Parameters
    ----------
    smiles_for_chem : pandas.Series
        SMILES (str) for each chemical.
    chemicals_to_exclude : list of str (optional)
        List of SMILES to exclude from the return.
    write_path : str (optional)
        Path to write the return as a CSV file.
    
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
        
    if chemicals_to_exclude is not None:
        mol_for_chem = {
            chem: mol for chem, mol in mol_for_chem.items() 
            if chem not in chemicals_to_exclude}

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
        
    descriptors.index.name = smiles_for_chem.index.name

    if write_path is not None:
        descriptors.to_csv(write_path)

    return descriptors
#endregion