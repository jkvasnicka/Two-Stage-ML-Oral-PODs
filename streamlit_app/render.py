'''
'''

import streamlit as st
from rdkit import Chem 
from rdkit.Chem import Draw

# TODO: Input chemical_id and check if present
# TODO: Derive headers from config?

#region: qsar_ready_structure
def qsar_ready_structure(qsar_smiles):
    '''
    '''
    st.header('QSAR-Ready Structure')
    image = structure_as_image(qsar_smiles)
    st.image(image)
#endregion

#region: structure_as_image
def structure_as_image(smiles):
    '''
    '''
    # Convert the SMILES to a molecule object
    mol = Chem.MolFromSmiles(smiles)

    if not mol:
        raise ValueError(f'Invalid SMILES string: {smiles}')

    # Draw the molecule as an image
    return Draw.MolToImage(mol)
#endregion

#region: features
def features(chem_features):
    '''
    ''' 
    st.header('Features from OPERA 2.9')
    st.write(chem_features)
#endregion

#region: points_of_departure
def points_of_departure(chem_pods):
    '''
    '''
    st.header('POD Estimates log10[mg/(kg-d)]')
    st.write(chem_pods)
#endregion