'''
'''

import streamlit as st
from rdkit import Chem 
from rdkit.Chem import Draw

import plot

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
    st.header('Points of Departure, log10[mg/(kg-d)]')
    st.write(chem_pods)
#endregion

#region: pod_figure
def pod_figure(fig, chem_pod_data, title):
    '''
    '''
    st.header(title)
    # Get the single Axes object for modification
    ax = fig.get_axes()[0]
    plot.single_chemical_pod_data(ax, chem_pod_data)
    st.pyplot(fig)
#endregion

#region: margins_of_exposure
def margins_of_exposure(chem_moe_data):
    '''
    '''
    st.header('Margins of Exposure')
    st.write(chem_moe_data)
#endregion

#region: moe_figure
def moe_figure(fig, chem_moe_data, title):
    '''
    '''
    st.header(title)
    # Get the single Axes object for modification
    ax = fig.get_axes()[0]
    plot.single_chemical_moe_data(ax, chem_moe_data)
    st.pyplot(fig)
#endregion