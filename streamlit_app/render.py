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

#region: points_of_departure
def points_of_departure(chem_pod_data):
    '''
    '''
    pods = chem_pod_data['POD']
    pods.name = 'log10 POD'
    st.write(pods)
#endregion

#region: pod_figure
def pod_figure(fig, chem_pod_data):
    '''
    '''
    # Get the single Axes object for modification
    ax = fig.get_axes()[0]
    plot.single_chemical_pod_data(ax, chem_pod_data)
    st.pyplot(fig)
    st.markdown(
        '''
        **Figure 1.** Cumulative distributions of point of departure across 
        different data sources. “Regulatory” refers to the regulatory 
        values. “ToxValDB” refers to the surrogate values derived using an 
        approach by Aurisano et al. (2023). “QSAR” refers to the final model 
        developed in this study, described in the main text. Data are shown
        for chemicals within the applicability domain of SEEM3.
        '''
    )
#endregion

#region: margins_of_exposure
def margins_of_exposure(chem_moe_data):
    '''
    '''
    chem_moe_data = (
        chem_moe_data
        .unstack()
        .drop('Cum_Count')
    )
    st.write(chem_moe_data)
#endregion

#region: moe_figure
def moe_figure(fig, chem_moe_data):
    '''
    '''
    # Get the single Axes object for modification
    ax = fig.get_axes()[0]
    plot.single_chemical_moe_data(ax, chem_moe_data)
    st.pyplot(fig)
    st.markdown(
        '''
        **Figure 2.** Cumulative counts of chemicals in relation to their 
        margins of exposure for an individual at the population median 
        exposure. Uncertainty is represented in two ways: (1) Exposure 
        uncertainty, reflected by examining margins of exposure at different 
        exposure percentiles; (2) Point of departure (hazard) uncertainty, 
        represented by a 90% prediction interval derived from the median 
        root-mean squared error based on cross validation. Vertical spans 
        highlight different risk categories. Data are shown for chemicals 
        within the applicability domain of SEEM3.
        '''
    )
#endregion