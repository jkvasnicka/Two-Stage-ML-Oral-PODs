'''
'''

import streamlit as st 

from data_management import DataManager
import plot

data_manager = DataManager()

st.set_page_config(
    page_title='Chemical Prediction',
    layout='wide'
)

st.title('Chemical Prediction Tool')

with st.sidebar:
    st.header('User Input')

    chemical_id = st.text_input('Enter the DTXSID for the chemical')

    effect_label = st.selectbox(
        'Select the effect category',
        options=data_manager.effect_labels
    )

    predict_button = st.button('Get POD Estimates')

data_manager.chemical_id = chemical_id
data_manager.effect_label = effect_label

# Using columns to display content side-by-side
if chemical_id:
    col1, col2 = st.columns(2)  # Create two columns
    
    with col1:  # This will go into the left column
        st.header('QSAR-Ready Structure')
        image = plot.render_as_image(data_manager.qsar_ready_smiles)
        st.image(image)

    with col2:  # This will go into the right column
        if effect_label:
            st.header('Features from OPERA 2.9')
            st.write(data_manager.features)

if predict_button:
    st.header('POD Estimates [mg/(kg-d)]')
    st.write(data_manager.point_of_departure)