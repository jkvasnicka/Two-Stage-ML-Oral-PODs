'''
'''

import streamlit as st

st.set_page_config(
    page_title='ML2 = POD',
    layout='wide'
)

# Create a title with superscript "2" indicated "squared"
st.markdown(
    '''
    <h1 style="font-size: 36px;">
        ML<sup style="font-size: 18px; vertical-align: super;">2</sup> = POD: 
        A Two-Stage Machine Learning-Based Approach for Predicting Human 
        Health Protective Points of Departure
    </h1>
    ''', 
    unsafe_allow_html=True
)

st.markdown(
    '''
    ## Background
    Determining a chemical's point of departure (POD) for a critical health 
    effect is crucial in evaluating and managing health risks from chemical 
    exposure. However, a lack of in vivo toxicity data for most chemicals in 
    commerce represents a major challenge for risk assessors and managers. To 
    address this issue, we developed a novel two-stage machine learning (ML) 
    framework for predicting protective human equivalent non-cancer PODs for 
    oral exposure based only on chemical structure: Utilizing ML-based 
    predictions for physical/chemical/toxicological properties from OPERA 2.9 
    as features (Stage 1), ML models using random forest regression were 
    trained using in vivo datasets of organic chemicals for general noncancer 
    effects (n = 1,790) and reproductive/developmental effects (n = 2,226), 
    with robust cross-validation employed for feature selection and estimating 
    prediction errors (Stage 2).

    ## Purpose of this App
    This app was designed to provide a user-friendly tool for predicting 
    protective human equivalent non-cancer PODs for oral exposure based on 
    chemical structure. Users can input specified chemicals and visualize 
    the results to aid in evaluating chemical safety.

    **Suggested Citation:**
    Kvasnicka, J.; Aurisano, N.; Lu, E.-H.; Fantke, P.; Jolliet, O. O.; 
    Wright, F. A.; Chiu, W. A. ML2 = POD: A Two-Stage Machine Learning-Based 
    Approach for Predicting Human Health Protective Points of Departure. 
    Environ. Sci. Technol. 2023, Vol. XX, No. XX, pp. XXX-XXX.

    **Contact:** 
    [Prof. Weihsueh A. Chiu](mailto:wchiu@cvm.tamu.edu), Department of 
    Veterinary Physiology and Pharmacology, Interdisciplinary Faculty of 
    Toxicology, Texas A&M University, College Station, Texas, United States 
    '''
)
