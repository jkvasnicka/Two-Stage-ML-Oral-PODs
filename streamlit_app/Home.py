'''
'''

import streamlit as st

st.set_page_config(
    page_title='Home',
    layout='wide'
)

st.title(
    'ML2 = POD: An Interactive Tool for Hazard and Risk Evaluation'
)

st.markdown(
    '''
    ### Background
    Determining a chemical's point of departure (POD) for a critical health 
    effect is crucial in evaluating and managing health risks from chemical 
    exposure. However, a lack of in vivo toxicity data for most chemicals in 
    commerce represents a major challenge for risk assessors and managers. To 
    address this issue, we developed a novel two-stage machine learning (ML) 
    framework for predicting protective human equivalent non-cancer PODs for 
    oral exposure based only on chemical structure:. Utilizing ML-based 
    predictions for physical/chemical/toxicological properties from OPERA 2.9 
    as features (Stage 1),  ML models using random forest regression were 
    trained using in vivo datasets of organic chemicals for general noncancer 
    effects (n = 1,790) and reproductive/developmental effects (n = 2,226), 
    with robust cross-validation employed for feature selection and estimating 
    prediction errors (Stage 2). These “ML2” models accurately and precisely 
    predicted PODs for both effect categories with errors less than an order 
    of magnitude. We then applied these models to 450,644 chemicals and 
    compared predicted PODs with high-throughput exposure estimates from the 
    SEEM3 model to prioritize chemicals' potential concern based on their 
    margins of exposure. 

    ### Purpose of this App
    This app provides a user-friendly interface for leveraging the ML2 models 
    for predicting protective human equivalent non-cancer PODs for oral 
    exposure based on chemical structure.

    ### App Overview
    - **Home**: Introduction to the app and the research behind it.
    - **Data Visualization**: Interactive visualizations of the data and model 
        predictions for a chemical of interest.
    - **Data Download**: Download the datasets used by the app.

    ### Suggested Citation
    Please cite our work as follows if you use this app for your research:
    
        Kvasnicka, J.; Aurisano, N.; Lu, E.-H.; Fantke, P.; Jolliet, O. O.; 
        Wright, F. A.; Chiu, W. A. ML2 = POD: A Two-Stage Machine Learning-Based 
        Approach for Predicting Human Health Protective Points of Departure. 
        Environ. Sci. Technol. 2023, Vol. XX, No. XX, pp. XXX-XXX.

    ### Contact
    For any inquiries, please reach out to 
    [Prof. Weihsueh A. Chiu](mailto:wchiu@cvm.tamu.edu), Department of 
    Veterinary Physiology and Pharmacology, Interdisciplinary Faculty of 
    Toxicology, Texas A&M University, College Station, Texas, United States 
    '''
)
