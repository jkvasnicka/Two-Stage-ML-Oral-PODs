'''This module contains various functions for loading/parsing raw data.
'''

import pandas as pd
import numpy as np
import re 

from . import pattern

#region: surrogate_toxicity_values_from_excel
def surrogate_toxicity_values_from_excel(
        tox_data_path, sheet_name, tox_metric, index_col, log10=False,
        study_count_thres=3, chemicals_to_exclude=None, effect_mapper=None, 
        write_path=None):
    '''Load the surrogate toxicity values.

    Parameters
    ----------
    tox_data_path : str
        File path to the raw data.
    tox_metric : str
        Name of the target variable in the headers.
    index_col : str
        Name of the chemical identifier in the headers to use as index.
    log10 : bool (optional)
        If True, apply a log10 transformation to the toxicity values.
    study_count_thres : int (optional)
        Chemicals with study counts exceeding this value will be retained.
    chemicals_to_exclude : list of str
        Names of chemicals to drop, which may or may not be present.
    effect_mapper : dict (optional)
        Mapping each original effect type (string) --> preferred string for
        the return.

    Returns
    -------
    dict of pandas.DataFrame for each effect type

    Notes
    -----
    The data were obtained from Table S5 of Aurisano et al. (2023).

    References
    ----------
    https://doi.org/10.1289/EHP11524
    '''
    tox_data = toxicity_data_and_study_counts_from_excel(
        tox_data_path, 
        tox_metric, 
        index_col, 
        sheet_name=sheet_name, 
        header=[0, 1],
        skiprows=[0]
        )
    
    if chemicals_to_exclude is not None:
        tox_data = tox_data.drop(
            chemicals_to_exclude, 
            errors='ignore'
            )
    
    # Initialize a container.
    parsed_data_for_effect = {}

    ## Split the DataFrame by effect type.
    for effect in tox_data.columns.unique(level=0):
        study_counts = tox_data[effect]['count'] 
        effect_data = tox_data[effect][tox_metric]

        if study_count_thres is not None:
            # Apply the study count filter.
            where_above_thres = study_counts > study_count_thres
            effect_data = effect_data.loc[where_above_thres]

        ## Perform string formatting.
        if effect_mapper is not None:
            # Use the preferred key for the return.
            effect = effect_mapper[effect]

        if log10 is True:
            effect_data = np.log10(effect_data)
            effect_data.name = 'log10_' + effect_data.name

        parsed_data_for_effect[effect] = effect_data
    parsed_data_for_effect = pd.DataFrame(parsed_data_for_effect)
    
    if write_path is not None:
        parsed_data_for_effect.to_csv(write_path)

    return parsed_data_for_effect
#endregion

#region: toxicity_data_and_study_counts_from_excel
def toxicity_data_and_study_counts_from_excel(
        tox_data_path, tox_metric, index_col, **kwargs):
    '''Helper function to load the toxicity data and study counts from the
    source Excel file.

    Returns
    -------
    pandas.DataFrame with MultiIndex columns
        Index = chemicals, columns = (effect_type, variable) where variable
        includes the target variable and study count.

    See Also
    --------
    filter_toxicity_data()
    '''
    # TODO: Move to helper function and use in processor.py
    tox_data = (
        pd.read_excel(tox_data_path, **kwargs)
        .swaplevel(axis=1)
        .set_index(index_col)
        [[tox_metric, 'count']]
        .swaplevel(axis=1)
        )
    # Remove the unnecessary second level.
    tox_data.index = [tup[0] for tup in tox_data.index]
    # Convert to uppercase for a consistent return.
    tox_data.index.name = index_col.upper()

    ## Filter out chemicals with missing identifiers/index.
    # Use regex to extract identifiers from the raw strings.
    chem_id_pattern_str = pattern.__dict__[index_col](as_group=True)
    pat = re.compile(chem_id_pattern_str)
    tox_data.index = tox_data.index.str.extract(pat, expand=False)
    where_not_missing = tox_data.index.notna()
    return tox_data.loc[where_not_missing]
#endregion

#region: regulatory_toxicity_values_from_csv
def regulatory_toxicity_values_from_csv(
        fig_s5_path, ilocs_for_effect, chem_id_for_casrn=None, 
        new_chem_id=None, write_path=None):
    '''Load and parse the regulatory toxicity values from CSV file.

    Parameters
    ----------

    Returns
    -------

    Reference
    ---------
    '''
    fig_s5_data = (
        pd.read_csv(fig_s5_path, skiprows=[0], header=[0, 1])
        .droplevel(0, axis=1)  # allows duplicate column names
    )

    # Initialize a container.
    reg_pods = {}
    for effect, ilocs in ilocs_for_effect.items():
        reg_pods[effect] = (
            fig_s5_data.iloc[:, ilocs]
            .drop_duplicates(subset='casrn')
            .set_index('casrn')
            .squeeze()
        )
    reg_pods = pd.DataFrame(reg_pods).dropna(how='all')
    reg_pods.index.name = reg_pods.index.name.upper()  # for consistency

    if chem_id_for_casrn is not None:
        # Replace the CASRN index with the new chemical identifier.
        chem_id_for_casrn = pd.Series(chem_id_for_casrn, name=new_chem_id)
        reg_pods = (
            reg_pods.join(chem_id_for_casrn, how='inner')
            .set_index(new_chem_id)
            )
        
    if write_path is not None:
        reg_pods.to_csv(write_path)
        
    return reg_pods
#endregion

#region: experimental_ld50s_from_excel
def experimental_ld50s_from_excel(
        ld50s_path, chem_identifiers, index_col, log10=False, 
        ld50_columns=None, study_count_thres=None, write_path=None):
    '''Load and parse the experimental LD50 values from an Excel file.

    Parameters
    ----------
    ld50s_path : str
        File path. 
    chem_identifiers : pandas.DataFrame
        Used to map CASRNs to the chemical identifier of interest.
    index_col : str
        Name of the chemical identifier in the headers of chem_identifiers to 
        use as the index.
    log10 : bool (optional)
        If False, apply an inverse log10-transformation to the values.
    ld50_columns : list of str (optional)
        Names of columns corresponding to LD50 statistics to extract.
    study_count_thres : int (optional)
        Chemicals with study counts exceeding this value will be retained.
    Write_path : str (optional)
        Path to write the return as a CSV file.

    Returns
    -------
    pandas.DataFrame
    '''
    ld50s = pd.read_excel(ld50s_path)

    if study_count_thres is not None:
        # Apply the filter.
        ld50s = ld50s.loc[ld50s['count_LD50'] > study_count_thres]
    ld50s = ld50s.drop('count_LD50', axis=1)

    if ld50_columns is None:
        # Use all LD50 columns in the original file.
        ld50_columns = [c for c in ld50s if 'LD50' in c]

    # Use the CASRN to get the specified identifier as the index.
    ld50s = (
        ld50s.merge(
            chem_identifiers.reset_index(), 
            left_on='casrn',
            right_on='CASRN')
        .set_index(index_col)
    )

    ld50s = ld50s[ld50_columns]

    if log10 is False:
        # Apply inverse transformation to get the original scale.
        ld50s = 10**ld50s

    if write_path is not None:
        ld50s.to_csv(write_path)

    return ld50s
#endregion