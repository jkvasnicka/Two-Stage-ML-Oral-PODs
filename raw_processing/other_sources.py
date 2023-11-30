'''This module contains various functions for loading/parsing raw data.
'''

import pandas as pd
import numpy as np
import re 

from . import pattern

#region: surrogate_toxicity_values_from_excel
def surrogate_toxicity_values_from_excel(
        tox_data_path, 
        tox_metric, 
        index_col, 
        tox_data_kwargs,
        log10=False,
        study_count_thres=3, 
        effect_mapper=None, 
        write_path=None
        ):
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
        tox_data_kwargs
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
        tox_data_path, 
        tox_metric, 
        index_col, 
        tox_data_kwargs
        ):
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
    tox_data = (
        pd.read_excel(tox_data_path, **tox_data_kwargs)
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

#region: regulatory_toxicity_values_from_excel
def regulatory_toxicity_values_from_excel(
        fig_s5_path, 
        reg_data_kwargs,
        ilocs_for_effect, 
        id_for_casrn=None, 
        id_name=None, 
        write_path=None
        ):
    '''Load and parse the regulatory toxicity values from CSV file.

    Parameters
    ----------

    Returns
    -------

    Reference
    ---------
    '''
    fig_s5_data = (
        pd.read_excel(fig_s5_path, **reg_data_kwargs)
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

    if id_for_casrn:
        reg_pods = _replace_casrn_index(reg_pods, id_for_casrn, id_name)
        
    if write_path is not None:
        reg_pods.to_csv(write_path)
        
    return reg_pods
#endregion

#region: experimental_ld50s_from_excel
def experimental_ld50s_from_excel(
        ld50s_path, 
        ld50_exp_column,
        id_for_casrn=None,
        id_name=None,
        inverse_transform=True,
        write_path=None
    ):
    '''Load and parse the experimental LD50 values from an Excel file.

    Parameters
    ----------
    ld50s_path : str
        File path. 
    ld50_exp_column : str
        Name of column corresponding to LD50 data to extract.
    id_for_casrn : dict, optional
        A dictionary mapping from CASRN to a new identifier.
    id_name : str, optional
        The name to be assigned to the new index.
    inverse_transform : bool (optional)
        If True, apply an inverse log10-transformation to the values.
    Write_path : str (optional)
        Path to write the return as a CSV file.

    Returns
    -------
    pandas.Series

    Notes
    -----
    These data were extracted from ToxValdDB and curated by Nicolo Aurisano. 
    All data were extrapolated to humans. The data represent acute studies.
    '''
    # Get the LD50 values in log10-units
    ld50s = pd.read_excel(ld50s_path, index_col='casrn')[ld50_exp_column]

    if id_for_casrn:
        ld50s = _replace_casrn_index(ld50s, id_for_casrn, id_name)

    if inverse_transform:
        ld50s = 10**ld50s

    if write_path is not None:
        ld50s.to_csv(write_path)

    return ld50s
#endregion

#region: _replace_casrn_index
def _replace_casrn_index(data, id_for_casrn, id_name):
    '''
    Replace the index of a DataFrame, originally based on CASRN, with a new 
    index.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame whose index is to be replaced. The current index should 
        be CASRN.
    id_for_casrn : dict
        A dictionary mapping from CASRN to a new identifier.
    id_name : str
        The name to be assigned to the new index.

    Returns
    -------
    pd.DataFrame
        The DataFrame with its index replaced by the new identifiers.

    Note
    ----
    Rows in `data` whose CASRN is not found in `id_for_casrn` will be excluded
    from the returned DataFrame.
    '''
    casrn_intersection = list(data.index.intersection(set(id_for_casrn)))
    data = data.loc[casrn_intersection]
    data.index = [id_for_casrn[casrn] for casrn in data.index]
    data.index.name = id_name    
    return data
#endregion

#region: seem3_exposure_data_from_excel
def seem3_exposure_data_from_excel(
        exposure_file, 
        exposure_data_kwargs,
        index_col,
        log10_transform=True,
        write_path=None
        ):
    '''
    Extract and process SEEM3 exposure data from raw data.

    Parameters
    ----------
    exposure_file : str
        Path to the Excel file containing the exposure data.
    exposure_data_kwargs : dict
        Key-word arguments for pandas.read_excel().
    index_col : str
        Column name to be used as the index.
    log10_transform : bool, optional
        If True, the data will be log10-transformed. Default True.
    write_path : str, optional
        Path to the output Parquet file. If present, the data will be written 
        to disk.
        
    Returns
    -------
    pandas.DataFrame
    '''
    exposure_data = (
        pd.read_excel(
            exposure_file,
            **exposure_data_kwargs
        )
        .set_index(index_col)
    )

    if log10_transform:
        exposure_data = np.log10(exposure_data)

    if write_path:
        exposure_data.to_parquet(write_path, compression='gzip')

    return exposure_data
#endregion