'''
'''

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

from . import utilities

#region: proportions_incomplete_subplots
def proportions_incomplete_subplots(
        features_file, 
        AD_file, 
        targets_file, 
        plot_settings, 
        base_size_per_feature=(0.2, 6),
        threshold=None
    ):
    '''

    Parameters
    ----------
    threshold : float, optional
        Expressed as a proportion between 0., 1. Features with a proportion of 
        missing values at or above this treshold would have been dropped from 
        the Pipeline.
    '''
    X = pd.read_csv(features_file, index_col=0)
    AD_flags = pd.read_csv(AD_file, index_col=0)
    ys = pd.read_csv(targets_file, index_col=0)

    ## Plot the training set data.
    samples_for_effect = {
        plot_settings.label_for_effect[effect] : y.dropna().index 
        for effect, y in ys.items()
        }
    proportions_incomplete_subplot(
        X, 
        AD_flags, 
        samples_for_effect,
        base_size_per_feature=base_size_per_feature,
        threshold=threshold
    )

    ## Plot all chemicals data.
    proportions_incomplete_subplot(
        X, 
        AD_flags, 
        {plot_settings.all_chemicals_label : X.index},
        base_size_per_feature=base_size_per_feature,
        threshold=threshold
    )
#endregion

#region: proportions_incomplete_subplot
def proportions_incomplete_subplot(
        X, 
        AD_flags, 
        samples_dict, 
        base_size_per_feature=(0.2, 6),
        threshold=None
    ):
    '''
    Generate a subplot for each group of samples in the provided dictionary.

    Parameters
    ----------
    X : pandas.DataFrame
        The complete dataset.
    AD_flags : pandas.Series
        The complete AD flags.
    samples_dict : dict
        Dictionary of sample groups, where the key is the group name and the 
        value is a list of sample IDs.
    base_size_per_feature : tuple
        The base size of the plot for a single feature.

    Returns
    -------
    tuple
        Tuple containing the generated figure and a list of Axes.
    '''
    n = len(samples_dict)
    n_features = X.shape[1]
    figsize = (base_size_per_feature[1]*n, n_features*base_size_per_feature[0])
    
    fig, axs = plt.subplots(
        ncols=n, 
        figsize=figsize, 
        sharex=False, 
        sharey=False,
        constrained_layout=True
        )

    if n == 1:
        axs = [axs]  # Wrap the single Axes object in a list

    titles = []
    for i, (title, sample_ID_set) in enumerate(samples_dict.items()):
        titles.append(title)

        sample_IDs = list((sample_ID_set).intersection(X.index))
        X_subset = X.loc[sample_IDs]
        AD_flags_subset = AD_flags.loc[sample_IDs]

        missing_prop, outside_AD_prop, valid_prop = proportions_incomplete(
            X_subset, AD_flags_subset)
        
        # Reorder the valid_prop, missing_prop, and outside_AD_prop
        valid_prop = valid_prop.sort_values(ascending=True)
        missing_prop = missing_prop.reindex(valid_prop.index)
        outside_AD_prop = outside_AD_prop.reindex(valid_prop.index)

        n_samples = len(sample_IDs)
        proportions_incomplete_barchart(
            axs[i], 
            missing_prop, 
            outside_AD_prop, 
            valid_prop, 
            title, 
            n_samples,
            threshold=threshold
            )

        # Remove ylabel for all but the first subplot
        if i != 0:
            axs[i].set_ylabel('')
            
    fig.tight_layout()

    utilities.save_figure(
        fig, 
        proportions_incomplete_subplot, 
        list(titles)
        )
#endregion

#region: proportions_incomplete_barchart
def proportions_incomplete_barchart(
        ax, 
        missing_prop, 
        outside_AD_prop, 
        valid_prop, 
        title, 
        n_samples,
        threshold=None,
        ):
    '''
    Plot the proportions of missing values, values outside AD, and valid 
    values on a given Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object on which the proportions will be plotted.
    missing_prop : pandas.Series
        The proportion of missing values.
    outside_AD_prop : pandas.Series
        The proportion of values outside AD.
    valid_prop : pandas.Series
        The proportion of valid values.
    title : str
        Title of the plot.
    n_samples : int
        Total number of samples.

    Returns
    -------
    None
    '''
    y = np.arange(len(valid_prop))

    ax.barh(
        y, 
        outside_AD_prop, 
        left=missing_prop, 
        height=0.8, 
        label='Outside AD', 
        color='orange', 
        edgecolor='black', zorder=3
        )
    ax.barh(
        y, 
        missing_prop, 
        height=0.8, 
        label='Missing Data (Inside AD)', 
        color='blue', 
        edgecolor='black', 
        zorder=3
        )
    if threshold:
        # Convert to percent to align with the x-axis
        threshold = 100. * threshold
        # Add vertical line on top
        ax.axvline(threshold, linestyle='-.', zorder=5)
    
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f'{index} ({int(value)}%)' 
         for index, value in valid_prop.items()], 
         fontsize=8)
    ax.set_xlabel('% Incomplete', size='medium')
    ax.set_ylabel('Feature Names (% Complete)', size='medium')
    n_samples = utilities.comma_separated(n_samples)
    ax.set_title(f'{title} (n={n_samples})', size='medium', loc='left')
    ax.set_xlim([0, 100])
    ax.grid(True, axis='x', linestyle='--', color='black', alpha=0.6)

    if any(missing_prop >= 1): # %
        # Some features have an appreciable % of values that are missing yet 
        # within the AD. Highlight these values using a legend.
        ax.legend(fontsize='small', loc='upper right')
#endregion

#region: proportions_incomplete
def proportions_incomplete(X_subset, AD_flags_subset):
    '''
    Calculate the proportion of missing values, values outside of 
    applicability domain (AD), and valid values.

    Parameters
    ----------
    X_subset : pandas.DataFrame
        Subset of the data.
    AD_flags_subset : pandas.Series
        Subset of the AD flags.

    Returns
    -------
    tuple
        Tuple containing the proportion of missing values, values outside AD, 
        and valid values.
    '''
    # Filter out any features that don't have applicability domains
    complete_features = X_subset.columns.difference(AD_flags_subset.columns)
    X_subset = X_subset.drop(complete_features, axis=1)

    # TODO: Rename "prop" to "props" for clarity
    outside_AD_prop = AD_flags_subset.mean() * 100
    not_outside_AD = ~AD_flags_subset
    missing_prop = (X_subset.isna() & not_outside_AD).mean() * 100
    valid_prop = 100 - missing_prop - outside_AD_prop
    
    # Order by the valid proportion in ascending order
    order = valid_prop.sort_values(ascending=False).index
    sorted_proportions = (
        missing_prop.loc[order], 
        outside_AD_prop.loc[order], 
        valid_prop.loc[order]
    )
    return sorted_proportions
#endregion