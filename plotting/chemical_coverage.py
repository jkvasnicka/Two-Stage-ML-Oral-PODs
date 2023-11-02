'''
'''

import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np

from . import utilities

#region: pairwise_scatters_and_kde_subplots
def pairwise_scatters_and_kde_subplots(
        features_file, 
        targets_file, 
        plot_settings, 
        figsize=(10, 10)
        ):
    '''
    Create a grid of scatter and KDE plots for combinations of features.

    Parameters
    ----------
    '''
    # TODO: Move to plot setting?
    features_subset = [  
        'CATMoS_LD50_pred',
        'P_pred',
        'TopoPolSurfAir',
        'MolWeight'
    ]
    label_for_category = {
        False : plot_settings.label_for_sample_type['out'],
        True :  plot_settings.label_for_sample_type['in']
    }
    color_for_category = {
        False : plot_settings.color_for_sample_type['out'], 
        True : plot_settings.color_for_sample_type['in']
    }
    marker_size_for_category = {
        False : plot_settings.marker_size_for_sample_type['out'],
        True : plot_settings.marker_size_for_sample_type['in']
    }

    X = pd.read_csv(features_file, index_col=0)
    X = X[features_subset]

    # Log10-transform desired features
    X = _log10_transform_feature(X, 'P_pred')
    X = _log10_transform_feature(X, 'CATMoS_LD50_pred')

    buffer_fraction = 0.05  # 5% buffer
    limits_for_feature = {}
    for feature in X.columns:
        min_val = X[feature].min()
        max_val = X[feature].max()
        buffer = (max_val - min_val) * buffer_fraction
        limits_for_feature[feature] = (min_val - buffer, max_val + buffer)

    y = pd.read_csv(targets_file, index_col=0).squeeze()
    chemical_union = list(y.index)  # across all effect types
    categories = X.index.isin(chemical_union)
    
    ## Plot the data

    fig, axs = plot_pairwise_scatters_and_kde(
        X, 
        categories, 
        color_for_category, 
        marker_size_for_category,
        figsize=figsize,
        limits_for_feature=limits_for_feature
        )

    handles = [
        plt.Line2D([0], [0], marker='o', color=color, linestyle='') 
        for color in color_for_category.values()
    ]
    labels = [label_for_category[cat] for cat in color_for_category]
    _ = fig.legend(
        handles, 
        labels, 
        fontsize='small', 
        ncol=len(labels),
        bbox_to_anchor=(1., 0.)
    )

    utilities.save_figure(
        fig, 
        pairwise_scatters_and_kde_subplots, 
        'all-opera-features-and-target-union',
        bbox_inches='tight'
        )
#endregion

#region: plot_pairwise_scatters_and_kde
def plot_pairwise_scatters_and_kde(
        X, 
        categories, 
        color_for_category, 
        marker_size_for_category,
        figsize=None,
        limits_for_feature=None
        ):
    '''
    Create a grid of scatter and KDE plots.

    Parameters
    ----------
    X : pandas.DataFrame
        Data for the features.
    categories : pandas.Series
        Categorical variable.
    color_for_category : dict
        Mapping of category to color.
    '''
    fig, axs = plt.subplots(
        len(X.columns), 
        len(X.columns), 
        figsize=figsize
    )
    for row, feature_row in enumerate(X.columns):
        for col, feature_col in enumerate(X.columns):
            ax = axs[row, col]
            if row == col:
                plot_kde(
                    ax, 
                    X, 
                    categories, 
                    feature_row, 
                    color_for_category,
                    limits_for_feature
                    )
            elif row > col:
                plot_scatter(
                    ax, 
                    X, 
                    categories, 
                    feature_col, 
                    feature_row, 
                    color_for_category,
                    marker_size_for_category,
                    limits_for_feature
                    )
            else:
                ax.set_visible(False)
                
            # Set x-label and y-label
            if row == len(X.columns) - 1:
                ax.set_xlabel(feature_col)
            if col == 0:
                ax.set_ylabel(feature_row)
                
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.07)

    return fig, axs
#endregion

#region: plot_kde
def plot_kde(
        ax, 
        X, 
        categories, 
        feature, 
        color_for_category,
        limits_for_feature
        ):
    '''
    Plot KDE on diagonal.

    Parameters
    ----------
    ax : matplotlib.axes
        Axis to plot on.
    X : pandas.DataFrame
        Data for the features.
    categories : pandas.Series
        Categorical variable.
    feature : str
        Feature name.
    color_for_category : dict
        Mapping of category to color.
    '''
    for cat in list(color_for_category):
        subset = X[categories == cat]
        sns.kdeplot(subset[feature], ax=ax, color=color_for_category[cat])
    ax.set_xlim(limits_for_feature[feature])  # Set consistent x limits
    ax.set_xlabel('')  
    ax.set_ylabel('')
#endregion

#region: plot_scatter
def plot_scatter(
        ax, 
        X, 
        categories, 
        feature_x, 
        feature_y, 
        color_for_category,
        marker_size_for_category,
        limits_for_feature
        ):
    '''
    Plot scatter plot in the lower triangle.

    Parameters
    ----------
    ax : matplotlib.axes
        Axis to plot on.
    X : pandas.DataFrame
        Data for the features.
    categories : pandas.Series
        Categorical variable.
    feature_x : str
        Feature name for the x-axis.
    feature_y : str
        Feature name for the y-axis.
    color_for_category : dict
        Mapping of category to color.
    '''
    for cat in list(color_for_category):
        subset = X[categories == cat]
        ax.scatter(
            subset[feature_x], 
            subset[feature_y], 
            c=color_for_category[cat], 
            s=marker_size_for_category[cat],
            label=cat
            )
    ax.set_xlim(limits_for_feature[feature_x])  # Set consistent x limits
    ax.set_ylim(limits_for_feature[feature_y])  # Set consistent y limits
#endregion

#region: _log10_transform_feature
def _log10_transform_feature(X, feature_name):
    '''
    Log10-transforms the given feature in the dataframe X.

    Parameters
    ----------
    X : pd.DataFrame
        The dataframe containing the feature to be transformed.
    feature_name : str
        The name of the feature to be transformed.

    Returns
    -------
    X : pd.DataFrame
        The dataframe with the transformed feature.
    '''
    new_name = '$log_{10}$' + feature_name
    X = X.rename({feature_name: new_name}, axis=1)
    X[new_name] = np.log10(X[new_name])
    return X
#endregion