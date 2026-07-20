'''
Plotting module for visualizing the distributions of features as histograms.

See Also
--------
plot.py
    The main plotting module where this sub-module is implemented as part of 
    the main package.
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from . import utilities

#region: feature_distributions
def feature_distributions(
        features_file,
        color='cornflowerblue',
        output_dir=None
        ):
    '''
    Visualize the distributions of features as histograms.

    Paramters
    ---------
    features_file : str
        Path to the features file. 
    color : str, optional
        Color for the histogram bars.

    Returns
    -------
    None
        The figures are saved to a dedicated directory derived from the 
        function name.
    '''
    X = pd.read_parquet(features_file)

    discrete_features = [name for name in X if 'discrete' in name]
    continuous_features = list(X.columns.difference(set(discrete_features)))

    # Determine the number of columns and calculate the number of rows
    ncols = 5
    n_features = len(continuous_features) + len(discrete_features)
    nrows = (n_features + ncols - 1) // ncols

    # Determine the size of each subplot and calculate the total figure size
    subplot_width = 2.5 
    subplot_height = 2  
    figsize = (ncols * subplot_width, nrows * subplot_height)

    # Set up the subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Plot continuous features
    for i, feature in enumerate(continuous_features):
        row = i // 5
        col = i % 5
        # Use the square-root rule
        number_of_bins = int(np.sqrt(len(X[feature].dropna())))
        axs[row, col].hist(
            X[feature].dropna(), 
            color=color, 
            bins=number_of_bins, 
            edgecolor=color, 
            linewidth=0.5
        )
        axs[row, col].set_title(feature)

    # Plot discrete features
    for i, feature in enumerate(discrete_features):
        non_null_values = X[feature].dropna()
        unique_values = non_null_values.unique()
        # Check if there are any non-null values
        if len(unique_values) > 0:
            row = (i + len(continuous_features)) // 5
            col = (i + len(continuous_features)) % 5
            value_counts = non_null_values.value_counts().sort_index()
            axs[row, col].bar(
                value_counts.index, 
                value_counts.values, 
                color=color, 
                edgecolor=color
            )
            axs[row, col].set_title(feature)
            
            if len(unique_values) <= 2:
                axs[row, col].set_xticks(unique_values)
                integer_labels = [int(value) for value in unique_values]
                axs[row, col].set_xticklabels(integer_labels)

    # Remove unused subplots
    total_features = len(continuous_features) + len(discrete_features)
    for i in range(total_features, nrows * ncols):
        axs.flatten()[i].axis('off')

    fig.tight_layout()

    utilities.save_figure(
        fig, 
        feature_distributions, 
        'feature-distributions',
        output_dir=output_dir
        )
#endregion
