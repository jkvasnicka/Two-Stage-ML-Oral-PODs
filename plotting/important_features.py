'''
Plotting module for visualizing counts of important features across replicate
models from cross-validation.

See Also
--------
plot.py
    The main plotting module where this sub-module is implemented as part of 
    the main package.
'''

import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import numpy as np

from . import utilities

#region: important_feature_counts
def important_feature_counts(
        results_analyzer, 
        plot_settings
        ):
    '''
    Visualize the counts of important features across replicate models from 
    cross-validation.

    This function plots horizontal bar graphs where each bar represents a 
    feature and its count across all replicate models. The features included 
    in the final model are highlighted in a different color.

    Parameters
    ----------
    results_analyzer : results_analysis.ResultsAnalyzer
        Manages the analysis of results.
    plot_settings : SimpleNamespace
        Configuration settings for plotting.

    Returns
    -------
    None
        The figures are saved to a dedicated directory derived from the 
        function name.
    '''
    # Define colorblind-friendly colors
    color_filled = '#1f77b4'  # Blue
    color_unfilled = '#dcdcdc'  # Light gray

    model_key_names = results_analyzer.read_model_key_names()
    grouped_keys = results_analyzer.group_model_keys(
        'target_effect', 
        string_to_exclude='false'
    )

    for grouping_key, model_keys in grouped_keys:
        n_keys = len(model_keys)

        # Figure layout is adjusted to have one column per model key
        fig, axs = plt.subplots(1, n_keys, figsize=(5*n_keys, 8), sharey=True)

        sorted_features = None  # initialize

        for i, model_key in enumerate(model_keys):
            ## Prepare the data for plotting.
            features_for_final_model = results_analyzer.get_important_features(model_key)
            features_for_replicate_model = (
                results_analyzer.get_important_features_replicates(model_key))

            # Initialize feature_counts dictionary
            key_for = dict(zip(model_key_names, model_key))
            all_feature_names = list(results_analyzer.load_features(**key_for))
            feature_counts = {feature: 0 for feature in all_feature_names}

            # Count the occurrences of each feature
            for feature_list in features_for_replicate_model.values():
                for feature in feature_list:
                    if feature in feature_counts:
                        feature_counts[feature] += 1

            if sorted_features is None:
                # Sort features based on their counts only for the left model.
                sorted_features = sorted(
                    all_feature_names, 
                    key=lambda f: feature_counts[f], 
                    reverse=True
                )

            final_model_features = set(features_for_final_model)
            bar_positions = np.arange(len(all_feature_names))
            bar_counts = [feature_counts[feature] for feature in sorted_features]
            bar_colors = [color_filled if feature in final_model_features 
                          else color_unfilled for feature in sorted_features]

            axs[i].barh(
                bar_positions, bar_counts, color=bar_colors, edgecolor='black', 
                linewidth=1)
            axs[i].set_ylim(-1, len(all_feature_names))
            axs[i].set_yticks(range(len(all_feature_names)))
            axs[i].set_yticklabels(sorted_features, fontsize=10)
            axs[i].tick_params(axis='y', pad=0)
            axs[i].invert_yaxis()
            axs[i].set_xlabel('Count', fontsize=12)
            axs[i].set_ylabel(plot_settings.feature_names_label, fontsize=12)
            axs[i].set_title(
                plot_settings.label_for_effect[key_for['target_effect']], fontsize=12)

            if i != 0:  # Only set ylabel for first subplot
                axs[i].set_ylabel('')

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.125)  # accomodate the legend

        # Set a single legend.
        legend_patches = [
            mpatches.Patch(color=color_filled, label='In Final Model'),
            mpatches.Patch(color=color_unfilled, label='Not in Final Model')
        ]
        fig.legend(
            handles=legend_patches, 
            loc='lower right', 
            fontsize='small',
            ncol=len(legend_patches), 
            bbox_to_anchor=(1., -0.01)
        )

        utilities.save_figure(
            fig, 
            important_feature_counts, 
            grouping_key
            )
#endregion