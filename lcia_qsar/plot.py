'''Support for plotting specifically for the LCIA QSAR Project.
'''

import os
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
import itertools

# Enable modules to be imported from the parent directory.
import sys
sys.path.append('..')
from common import plot 

_prediction_label = '$log_{10}POD$'  # LaTeX rendering for subscript

#region: performance_and_prediction_comparison
def performance_and_prediction_comparison(
        workflow, 
        label_for_metric, 
        label_for_model_build,
        label_for_scoring,
        ylim=(0., 1.)
        ):
    '''
    Generate and plot categorical performances and predictions.
    
    Parameters
    ----------
    workflow : Workflow
        The workflow object containing the data.
    ylim : tuple, optional
        A tuple specifying the y-axis limits. Default is (0., 1.).

    Returns
    -------
    dict
        A dictionary of Figure objects, where the keys are unique combinations
        of levels from the workflow data.
    '''
    model_key_names = get_model_key_names(workflow)
    combination_key_groups = get_model_key_groups(
        workflow, model_key_names, 'model_build')

    for combination_key, group in combination_key_groups:
        model_keys = list(group)

        # Initialize a Figure for the subplot.
        fig = plt.figure()

        gs1 = gridspec.GridSpec(3, 1)
        gs2 = gridspec.GridSpec(2, 1)

        _plot_performances_boxplots_left_half(
            fig, 
            gs1, 
            workflow, 
            model_keys, 
            label_for_metric, 
            label_for_model_build,
            ylim=ylim
            )

        _plot_prediction_scatterplots_right_half(
            fig, 
            gs2, 
            workflow, 
            model_keys, 
            label_for_metric,
            label_for_model_build,
            label_for_scoring
            )

        gs1.tight_layout(fig, rect=[0, 0, 0.5, 1])
        gs2.tight_layout(fig, rect=[0.5, 0, 1, 1], h_pad=0.5)


        save_figure(
            fig, 
            performance_and_prediction_comparison, 
            combination_key
            )
#endregion

#region: _plot_performances_boxplots_left_half
def _plot_performances_boxplots_left_half(
        fig, 
        gs1, 
        workflow, 
        model_keys, 
        label_for_metric, 
        label_for_model_build,
        ylim=(0., 1.)
        ):
    '''
    Plot a categorical combination using a boxplot on the left half of a figure.
    
    Parameters
    ----------
    fig : Figure
        The figure on which to add the subplots.
    gs1 : GridSpec
        The GridSpec object used to specify the subplot layout.
    workflow : Workflow
        The workflow object containing the data.
    model_keys : list
        A list of model keys used to filter and format the data.
    ylim : tuple, optional
        A tuple specifying the y-axis limits. Default is (0., 1.).
    '''
    ## Prepare the data.
    performances_wide = workflow.concatenate_history('performances')
    # Filter the data.
    where_subset_metrics = (
        performances_wide.columns
        .get_level_values('metric')
        .isin(label_for_metric)
    )
    performances_wide = performances_wide.loc[:, where_subset_metrics]
    # Rename the columns for visualization.
    label_for_column = {
        **label_for_metric, 
        **label_for_model_build
    }
    performances_wide = performances_wide.rename(label_for_column, axis=1)

    ## Plot the data. 

    metrics = performances_wide.columns.unique(level='metric')

    # The new keys will be used to get the data.
    model_keys_renamed = [
        tuple(label_for_column.get(k, k) for k in model_key) 
        for model_key in model_keys
        ]

    for i, metric in enumerate(metrics):

        ax = fig.add_subplot(gs1[i])                

        # Filter and format the data.
        metric_data_long = (
            performances_wide.xs(metric, axis=1, level='metric')
            [model_keys_renamed]
            .melt()
        )

        sns.boxplot(
            data=metric_data_long,
            x='model_build',
            y='value',
            ax=ax
        )

        # Set the y-axis limits.
        ax.set_ylim(ylim)

        # Re-format the labels.
        ax.set_ylabel(metric, rotation=0, labelpad=15, size='small')
        ax.tick_params(axis='x', labelsize='small')
        ax.set_xlabel('')
        ax.set_title('')
#endregion

#region: _plot_prediction_scatterplots_right_half
def _plot_prediction_scatterplots_right_half(
        fig, 
        gs2, 
        workflow, 
        model_keys, 
        label_for_metric,
        label_for_model_build,
        label_for_scoring
        ):
    '''
    Plot scatterplots of predictions on the right half of a figure.
    
    Parameters
    ----------
    fig : Figure
        The figure on which to add the subplots.
    gs2 : GridSpec
        The GridSpec object used to specify the subplot layout.
    workflow : Workflow
        The workflow object containing the data.
    model_keys : list
        A list of model keys used to filter and format the data.
    '''
    # TODO: Could raise error if len(model_keys) != 2, etc.
    key_without_selection = next(
        k for k in model_keys if 'without_selection' in k)
    key_with_selection = next(k for k in model_keys if 'with_selection' in k)

    def create_title(sample_type, n_chemicals):
        '''Adds comma separation'''
        description = f"{sample_type} ({'{:,}'.format(n_chemicals)})"
        return f"{_prediction_label}, {description}"
    
    xlabel = label_for_model_build['without_selection']
    ylabel = label_for_model_build['with_selection']

    ## Create the top Axes.
    ax0 = fig.add_subplot(gs2[0])
    x0, *_ = get_in_sample_prediction(workflow, key_without_selection)
    y0, *_ = get_in_sample_prediction(workflow, key_with_selection)
    title0 = create_title('Training Set', len(x0))
    _plot_prediction_scatterplot(
        ax0, 
        x0, 
        y0, 
        workflow, 
        label_for_metric,
        label_for_scoring,
        title0, 
        xlabel, 
        ylabel
        )
    
    ## Create the bottom Axes.
    ax1 = fig.add_subplot(gs2[1])
    x1, *_ = predict_out_of_sample(workflow, key_without_selection)
    y1, *_ = predict_out_of_sample(workflow, key_with_selection)
    title1 = create_title('All Chemicals', len(x1))
    _plot_prediction_scatterplot(
        ax1, 
        x1, 
        y1, 
        workflow, 
        label_for_metric,
        label_for_scoring,
        title1, 
        xlabel, 
        ylabel
        )
#endregion

#region: _plot_prediction_scatterplot
def _plot_prediction_scatterplot(
        ax, 
        x, 
        y, 
        workflow, 
        label_for_metric,
        label_for_scoring,
        title, 
        xlabel, 
        ylabel
        ):
    '''
    '''
    color = 'black'

    generate_scatterplot(
        ax, 
        x,  
        y,  
        workflow, 
        label_for_metric,
        label_for_scoring,
        title=title, 
        color=color, 
        xlabel=xlabel, 
        ylabel=ylabel
    )

    xmin, xmax = ax.get_xlim()
    plot_one_one_line(ax, xmin, xmax)
#endregion

#region: get_in_sample_prediction
def get_in_sample_prediction(workflow, model_key, inverse_transform=False):
    '''
    '''
    model_key_names = get_model_key_names(workflow)
    key_for = dict(zip(model_key_names, model_key))
    X, y = workflow.load_features_and_target(**key_for)

    y_pred, X = get_prediction(X, workflow, model_key, inverse_transform)

    return y_pred, X, y
#endregion

#region: predict_out_of_sample
def predict_out_of_sample(workflow, model_key, inverse_transform=False):
    '''
    '''
    model_key_names = get_model_key_names(workflow)
    key_for = dict(zip(model_key_names, model_key))
    X = workflow.load_features(**key_for)

    y_pred, X = get_prediction(X, workflow, model_key, inverse_transform)

    return y_pred, X
#endregion

#region: get_prediction
def get_prediction(X, workflow, model_key, inverse_transform=False):
    '''
    '''
    estimator = workflow.get_estimator(model_key)
    X = X[estimator.feature_names_in_]
    y_pred = pd.Series(estimator.predict(X), index=X.index)
    if inverse_transform:
        y_pred = 10**y_pred
    return y_pred, X
#endregion

#region: important_feature_counts
def important_feature_counts(workflow, label_for_effect):
    '''
    '''
    # Define colorblind-friendly colors
    color_filled = '#1f77b4'  # Blue
    color_unfilled = '#dcdcdc'  # Light gray

    model_key_names = get_model_key_names(workflow)
    combination_key_groups = get_model_key_groups(
        workflow, model_key_names, 'target_effect', 
        exclusion_string='without_selection'
    )

    for combination_key, model_key_group in combination_key_groups:
        model_keys = list(model_key_group)
        n_keys = len(model_keys)

        # Figure layout is adjusted to have one column per model key
        fig, axs = plt.subplots(1, n_keys, figsize=(5*n_keys, 8), sharey=True)

        for idx, model_key in enumerate(model_keys):
            ## Prepare the data for plotting.
            features_for_final_model = workflow.get_important_features(model_key)
            features_for_replicate_model = (
                workflow.get_important_features_replicates(model_key))

            # Initialize feature_counts dictionary
            key_for = dict(zip(model_key_names, model_key))
            all_feature_names = list(workflow.load_features(**key_for))
            feature_counts = {feature: 0 for feature in all_feature_names}

            # Count the occurrences of each feature
            for feature_list in features_for_replicate_model.values():
                for feature in feature_list:
                    if feature in feature_counts:
                        feature_counts[feature] += 1

            # Sort features based on their counts
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

            axs[idx].barh(
                bar_positions, bar_counts, color=bar_colors, edgecolor='black', 
                linewidth=1)
            axs[idx].set_ylim(-1, len(all_feature_names))
            axs[idx].set_yticks(range(len(all_feature_names)))
            axs[idx].set_yticklabels(sorted_features, fontsize=10)
            axs[idx].tick_params(axis='y', pad=0)
            axs[idx].invert_yaxis()
            axs[idx].set_xlabel('Count', fontsize=12)
            axs[idx].set_title(
                label_for_effect[key_for['target_effect']], fontsize=12)

            if idx != 0:  # Only set ylabel for first subplot
                axs[idx].set_ylabel('')

        ## Add legend below the plots
        legend_patches = [
            mpatches.Patch(color=color_filled, label='Included in final model'),
            mpatches.Patch(color=color_unfilled, label='Not included in final model')
        ]
        fig.legend(handles=legend_patches, loc='lower center', fontsize=10, ncol=2,
                bbox_to_anchor=(0.5, -0.05))

        fig.tight_layout()

        save_figure(
            fig, 
            important_feature_counts, 
            combination_key
            )
#endregion

#region: importances_boxplots
def importances_boxplots(workflow, label_for_scoring):
    '''
    '''
    _feature_importances_boxplots(
        workflow, 
        'importances', 
        importances_boxplots,
        label_for_scoring
        )
#endregion

#region: importances_replicates_boxplots
def importances_replicates_boxplots(workflow, label_for_scoring):
    '''
    '''
    _feature_importances_boxplots(
        workflow, 
        'importances_replicates', 
        importances_replicates_boxplots,
        label_for_scoring
        )
#endregion

#region: _feature_importances_boxplots
def _feature_importances_boxplots(
        workflow, 
        importances_key, 
        function, 
        label_for_scoring
        ):
    '''
    Plot boxplots for feature importances for all models in workflow.

    Parameters
    ----------
    workflow : object
        The workflow object containing the model keys and corresponding history.

    Returns
    -------
    None : None
    '''
    importances_wide = workflow.concatenate_history(importances_key)
    xlabel = 'Δ Score'
    figsize = (8, 10)

    model_keys = [k for k in workflow.model_keys if 'with_selection' in k]
    for model_key in model_keys:
        
        # Format the data for plotting.
        cv_importances_long = {}
        for score in label_for_scoring:
            cv_importances_long[score] = (
                importances_wide[model_key][score]
                .melt()
                .sort_values(by='value', ascending=False) 
            )
        
        # Get the column names from the last iteration.
        y, x = list(cv_importances_long[score])
        
        fig, axs = plot.vertical_boxplots(
            cv_importances_long, 
            x, 
            y,
            xlabel,
            title_for_key=label_for_scoring,
            figsize=figsize,
            palette='icefire', 
            whis=[0, 100], 
            linewidth=0.8
        )

        fig.tight_layout()

        save_figure(
            fig, 
            function, 
            model_key
            )
#endregion

#region: benchmarking_scatterplots
def benchmarking_scatterplots(
        workflow,
        y_regulatory_df,
        y_toxcast,
        label_for_effect,
        color_for_effect,
        label_for_metric,
        label_for_scoring,
        figsize=(6, 9)
        ):
    '''
    Generate scatterplots for each unique combination of the evaluation and
    comparison datasets. One subplot is created for each model key grouped by
    target effect.

    Parameters
    ----------
    workflow : object
        The workflow object containing the model keys.
    y_regulatory_df : pd.DataFrame
        The DataFrame containing the comparison data.
    y_toxcast : pd.DataFrame
        The ToxCast data for evaluation.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    color_for_effect : dict
        A dictionary mapping target effect to color.
    label_for_effect : dict
        A dictionary mapping target effect to label.
    figsize : tuple, optional
        Figure size. If None, a default size is used.
    write_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    figs : list
        List of generated figures.
    axs : list
        List of axes corresponding to the figures.
    '''
    xlabel = f'Regulatory {_prediction_label}'

    model_key_names = get_model_key_names(workflow)
    combination_key_groups = get_model_key_groups(
        workflow, model_key_names, 'target_effect')

    for combination_key, group in combination_key_groups:
        model_keys = list(group)
        num_subplots = len(model_keys)

        fig, ax_objs = plt.subplots(3, num_subplots, figsize=figsize)

        xmin, xmax = np.inf, -np.inf

        for i, model_key in enumerate(model_keys):
            
            y_pred, _, y_true = get_in_sample_prediction(workflow, model_key)

            key_for = dict(zip(model_key_names, model_key))
            y_comparison = y_regulatory_df[key_for['target_effect']].dropna()
            y_evaluation_dict = {
                'ToxValDB' : y_true, 
                'QSAR' : y_pred,
                'ToxCast/httk' : y_toxcast,
            }

            for j, (label, y_evaluation) in enumerate(
                    y_evaluation_dict.items()):

                ax = ax_objs[j, i]

                if j == 0:
                    # Set title only for the first row.
                    title = label_for_effect[key_for['target_effect']]
                else:
                    title = ''

                color = color_for_effect[key_for['target_effect']]
                ylabel = f'{label} {_prediction_label}'

                generate_scatterplot(
                    ax, 
                    y_comparison, 
                    y_evaluation, 
                    workflow, 
                    label_for_metric,
                    label_for_scoring,
                    title, 
                    color, 
                    xlabel, 
                    ylabel
                    )

                xmin = min(xmin, *ax.get_xlim())
                xmax = max(xmax, *ax.get_xlim())

            for ax in ax_objs.flatten():
                plot_one_one_line(ax, xmin, xmax)

        fig.tight_layout()
        
        save_figure(
            fig, 
            benchmarking_scatterplots, 
            combination_key
            )
#endregion

#region: generate_scatterplot
def generate_scatterplot(
        ax, 
        y_comparison, 
        y_evaluation, 
        workflow, 
        label_for_metric,
        label_for_scoring,
        title='', 
        color=None, 
        xlabel='', 
        ylabel=''
        ):
    '''
    Generate a scatterplot comparing two sets of data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot the scatterplot on.
    y_comparison : pd.Series
        The comparison data (x).
    y_evaluation : pd.Series
        The evaluation data (y).
    workflow : object
        The workflow object containing additional information.
    title : str
        The title of the scatterplot.
    color : str
        The color for the scatterplot points.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    '''
    chem_intersection = y_comparison.index.intersection(y_evaluation.index)
    y_comparison = y_comparison.loc[chem_intersection]
    y_evaluation = y_evaluation.loc[chem_intersection]

    scores = [
        workflow.function_for_metric[met](y_comparison, y_evaluation)
        for met in label_for_metric
    ]
    scores = [score_to_string(score) for score in scores]
    score_for_label = dict(zip(
        label_for_scoring.values(),
        scores))

    ax.scatter(
        y_comparison,
        y_evaluation,
        alpha=0.7,
        color=color
    )

    ax.set_title(title, fontsize='medium')

    score_text = format_score_text(score_for_label)
    ax.text(0.05, 0.95, score_text, transform=ax.transAxes,
            va='top', ha='left', size='small')

    ax.set_xlabel(xlabel, size='small')
    ax.set_ylabel(ylabel, size='small')
#endregion

#region: plot_one_one_line
def plot_one_one_line(ax, xmin, xmax):
    '''
    Add one-one line to a subplot

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object to add the line.
    xmin : float
        Minimum limit for x and y axes.
    xmax : float
        Maximum limit for x and y axes.
    '''
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.plot([xmin, xmax], [xmin, xmax], color='#808080', 
            linestyle='--', linewidth=1)
#endregion

#region: score_to_string
def score_to_string(score, precision=2):
    '''Helper function for plotting scores (float) as text.
    '''
    return f'{score:.{precision}f}'
#endregion

#region: format_score_text
def format_score_text(score_labels):
    '''
    Convert score labels dictionary to formatted string.

    Parameters
    ----------
    score_labels : dict
        Score labels.

    Returns
    -------
    str
        Formatted score text.
    '''
    return '\n'.join([f'{key}: {value}' for key, value in score_labels.items()])
#endregion

# TODO: Map the percentile columns to new labels.
#region: margins_of_exposure_cumulative
def margins_of_exposure_cumulative(
        workflow, exposure_df, label_for_effect, right_truncation=None):
    '''
    Function to plot cumulative count of chemicals for different MOE categories.

    Parameters
    ----------
    workflow : object
        The workflow object which contains models and functions to predict and plot.
    exposure_df : pd.DataFrame
        DataFrame with exposure estimates.

    Returns
    -------
    None : None
    '''
    model_key_names = get_model_key_names(workflow)
    combination_key_groups = get_model_key_groups(
        workflow, model_key_names, 'target_effect'
    )

    # Define colors for different percentiles
    colors = sns.color_palette("Set2", len(exposure_df.columns))

    # Define MOE categories
    moe_categories = {"Potential Concern": [1, 100], "Definite Concern": [0, 1]}
    moe_colors = sns.color_palette("Paired", len(moe_categories))

    for combination_key, group in combination_key_groups:
        model_keys = list(group)

        # Initialize a figure
        fig, axs = plt.subplots(
            1,
            len(model_keys),
            figsize=(len(model_keys) * 5, 5),
            sharey=True
        )

        for i, model_key in enumerate(model_keys):
            y_pred, *_ = predict_out_of_sample(
                workflow,
                model_key,
                inverse_transform=True
            )

            for j, percentile in enumerate(exposure_df.columns):
                exposure_estimates = exposure_df[percentile]
                margins_of_exposure = y_pred.divide(exposure_estimates)

                # Sort the MOEs and calculate cumulative counts
                sorted_moe = margins_of_exposure.sort_values()
                cumulative_counts = np.arange(1, len(sorted_moe) + 1)

                # Plot the cumulative counts
                axs[i].plot(sorted_moe, cumulative_counts, color=colors[j],
                            label=f'Exposure {percentile}')

            # Set titles, labels, scale, and gridlines
            key_for = dict(zip(model_key_names, model_key))
            effect = key_for['target_effect']
            axs[i].set_title(label_for_effect[effect])
            axs[i].set_xlabel('Margin of Exposure')
            axs[i].set_xscale('log')
            axs[i].set_yscale('log')
            axs[i].grid(True, which='both', linestyle='--', linewidth=0.5)

            if i == 0:  # only label y-axis for the leftmost plot
                axs[i].set_ylabel('Cumulative Count of Chemicals')

            # Set x-axis limit based on the user input
            if right_truncation:
                left = axs[i].get_xlim()[0]
                axs[i].set_xlim(left, right_truncation)

            # Indicate MOE categories with vertical spans
            for k, (category, (lower, upper)) in enumerate(moe_categories.items()):
                axs[i].axvspan(lower, upper, alpha=0.2, color=moe_colors[k])

                # Calculate x-position for category annotations
                x_position = axs[i].get_xlim()[0] + lower
                data_to_axes = axs[i].transData + axs[i].transAxes.inverted()
                x_position = data_to_axes.transform((x_position, 0))[0]

                # Calculate y-position for category annotations
                y_position = 0.99

                # Plot category annotations
                axs[i].text(
                    x_position,
                    y_position,
                    category.replace(" ", "\n"),
                    ha='left',
                    va='top',
                    fontsize='small',
                    transform=axs[i].transAxes
                )

        # Set a single legend for all subplots
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', fontsize='small',
                   ncol=len(exposure_df.columns) + 2, bbox_to_anchor=(0.5, -0.05))

        fig.tight_layout()

        save_figure(
            fig,
            margins_of_exposure_cumulative,
            combination_key
        )
#endregion

#region: predictions_by_missing_feature
def predictions_by_missing_feature(workflow, label_for_effect):
    '''
    Generate boxplots of the predictions, organized by the combination of the 
    features of the models.

    For each unique combination of the features in the model, the function
    generates a subplot with a boxplot for each target effect. The boxplots
    represent the predictions, categorized by the missing samples for each 
    feature. The boxplots are colored based on the samples, with a unique 
    color for all samples and a different color for the remaining.

    Parameters
    ----------
    workflow : object
    '''
    # Set seaborn style locally within the function
    with sns.axes_style('whitegrid'):
        # Create a color palette for the boxes
        palette = sns.color_palette('Set2')
        all_samples_color = palette[1]  # red
        remaining_color = palette[0]  # green

        model_key_names = get_model_key_names(workflow)

        # Use the helper function to get the combination key group
        combination_key_groups = get_model_key_groups(
            workflow, model_key_names, 'target_effect')
        
        # Iterate over combination_key_group
        for combination_key, group in combination_key_groups:
            model_keys = list(group)
            n_effects = len(model_keys)

            fig, axs = plt.subplots(
                nrows=n_effects, 
                figsize=(6, 4 * n_effects)
                )

            for i, model_key in enumerate(model_keys):

                key_for = dict(zip(model_key_names, model_key))
                title = label_for_effect[key_for['target_effect']]

                ## Plot the distributions for training chemicals.
                y_pred, X, *_ = get_in_sample_prediction(workflow, model_key)
                _boxplot_by_missing_feature(
                    axs[i], 
                    y_pred, 
                    X, 
                    all_samples_color, 
                    remaining_color, 
                    title
                    )

            fig.tight_layout()

            save_figure(
                fig, 
                predictions_by_missing_feature, 
                combination_key
                )
#endregion

#region: _boxplot_by_missing_feature
def _boxplot_by_missing_feature(
            ax, 
            df, 
            X, 
            all_samples_color, 
            remaining_color, 
            title
            ):
        '''
        Draw a boxplot on the given axis, illustrating the distributions
        for all samples and the remaining samples after excluding missing 
        values.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to draw the boxplot on.
        df : pandas.DataFrame
            The input data.
        X : pandas.DataFrame
            DataFrame containing feature values, used to identify missing 
            samples.
        all_samples_color : matplotlib color
            Color for the 'All Samples' box in the boxplot.
        remaining_color : matplotlib color
            Color for the remaining boxes in the boxplot.
        title : str
            Title for the boxplot.

        Returns
        -------
        None
        '''
        df_for_name = {}
        df_for_name['All Samples'] = df
        for feature_name in X.columns:
            missing_samples = X[X[feature_name].isna()].index
            df_for_name[feature_name] = df.loc[missing_samples]

        # Sort by the sample size and add sample size to labels
        df_for_name = {_relabel_box(k, v): v for k, v in sorted(
            df_for_name.items(), key=lambda item: item[1].size, reverse=False)}

    # Define properties for outliers
        flierprops = dict(marker='o', markerfacecolor='lightgray', markersize=2,
                    linestyle='none', markeredgecolor='lightgray')

        boxplot = ax.boxplot(
            list(df_for_name.values()),
            vert=False,
            labels=list(df_for_name.keys()),
            widths=0.6,
            patch_artist=True,
            medianprops={'color': 'black'},
            flierprops=flierprops
        )

        for patch in boxplot['boxes']:
            patch.set(facecolor=remaining_color)

        patch.set(facecolor=all_samples_color)

        for key in ['whiskers', 'caps', 'medians']:
            for element in boxplot[key]:
                element.set(color='black')

        median_value = boxplot['medians'][-1].get_xdata()[0]
        ax.axvline(
            x=median_value,
            color=all_samples_color,
            linestyle='--',
            alpha=0.5
        )
        
        ax.set_title(title)
        ax.set_xlabel(f'Predicted {_prediction_label}')
        # Make y-axis tick labels smaller
        ax.tick_params(axis='y', which='major', labelsize=8)
#endregion

#region: _relabel_box
def _relabel_box(name, series):
    '''Return an updated box label for the missing feature. 
    '''
    prefix = 'Missing' if name != 'All Samples' else ''
    sample_size = len(series)
    return f'{prefix} {name} ({sample_size})'
#endregion

#region: get_model_key_groups
def get_model_key_groups(
        workflow, key_names, grouping_key, exclusion_string=None):
    '''
    Get combination key group by sorting model keys based on combination key.

    Parameters
    ----------
    workflow : object
        The workflow object containing the model keys.
    key_names : list
        List of key names to construct the model keys.
    grouping_key : str
        The key to exclude from the combination key.
    exclusion_string : str, optional
        The string to check for in the model keys. Keys containing this string 
        will be excluded.

    Returns
    -------
    combination_key_group : generator
        Generator yielding combination key and group of model keys.
    '''
    # Get the index of the key to exclude in the key names
    key_idx = key_names.index(grouping_key)

    # Filter model keys to exclude those containing the exclusion_string
    model_keys = workflow.model_keys
    if exclusion_string is not None:
        model_keys = [k for k in model_keys if exclusion_string not in k]

    # Sort model keys by combination key without affecting the original model keys
    sorted_model_keys = sorted(model_keys, key=lambda x: tuple(
        item for idx, item in enumerate(x) if idx != key_idx))

    # Define variable for clarity
    combination_key_groups = (
        (combination_key, group)
        for combination_key, group in itertools.groupby(
            sorted_model_keys, key=lambda x: tuple(
                item for idx, item in enumerate(x) if idx != key_idx
            )
        )
    )

    return combination_key_groups
#endregion

# TODO: Move this to the Workflow.History?
#region: get_model_key_names
def get_model_key_names(workflow):
    return workflow.instruction_names + ['estimator']
#endregion

#region: save_figure
def save_figure(fig, function, combination_key):
    '''
    '''
    output_dir = function_directory(function)
    fig_path = figure_path(output_dir, combination_key)
    print(f'Saving figure --> "{fig_path}"')
    fig.savefig(fig_path)
#endregion

#region: figure_path
def figure_path(function_dir, combination_key):
    '''
    '''
    fig_name = '-'.join(map(str, combination_key)) + '.png'
    return os.path.join(function_dir, fig_name)
#endregion

#region: function_directory
def function_directory(function):
    '''
    Get the name of the function and create a directory in the output 
    directory for it
    '''
    output_dir = os.path.join('Figures', function.__name__)
    ensure_directory_exists(output_dir)
    return output_dir
#endregion

#region: ensure_directory_exists
def ensure_directory_exists(path):
    '''Check if the directory at `path` exists and if not, create it.'''
    if not os.path.exists(path):
        os.makedirs(path)
#endregion