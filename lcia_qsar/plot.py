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
            label_for_model_build
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
        if i == 0:
            ax.set_title('Performance Comparison', size='medium')
#endregion

#region: _plot_prediction_scatterplots_right_half
def _plot_prediction_scatterplots_right_half(
        fig, 
        gs2, 
        workflow, 
        model_keys, 
        label_for_metric,
        label_for_model_build
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
        return f"{sample_type} ({_comma_separated(n_chemicals)})"
    
    xlabel = label_for_model_build['without_selection']
    ylabel = label_for_model_build['with_selection']

    ## Create the top Axes.
    ax0 = fig.add_subplot(gs2[0])
    x0, *_ = get_in_sample_prediction(workflow, key_without_selection)
    y0, *_ = get_in_sample_prediction(workflow, key_with_selection)
    overall_title = f'Predicted {_prediction_label} Comparison'
    ax0_title = f"{overall_title}\n{create_title('Training Set', len(x0))}"
    _plot_prediction_scatterplot(
        ax0, 
        x0, 
        y0, 
        workflow, 
        label_for_metric,
        ax0_title, 
        xlabel, 
        ylabel
        )
    
    ## Create the bottom Axes.
    ax1 = fig.add_subplot(gs2[1])
    x1, *_ = predict_out_of_sample(workflow, key_without_selection)
    y1, *_ = predict_out_of_sample(workflow, key_with_selection)
    ax1_title = create_title('All Chemicals', len(x1))
    _plot_prediction_scatterplot(
        ax1, 
        x1, 
        y1, 
        workflow, 
        label_for_metric,
        ax1_title, 
        xlabel, 
        ylabel
        )
#endregion

#region: _comma_separated
def _comma_separated(number):
    '''Convert float or int to a string with comma-separated thousands.
    '''
    return '{:,}'.format(number)
#endregion

#region: _plot_prediction_scatterplot
def _plot_prediction_scatterplot(
        ax, 
        x, 
        y, 
        workflow, 
        label_for_metric,
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
        color=color, 
        title=title, 
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
            axs[idx].set_ylabel('Feature Names', fontsize=12)
            axs[idx].set_title(
                label_for_effect[key_for['target_effect']], fontsize=12)

            if idx != 0:  # Only set ylabel for first subplot
                axs[idx].set_ylabel('')

        ## Add legend on the first plot below the xlabel
        legend_patches = [
            mpatches.Patch(color=color_filled, label='In Final Model'),
            mpatches.Patch(color=color_unfilled, label='Not in Final Model')
        ]
        
        # Set legend below the right Axes, right-flush.
        legend_ax = axs[-1]
        legend_ax.legend(
            handles=legend_patches,
            loc='upper right',
            bbox_to_anchor=(1, -0.1),
            bbox_transform=legend_ax.transAxes,
            fontsize=10,
            ncol=1
        )
        
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

# TODO: Define Feature Names label globally.
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
        # TODO: Add this logic in the function, vertical_boxplots?
        for i, ax in enumerate(axs.flatten()):
            ax.tick_params(axis='y', size=10)
            if i == 0:
                ax.set_ylabel('Feature Names', size=12)

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

                color = color_for_effect[key_for['target_effect']]

                ## Set labels depending on the Axes.
                title, xlabel, ylabel = '', '', ''
                if j == 0:  # first row
                    title = label_for_effect[key_for['target_effect']]
                if j == len(y_evaluation_dict)-1:  # last row
                    xlabel = f'Regulatory {_prediction_label}'
                if i == 0:  # first column
                    ylabel = f'{label} {_prediction_label}'
                
                generate_scatterplot(
                    ax, 
                    y_comparison, 
                    y_evaluation, 
                    workflow, 
                    label_for_metric,
                    color=color, 
                    title=title, 
                    xlabel=xlabel, 
                    ylabel=ylabel
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
        y_true, 
        y_pred, 
        workflow, 
        label_for_metric,
        color=None, 
        title='', 
        xlabel='', 
        ylabel=''
        ):
    '''
    Generate a scatterplot comparing two sets of data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot the scatterplot on.
    y_true : pd.Series
        The comparison data (x).
    y_pred : pd.Series
        The evaluation data (y).
    workflow : object
        The workflow object containing additional information.
    color : str
        The color for the scatterplot points.
    title : str
        The title of the scatterplot.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    '''
    chem_intersection = y_true.index.intersection(y_pred.index)
    y_true = y_true.loc[chem_intersection]
    y_pred = y_pred.loc[chem_intersection]

    ax.scatter(
        y_true,
        y_pred,
        alpha=0.7,
        color=color
    )

    ## Set the performance scores as text.
    float_to_string = lambda score : format(score, '.2f')  # limit precision
    dict_to_string = lambda d : '\n'.join([f'{k}: {v}' for k, v in d.items()])
    get_score = (
        lambda metric : workflow.function_for_metric[metric](y_true, y_pred))
    score_text = dict_to_string(
        {label : float_to_string(get_score(metric)) 
         for metric, label in label_for_metric.items()}
    )
    ax.text(0.05, 0.95, score_text, transform=ax.transAxes,
            va='top', ha='left', size='small')

    if title:
        ax.set_title(title, fontsize='medium')
    if xlabel:
        ax.set_xlabel(xlabel, size='small')
    if ylabel:
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

#region: margins_of_exposure_cumulative
def margins_of_exposure_cumulative(
        workflow, 
        exposure_df, 
        label_for_effect, 
        label_for_exposure_column, 
        right_truncation=None
        ):
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
    moe_categories = {
        'Potential Concern': (1., 100.), 
        'Definite Concern': (0., 1.)
        }
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
                axs[i].plot(
                    sorted_moe, 
                    cumulative_counts, 
                    color=colors[j],
                    label=label_for_exposure_column[percentile]
                    )

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
                # If lower is outside the left limit, set it to the left limit
                lower = max(lower, left) 
                 # If upper is outside the right limit, set it to the right limit
                upper = min(upper, right_truncation) 
                # Compute geometric mean for logarithmic scale
                x_position = np.sqrt(lower * upper)

                # Transform the calculated x_position from data coordinates 
                # to axes fraction coordinates
                xlim = axs[i].get_xlim()
                x_position = (
                    (np.log10(x_position) - np.log10(xlim[0])) 
                    / (np.log10(xlim[1]) - np.log10(xlim[0]))
                )

                # Calculate y-position for category annotations
                y_position = 0.97

                # Plot category annotations
                axs[i].text(
                    x_position,
                    y_position,
                    category.replace(" ", "\n"),
                    ha='center',  # change alignment to 'center'
                    va='top',
                    fontsize='small',
                    transform=axs[i].transAxes
                )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)  # accomodate the legend

        # Set a single legend.
        legend_ax = axs[-1]
        handles, labels = legend_ax.get_legend_handles_labels()
        fig.legend(
            handles, 
            labels, 
            loc='lower center', 
            fontsize='small',
            ncol=len(labels), 
            bbox_to_anchor=(0.5, -0.01)
        )

        save_figure(
            fig,
            margins_of_exposure_cumulative,
            combination_key
        )
#endregion

# TODO: Define labels globally for the chemical sets.
#region: predictions_by_missing_feature
def predictions_by_missing_feature(workflow, label_for_effect):
    '''
    Generate and plot in-sample and out-of-sample predictions.

    Parameters
    ----------
    workflow : object
        The workflow object containing the trained model.
    label_for_effect : dict
        A dictionary mapping the effect to its label.

    Returns
    -------
    None
    '''
    # Set seaborn style locally within the function
    with sns.axes_style('whitegrid'):

        all_samples_color = '#00008b'  # dark blue
        remaining_color = '#ffff99'  # pale yellow

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
                ncols=2,
                figsize=(8, 4 * n_effects)
                )

            for i, model_key in enumerate(model_keys):
                key_for = dict(zip(model_key_names, model_key))

                y_pred_out, X_out, *_ = predict_out_of_sample(workflow, model_key)
                y_pred_in, X_in, *_ = get_in_sample_prediction(workflow, model_key)

                dfs_out = _boxplot_by_missing_feature(
                    axs[i, 0], 
                    y_pred_out, 
                    X_out, 
                    all_samples_color, 
                    remaining_color
                    )
                # Use the sames sort order of boxes, based on the left Axes.
                sort_order = list(dfs_out.keys())
                dfs_in = _boxplot_by_missing_feature(
                    axs[i, 1], 
                    y_pred_in, 
                    X_in, 
                    all_samples_color, 
                    remaining_color, 
                    sort_order
                    )

                _set_ytick_labels(axs[i, 0], dfs_out, True)
                _set_ytick_labels(axs[i, 1], dfs_in, False)
                
                effect = label_for_effect[key_for['target_effect']]
                axs[i, 0].set_title(
                    f'{effect}\nAll Chemicals', size='medium', loc='left')
                axs[i, 1].set_title(
                    f'{effect}\nTraining Set', size='medium', loc='left')

            fig.tight_layout()

            save_figure(
                fig, 
                predictions_by_missing_feature, 
                combination_key
                )
#endregion

#region: _boxplot_by_missing_feature
def _boxplot_by_missing_feature(
        ax, df, X, all_samples_color, remaining_color, sort_order=None):
    '''
    Generate boxplot by missing feature on the provided axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw the boxplot on.
    df : pd.DataFrame
        The input data.
    X : pd.DataFrame
        DataFrame containing feature values, used to identify missing samples.
    all_samples_color : str
        Color for the 'All Samples' box in the boxplot.
    remaining_color : str
        Color for the remaining boxes in the boxplot.
    sort_order : list of str, optional
        Order to sort the features. If None, sort by the sample size.

    Returns
    -------
    df_for_name : dict
        A dictionary with keys as feature names and values as dataframes 
        representing the distributions.
    '''
    df_for_name = {}
    df_for_name['All Samples'] = df
    for feature_name in X.columns:
        missing_samples = X[X[feature_name].isna()].index
        df_for_name[feature_name] = df.loc[missing_samples]

    # If no sort order is provided, sort by the sample size.
    if sort_order is None:
        sort_order = sorted(
            df_for_name.keys(), 
            key=lambda name: df_for_name[name].size, 
            reverse=False
        )

    df_for_name = {name: df_for_name[name] for name in sort_order}

    _create_boxplot(ax, df_for_name, all_samples_color, remaining_color)

    return df_for_name
#endregion

#region: _create_boxplot
def _create_boxplot(ax, df_for_name, all_samples_color, remaining_color):
    '''
    Draw a boxplot on the given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw the boxplot on.
    df_for_name : dict of pd.DataFrame
        Dictionary with keys as feature names and values as dataframes 
        representing the distributions.
    all_samples_color : str
        Color for the 'All Samples' box in the boxplot.
    remaining_color : str
        Color for the remaining boxes in the boxplot.
    '''
    # Define properties for outliers
    flierprops = dict(marker='o', markerfacecolor='lightgray', markersize=2,
                      linestyle='none', markeredgecolor='lightgray')

    boxplot = ax.boxplot(
        list(df_for_name.values()),
        vert=False,
        labels=[None]*len(df_for_name),  # will be updated later
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

    ax.set_xlabel(f'Predicted {_prediction_label}')
    ax.tick_params(axis='y', which='major', labelsize=8)
#endregion

#region: _set_ytick_labels
def _set_ytick_labels(ax, df_for_name, do_label_features):
    '''
    Sets the ytick labels for a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw the boxplot on.
    df_for_name : dict of pd.DataFrame
        Dictionary with keys as feature names and values as dataframes 
        representing the distributions.
    do_label_features : bool
        If True, feature labels are set for the y-axis ticks. If False, 
        feature labels are not set.
    '''
    feature_labels = []
    sample_size_labels = []
    for name, series in df_for_name.items():
        feature_label, sample_size_label = _get_box_tick_labels(name, series)
        if do_label_features:
            feature_labels.append(feature_label)
        else:
            feature_labels.append(None)
        sample_size_labels.append(sample_size_label)

    ax.set_yticklabels(feature_labels, fontsize=8)
    ax.set_ylabel(None)
    
    # Add secondary y-axis for the sample size labels
    axs2 = ax.twinx()
    axs2.set_yticks(ax.get_yticks())
    axs2.set_ylim(ax.get_ylim())
    axs2.set_yticklabels(sample_size_labels, fontsize=8)
    axs2.yaxis.set_label_position("right")
#endregion

#region: _get_box_tick_labels
def _get_box_tick_labels(name, series):
    '''
    Return the box label components.

    Parameters
    ----------
    name : str
        Feature name.
    series : pd.Series
        Data series.

    Returns
    -------
    feature_label : str
        Updated box label for the feature.
    sample_size_label : str
        Sample size of the data series in a string format.
    '''
    if name != 'All Samples':
        prefix = 'Missing'
        suffix = f'"{name}"'
    else:
        prefix = ''
        suffix = name
    feature_label = f'{prefix} {suffix}'

    sample_size_label = _comma_separated(len(series))

    return feature_label, sample_size_label 
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