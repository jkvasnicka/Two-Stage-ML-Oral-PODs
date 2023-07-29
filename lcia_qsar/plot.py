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

# TODO: Move to configuration file.
_flierprops = dict(
    marker='o', 
    markerfacecolor='lightgray', 
    markersize=2,
    linestyle='none', 
    markeredgecolor='lightgray'
    )

#region: proportions_incomplete_subplots
def proportions_incomplete_subplots(
        features_file, 
        AD_file, 
        targets_file, 
        label_for_effect,
        all_chemicals_label, 
        base_size_per_feature=(0.2, 6)
    ):
    '''
    '''
    X = pd.read_csv(features_file, index_col=0)
    AD_flags = pd.read_csv(AD_file, index_col=0)
    ys = pd.read_csv(targets_file, index_col=0)

    ## Plot the training set data.
    samples_for_effect = {
        label_for_effect[effect] : y.dropna().index 
        for effect, y in ys.items()
        }
    proportions_incomplete_subplot(
        X, 
        AD_flags, 
        samples_for_effect,
        base_size_per_feature=base_size_per_feature
    )

    ## Plot all chemicals data.
    proportions_incomplete_subplot(
        X, 
        AD_flags, 
        {all_chemicals_label : X.index},
        base_size_per_feature=base_size_per_feature
    )
#endregion

#region: proportions_incomplete_subplot
def proportions_incomplete_subplot(
        X, 
        AD_flags, 
        samples_dict, 
        base_size_per_feature=(0.2, 6)
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
        valid_prop = valid_prop.sort_values(ascending=False)
        missing_prop = missing_prop.reindex(valid_prop.index)
        outside_AD_prop = outside_AD_prop.reindex(valid_prop.index)

        n_samples = len(sample_IDs)
        proportions_incomplete_barchart(
            axs[i], 
            missing_prop, 
            outside_AD_prop, 
            valid_prop, 
            title, 
            n_samples
            )

        # Remove ylabel for all but the first subplot
        if i != 0:
            axs[i].set_ylabel('')
            
    fig.tight_layout()

    save_figure(
        fig, 
        proportions_incomplete_subplot, 
        list(titles)
        )
#endregion

#region: proportions_incomplete_barchart
def proportions_incomplete_barchart(
        ax, missing_prop, outside_AD_prop, valid_prop, title, n_samples):
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
    
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f'{index} ({int(value)}%)' 
         for index, value in valid_prop.items()], 
         fontsize=8)
    ax.set_xlabel('% Incomplete', size='medium')
    ax.set_ylabel('Feature Names (% Complete)', size='medium')
    n_samples =  _comma_separated(n_samples)
    ax.set_title(f'{title} (n={n_samples})', size='medium', loc='left')
    ax.set_xlim([0, 100])
    ax.grid(True, axis='x', linestyle='--', color='black', alpha=0.6)
    ax.legend(fontsize='small', loc='lower right')
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
    outside_AD_prop = AD_flags_subset.mean() * 100
    not_outside_AD = ~AD_flags_subset
    missing_prop = (X_subset.isna() & not_outside_AD).mean() * 100
    missing_prop, outside_AD_prop = missing_prop.align(
        outside_AD_prop, fill_value=0)
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

#region: performance_and_prediction_comparison
def performance_and_prediction_comparison(
        workflow, 
        label_for_metric, 
        label_for_model_build,
        label_for_effect,
        prediction_label, 
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
    grouped_keys_outer = group_model_keys(
        workflow.model_keys, 
        model_key_names, 
        ['target_effect', 'model_build']
        )

    for grouping_key_outer, model_keys in grouped_keys_outer:

        grouped_keys_inner = group_model_keys(
            model_keys, 
            model_key_names, 
            'model_build'
            )
        
        # Initialize a Figure for the subplot.
        fig = plt.figure(figsize=(7, 6))

        gs1 = gridspec.GridSpec(2, 2)
        gs2 = gridspec.GridSpec(6, 1)

        _plot_prediction_scatterplots(
            fig, 
            gs1, 
            workflow, 
            grouped_keys_inner, 
            model_key_names,
            label_for_metric,
            label_for_model_build,
            label_for_effect,
            prediction_label
            )
        
        _plot_performances_boxplots(
            fig, 
            gs2, 
            workflow, 
            grouped_keys_inner, 
            label_for_metric, 
            label_for_model_build,
            ylim=ylim
            )

        # adjust the 'right' value for gs1 and 'left' value for gs2
        gs1.tight_layout(fig, rect=[0, 0, 0.7, 1]) 
        gs2.tight_layout(fig, rect=[0.7, 0, 1, 1], h_pad=0.5)

        save_figure(
            fig, 
            performance_and_prediction_comparison, 
            grouping_key_outer
            )
#endregion

#region: _plot_performances_boxplots
def _plot_performances_boxplots(
        fig, 
        gs2, 
        workflow, 
        grouped_keys_inner, 
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
    gs2 : GridSpec
        The GridSpec object used to specify the subplot layout.
    workflow : Workflow
        The workflow object containing the data.
    model_keys : list
        A list of model keys used to filter and format the data.
    ylim : tuple, optional
        A tuple specifying the y-axis limits. Default is (0., 1.).
    '''
    # TODO: Is all this necessary?
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
    label_for_model_build = {
        k : v.split(' ')[0] for k, v in label_for_model_build.items()}
    label_for_column = {
        **label_for_metric, 
        **label_for_model_build
    }
    performances_wide = performances_wide.rename(label_for_column, axis=1)

    ## Plot the data. 

    metrics = performances_wide.columns.unique(level='metric')

    # Create a counter for the current row
    index = 0
    for i, (_, model_keys) in enumerate(grouped_keys_inner):

        # The new keys will be used to get the data.
        model_keys_renamed = [
            tuple(label_for_column.get(k, k) for k in model_key) 
            for model_key in model_keys
        ]
        
        for j, metric in enumerate(metrics):

            ax = fig.add_subplot(gs2[index])                

            # Filter and format the data.
            metric_data_long = (
                performances_wide.xs(metric, axis=1, level='metric')
                [model_keys_renamed]
                .melt()
            )

            sns.boxplot(
                data=metric_data_long,
                y='model_build',  # swapped 'x' and 'y'
                x='value',  # swapped 'x' and 'y'
                ax=ax,
                flierprops=_flierprops,
                orient='h'  # added 'orient' to specify the orientation of the boxplots
            )

            # Set the x-axis limits.
            ax.set_xlim(ylim)

            # Re-format the labels.
            ax.set_xlabel(metric, size='small') 
            ax.set_ylabel('')
            ax.tick_params(axis='both', labelsize='small')

            # Increase the counter
            index += 1
#endregion

#region: _plot_prediction_scatterplots
def _plot_prediction_scatterplots(
        fig, 
        gs1, 
        workflow, 
        grouped_keys_inner, 
        model_key_names, 
        label_for_metric,
        label_for_model_build,
        label_for_effect, 
        prediction_label
        ):
    '''
    Plot scatterplots of predictions on the right half of a figure.
    
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
    '''
    for i, (_, model_keys) in enumerate(grouped_keys_inner):

        for j, model_key in enumerate(model_keys):

            key_for = dict(zip(model_key_names, model_key))

            ax = fig.add_subplot(gs1[i, j])

            y_pred, _, y_true = get_in_sample_prediction(workflow, model_key)

            xlabel = f'Observed {prediction_label}'
            if i == len(grouped_keys_inner) - 1:
                model_build = label_for_model_build[key_for['model_build']]
                xlabel = f'{xlabel}\n' + model_build
            ax.set_xlabel(xlabel, size='small')

            ylabel = f'Predicted {prediction_label}'
            if j == 0:
                effect = label_for_effect[key_for['target_effect']]
                ylabel = f'{effect}\n' + ylabel
            ax.set_ylabel(ylabel, size='small')

            generate_scatterplot(
                ax, 
                y_true, 
                y_pred,
                workflow, 
                label_for_metric,
                color='black'
            )

            ax.tick_params(axis='both', labelsize='small')

            # TODO: Use the same scale in in benchmarking_scatter.
            xmin, xmax = ax.get_xlim()
            plot_one_one_line(ax, xmin, xmax)
#endregion

#region: _comma_separated
def _comma_separated(number):
    '''Convert float or int to a string with comma-separated thousands.
    '''
    return '{:,}'.format(number)
#endregion

# TODO: Is this helper function needed?
#region: _plot_prediction_scatterplot
def _plot_prediction_scatterplot(
        ax, 
        x, 
        y, 
        workflow, 
        label_for_metric,
        title=None, 
        xlabel=None, 
        ylabel=None
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

    ax.tick_params(axis='both', labelsize='small')
#endregion

#region: get_in_sample_prediction
def get_in_sample_prediction(workflow, model_key, inverse_transform=False):
    '''
    '''
    model_key_names = get_model_key_names(workflow)
    key_for = dict(zip(model_key_names, model_key))
    X, y_true = workflow.load_features_and_target(**key_for)

    y_pred, X = get_prediction(X, workflow, model_key, inverse_transform)

    return y_pred, X, y_true
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
def important_feature_counts(
        workflow, 
        label_for_effect, 
        feature_names_label
        ):
    '''
    '''
    # Define colorblind-friendly colors
    color_filled = '#1f77b4'  # Blue
    color_unfilled = '#dcdcdc'  # Light gray

    model_key_names = get_model_key_names(workflow)
    grouped_keys = group_model_keys(
        workflow.model_keys, model_key_names, 'target_effect', 
        string_to_exclude='without_selection'
    )

    for grouping_key, model_keys in grouped_keys:
        n_keys = len(model_keys)

        # Figure layout is adjusted to have one column per model key
        fig, axs = plt.subplots(1, n_keys, figsize=(5*n_keys, 8), sharey=True)

        for i, model_key in enumerate(model_keys):
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

            axs[i].barh(
                bar_positions, bar_counts, color=bar_colors, edgecolor='black', 
                linewidth=1)
            axs[i].set_ylim(-1, len(all_feature_names))
            axs[i].set_yticks(range(len(all_feature_names)))
            axs[i].set_yticklabels(sorted_features, fontsize=10)
            axs[i].tick_params(axis='y', pad=0)
            axs[i].invert_yaxis()
            axs[i].set_xlabel('Count', fontsize=12)
            axs[i].set_ylabel(feature_names_label, fontsize=12)
            axs[i].set_title(
                label_for_effect[key_for['target_effect']], fontsize=12)

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

        save_figure(
            fig, 
            important_feature_counts, 
            grouping_key
            )
#endregion

#region: importances_boxplots
def importances_boxplots(
        workflow, 
        label_for_scoring,
        feature_names_label, 
        xlabel='Δ Score',
        figsize=(8, 10)
        ):
    '''
    '''
    importances_wide = workflow.concatenate_history('importances')
    model_keys = [k for k in workflow.model_keys if 'with_selection' in k]

    _feature_importances_boxplots(
        importances_wide, 
        model_keys, 
        importances_boxplots,
        label_for_scoring,
        xlabel=xlabel,
        ylabel=feature_names_label,
        figsize=figsize
        )
#endregion

#region: importances_replicates_boxplots
def importances_replicates_boxplots(
        workflow, 
        label_for_scoring,
        feature_names_label,
        xlabel='Δ Score',
        figsize=(8, 10)
        ):
    '''
    '''
    importances_wide = workflow.concatenate_history('importances_replicates')
    model_keys = [k for k in workflow.model_keys if 'with_selection' in k]

    _feature_importances_boxplots(
        importances_wide, 
        model_keys, 
        importances_replicates_boxplots,
        label_for_scoring,
        xlabel=xlabel,
        ylabel=feature_names_label,
        figsize=figsize
        )
#endregion

#region: _feature_importances_boxplots
def _feature_importances_boxplots(
        df_wide, 
        model_keys,
        function, 
        label_for_scoring,
        xlabel,
        ylabel='', 
        figsize=None,
        ):
    '''
    Plot boxplots for feature importances for all models in workflow.

    Parameters
    ----------
    workflow : object
        The workflow object containing the model keys and corresponding 
        history.

    Returns
    -------
    None : None
    '''
    for model_key in model_keys:
        
        fig, axs = plt.subplots(
            ncols=len(label_for_scoring),
            sharey=True,
            figsize=figsize
        )

        for i, (scoring, title) in enumerate(label_for_scoring.items()):

            df_long = (
                df_wide[model_key][scoring]
                .melt()
                .sort_values(by='value', ascending=False) 
            )
        
            y, x = list(df_long.columns)
        
            sns_boxplot_wrapper(
                axs[i], 
                df_long,
                x, 
                y,
                xlabel=xlabel, 
                title=title,
                palette='icefire' 
            )

        if ylabel:
            # Set ylabel, first column only.
            for i, ax in enumerate(axs.flatten()):
                ax.tick_params(axis='y', size=10)
                if i == 0:
                    ax.set_ylabel(ylabel, size=12)
        
        fig.tight_layout()

        save_figure(
            fig, 
            function, 
            model_key
            )
#endregion

#region: sns_boxplot_wrapper
def sns_boxplot_wrapper(
        ax, 
        df_long, 
        x, 
        y, 
        xlabel='',
        ylabel='', 
        title='',
        **kwargs
        ):
    '''Wrapper around seaborn.boxplot() with customization.
    '''
    sns.boxplot(    
        x=x, 
        y=y, 
        data=df_long,
        linewidth=0.8,
        dodge=False,
        ax=ax,
        flierprops=_flierprops, 
        **kwargs
    )

    ## Set gridlines below all artists.
    ax.grid(axis='x', linestyle='--', linewidth=0.5)
    ax.grid(axis='y', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
#endregion

#region: benchmarking_scatterplots
def benchmarking_scatterplots(
        workflow,
        y_regulatory_df,
        y_toxcast,
        label_for_effect,
        color_for_effect,
        label_for_metric,
        prediction_label,
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
    grouped_keys = group_model_keys(
        workflow.model_keys, model_key_names, 'target_effect')

    for grouping_key, model_keys in grouped_keys:
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
                    xlabel = f'Regulatory {prediction_label}'
                if i == 0:  # first column
                    ylabel = f'{label} {prediction_label}'
                
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
            grouping_key
            )
#endregion

#region: generate_scatterplot
def generate_scatterplot(
        ax, 
        y_true, 
        y_pred, 
        workflow, 
        label_for_metric,
        with_sample_size=True,
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
    
    score_dict = {
        label : float_to_string(get_score(metric)) 
        for metric, label in label_for_metric.items()
        }
    if with_sample_size:
        score_dict['n'] = _comma_separated(len(y_true))

    ax.text(
        0.05, 
        0.95, 
        dict_to_string(score_dict), 
        transform=ax.transAxes,
        va='top', 
        ha='left', 
        size='small'
        )

    fontsize = 'medium'
    if title:
        ax.set_title(title, size=fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, size=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, size=fontsize)
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
    Plots margins of exposure for different chemicals. 

    Parameters
    ----------
    workflow : object
        An object representing the workflow.
    exposure_df : pd.DataFrame
        DataFrame with exposure estimates.
    label_for_effect : dict
        Dictionary mapping target effects to their labels.
    label_for_exposure_column : dict
        Dictionary mapping percentile columns to their labels.
    right_truncation : float, optional
        If provided, sets the right truncation limit for x-axis.

    Returns
    -------
    None
    '''
    # Get the error estimate for prediction intervals.
    rmse_for_model = (
        workflow.concatenate_history('performances')
        .xs('root_mean_squared_error', axis=1, level='metric')
        .quantile()
        .to_dict()
    )

    exposure_df = np.log10(exposure_df)

    model_key_names = get_model_key_names(workflow)
    grouped_keys = group_model_keys(
        workflow.model_keys, model_key_names, 'target_effect'
    )

    colors = sns.color_palette("Set2", len(exposure_df.columns))

    # Define the limits of the vertical spans in log10 units of MOE.
    # log10(0) is undefined and will be handled dynamically
    moe_categories = {
        'Potential Concern': (0., 2.),  # 1, 100
        'Definite Concern' : (-np.inf, 0.)  # 0, 1
    }
    moe_colors = sns.color_palette("Paired", len(moe_categories)+1)

    for grouping_key, model_keys in grouped_keys:

        fig, axs = plt.subplots(
            1,
            len(model_keys),
            figsize=(len(model_keys) * 5, 5),
        )

        for i, model_key in enumerate(model_keys):

            y_pred, *_ = predict_out_of_sample(
                workflow,
                model_key,
                inverse_transform=False
            )

            rmse = rmse_for_model[model_key]
            
            for j, percentile in enumerate(exposure_df.columns):

                ## Get the data for plotting.
                exposure_estimates = exposure_df[percentile]
                margins_of_exposure = y_pred - exposure_estimates  # in log10 scale
                sorted_moe = margins_of_exposure.sort_values()
                cumulative_counts = np.arange(1, len(sorted_moe) + 1)
                lb, ub = prediction_interval(sorted_moe, rmse)

                plot_with_prediction_interval(
                    axs[i], 
                    sorted_moe, 
                    cumulative_counts, 
                    lb, 
                    ub, 
                    colors[j], 
                    label=label_for_exposure_column[percentile]
                    )

            ## Update the limits.
            set_even_ticks(axs[i], axis_type='x', data_type='fill')
            set_even_log_ticks(axs[i], axis_type='y', data_type='fill')
            if right_truncation:
                axs[i].set_xlim(axs[i].get_xlim()[0], right_truncation)

            ## Set labels, etc. 
            key_for = dict(zip(model_key_names, model_key))
            effect = key_for['target_effect']
            axs[i].set_title(label_for_effect[effect])
            axs[i].set_xlabel("$log_{10}MOE$")
            axs[i].set_yscale('log')
            axs[i].grid(True, which='both', linestyle='--', linewidth=0.5)
            if i == 0: 
                axs[i].set_ylabel('Cumulative Count of Chemicals')

            annotate_vertical_spans(
                axs[i], 
                moe_categories, 
                moe_colors
                )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)  

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
            grouping_key
        )
#endregion

#region: plot_with_prediction_interval
def plot_with_prediction_interval(
        ax, 
        sorted_values, 
        cumulative_counts, 
        lower_bound, 
        upper_bound, 
        color, 
        label=None
    ):
    '''
    Plot values with prediction interval.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    sorted_values : pd.Series
        Sorted values in log10 scale.
    cumulative_counts : np.ndarray
        Cumulative counts for the sorted values.
    lower_bound : pd.Series
        Lower bound of the prediction interval.
    upper_bound : pd.Series
        Upper bound of the prediction interval.
    color : str
        Color of the plot.
    label : str, optional
        Label for the line plot.

    Returns
    -------
    None
    '''
    # Plot the prediction interval
    ax.fill_betweenx(
        cumulative_counts, 
        lower_bound, 
        upper_bound, 
        color=color, 
        alpha=0.2
        )

    # Plot the main line
    ax.plot(
        sorted_values, 
        cumulative_counts, 
        color=color, 
        label=label
        )
#endregion

#region: prediction_interval
def prediction_interval(prediction, error):
    '''
    Calculate the prediction interval.

    Parameters
    ----------
    prediction : pd.Series
        Predicted values in log10 scale.
    error : float
        Measure of uncertainty, such as the Root Mean Squared Error of a model.

    Returns
    -------
    lower_bound : pd.Series
        Lower bound of the prediction interval.
    upper_bound : pd.Series
        Upper bound of the prediction interval.
    '''
    lower_bound = prediction - 1.645 * error
    upper_bound = prediction + 1.645 * error
    return lower_bound, upper_bound
#endregion

#region: annotate_vertical_spans
def annotate_vertical_spans(ax, categories, colors, y_pos_axes=0.97):
    '''
    Annotate vertical spans and category labels in a matplotlib Axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object in which to annotate vertical spans.
    categories : dict
        Dictionary defining the categories and their corresponding 
        vertical span limits.
    colors : list
        List of colors for the vertical spans.
    y_pos_axes : float
        The y-position of the category labels in Axes fraction units.

    Returns
    -------
    None
    '''
    for k, (category, (lower, upper)) in enumerate(categories.items()):
        if lower == -np.inf:
            # Extend the lower limit to the xmin of the Axes
            lower = ax.get_xlim()[0]

        ax.axvspan(lower, upper, alpha=0.2, color=colors[k])

        x_pos_data = (lower + upper) / 2  # arithmetic mean in linear scale

        # Convert to data coordinates
        coords_axes = (0, y_pos_axes)
        _, y_pos_data = ax.transData.inverted().transform(
            ax.transAxes.transform(coords_axes))

        ax.text(
            x_pos_data,
            y_pos_data,
            category.replace(" ", "\n"),
            ha='center', 
            va='top',
            fontsize='small',
        )
#endregion

#region: set_even_ticks
def set_even_ticks(ax, axis_type='x', data_type='line'):
    '''
    Set the ticks on an axis of a matplotlib Axes object to be at even 
    intervals.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object on which to set the ticks.
    axis_type : str
        The axis on which to set the ticks. Should be either 'x' or 'y'.

    Returns
    -------
    None
    '''
    # Define the sequence of "nice" numbers
    nice_numbers = [
        1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    data_min, data_max = get_data_limits(
        ax, 
        axis_type=axis_type, 
        data_type=data_type
        )
    
    # Handle when data_min or data_max is NaN
    if data_min is np.nan or data_max is np.nan:  
        return
    
    # Calculate the range of data and the raw step
    data_range = data_max - data_min
    raw_step = data_range / 10  # We desire around 10 ticks on the axis

    # Get the "nice" step by finding the closest nice number to the raw step
    nice_step = min(nice_numbers, key=lambda x:abs(x-raw_step))

    # Calculate new minimum and maximum values to be multiples of nice_step
    data_min = nice_step * np.floor(data_min/nice_step)
    data_max = nice_step * np.ceil(data_max/nice_step)
    
    if axis_type == 'x':
        ax.set_xlim(data_min, data_max)
        # +nice_step to include data_max in the ticks
        ax.set_xticks(
            np.arange(data_min, data_max + nice_step, step=nice_step))  
    elif axis_type == 'y':
        ax.set_ylim(data_min, data_max)
        # +nice_step to include data_max in the ticks
        ax.set_yticks(
            np.arange(data_min, data_max + nice_step, step=nice_step))  
#endregion

#region: set_even_log_ticks
def set_even_log_ticks(ax, axis_type='x', data_type='line'):
    '''
    Set the ticks on a logarithmic axis of a matplotlib Axes object to be at 
    even intervals.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object on which to set the ticks.
    axis_type : str
        The axis on which to set the ticks. Should be either 'x' or 'y'.

    Returns
    -------
    None
    '''
    # Get the limits of the data
    data_min, data_max = get_data_limits(
        ax, 
        axis_type=axis_type, 
        data_type=data_type
        )

    # Get the powers of 10 that bound the data
    data_min_pow = np.floor(np.log10(data_min))
    data_max_pow = np.ceil(np.log10(data_max))

    # Make sure the exponents are even
    if data_min_pow % 2 != 0:
        data_min_pow -= 1
    if data_max_pow % 2 != 0:
        data_max_pow += 1

    # Create the list of ticks
    # Step by 2 to get even exponents
    ticks = [
        10**i for i in range(int(data_min_pow), int(data_max_pow) + 1, 2)]  

    # Set the ticks on the appropriate axis
    if axis_type == 'x':
        ax.set_xticks(ticks)
    elif axis_type == 'y':
        ax.set_yticks(ticks)
#endregion

#region: get_data_limits
def get_data_limits(ax, axis_type='x', data_type='line'):
    '''
    Get the minimum and maximum limits of the data on a specific axis of a 
    matplotlib Axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object containing the data.
    axis_type : str
        The axis for which to get the data limits. Should be either 'x' or 'y'.
    data_type : str
        The type of data for which to get the limits. Should be either 'line' 
        or 'fill'.

    Returns
    -------
    float, float
        The minimum and maximum data limits.
    '''
    if data_type == 'line':
        # Get the data from line plots
        if axis_type == 'x':
            lines = ax.get_lines()
            data = np.concatenate([line.get_xdata() for line in lines])
        elif axis_type == 'y':
            lines = ax.get_lines()
            data = np.concatenate([line.get_ydata() for line in lines])
    elif data_type == 'fill':
        # Get the data from filled areas
        collections = ax.collections
        axis_index = 0 if axis_type == 'x' else 1
        data = np.concatenate([collection.get_paths()[0].vertices[:, axis_index] for collection in collections])
    else:
        raise ValueError(f"Invalid data_type: {data_type}. Choose either 'line' or 'fill'.")

    # Exclude NaN values
    data = data[~np.isnan(data)]
    
    # Check if data is empty after excluding NaN values
    if len(data) == 0:
        # Return None for both min and max if no valid data is available
        return None, None  

    # Get the min and max
    data_min = np.min(data)
    data_max = np.max(data)

    return data_min, data_max
#endregion

#region: predictions_by_missing_feature
def predictions_by_missing_feature(
    workflow, 
    label_for_effect, 
    prediction_label
    ):
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
        grouped_keys = group_model_keys(
            workflow.model_keys, model_key_names, 'target_effect')
        
        # Iterate over grouping_key_group
        for grouping_key, model_keys in grouped_keys:
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
                    remaining_color,
                    prediction_label
                    )
                # Use the sames sort order of boxes, based on the left Axes.
                sort_order = list(dfs_out.keys())
                dfs_in = _boxplot_by_missing_feature(
                    axs[i, 1], 
                    y_pred_in, 
                    X_in, 
                    all_samples_color, 
                    remaining_color, 
                    prediction_label,
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
                grouping_key
                )
#endregion

#region: _boxplot_by_missing_feature
def _boxplot_by_missing_feature(
        ax, 
        df, 
        X, 
        all_samples_color, 
        remaining_color, 
        prediction_label, 
        sort_order=None
        ):
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

    _create_boxplot(
        ax, 
        df_for_name, 
        all_samples_color, 
        remaining_color, 
        prediction_label
        )

    return df_for_name
#endregion

#region: _create_boxplot
def _create_boxplot(
        ax, 
        df_for_name, 
        all_samples_color, 
        remaining_color, 
        prediction_label
        ):
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
    boxplot = ax.boxplot(
        list(df_for_name.values()),
        vert=False,
        labels=[None]*len(df_for_name),  # will be updated later
        widths=0.6,
        patch_artist=True,
        medianprops={'color': 'black'},
        flierprops=_flierprops
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

    ax.set_xlabel(f'Predicted {prediction_label}')
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

#region: group_model_keys
def group_model_keys(
        model_keys, 
        key_names, 
        exclusion_key_names , 
        string_to_exclude=None
    ):
    '''
    Group model keys by grouping keys. A grouping key is formed by taking a model 
    key and excluding one or more of its elements.

    Parameters
    ----------
    model_keys : list of tuples
        Model keys to be grouped. Each tuple represents a model key.
    key_names : list
        Names of keys corresponding to elements in the model keys. The order of 
        names should match the order of elements in each tuple of model_keys.
    exclusion_key_names  : str or list of str
        Names of keys (which should be in key_names) to exclude when forming 
        the grouping key.
    string_to_exclude : str, optional
        String to exclude model keys containing it. If a model key contains this 
        string, it will be excluded from the final output. If None, no model keys 
        are excluded based on this criterion.

    Returns
    -------
    grouped_model_keys : list of 2-tuple
        Contains each grouping key and its corresponding group of model keys.
    '''

    # Convert exclusion_key_names  to list if it's a string
    if isinstance(exclusion_key_names , str):
        exclusion_key_names  = [exclusion_key_names ]

    # Exclude model keys that contain string_to_exclude
    if string_to_exclude:
        model_keys = [k for k in model_keys if string_to_exclude not in k]

    # Get the indices of the keys to exclude in the key names
    exclusion_key_indices = [key_names.index(key) for key in exclusion_key_names ]

    def create_grouping_key(model_key):
        return tuple(item for idx, item in enumerate(model_key) 
                     if idx not in exclusion_key_indices)

    # Sort model keys by grouping key
    # This is necessary because itertools.groupby() groups only consecutive 
    # elements with the same key
    sorted_model_keys = sorted(model_keys, key=create_grouping_key)

    # Group the sorted keys by grouping key
    grouped_model_keys = [
        (grouping_key, list(group))
        for grouping_key, group in itertools.groupby(
        sorted_model_keys, key=create_grouping_key)
    ]

    return grouped_model_keys
#endregion

# TODO: Move this to the Workflow.History?
#region: get_model_key_names
def get_model_key_names(workflow):
    return workflow.instruction_names + ['estimator']
#endregion

#region: save_figure
def save_figure(fig, function, fig_label, extension='.png'):
    '''
    '''
    output_dir = function_directory(function)
    fig_path = figure_path(output_dir, fig_label, extension=extension)
    print(f'Saving figure --> "{fig_path}"')
    fig.savefig(fig_path)
#endregion

#region: figure_path
def figure_path(function_dir, fig_label, extension='.png'):
    '''
    '''
    if isinstance(fig_label, (tuple, list)):
        fig_label = '-'.join(map(str, fig_label))
    filename = fig_label.replace(' ', '-').replace('/', '-') + extension
    return os.path.join(function_dir, filename)
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