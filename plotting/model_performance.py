'''
'''

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np

from . import utilities

# TODO: Move to configuration file.
_flierprops = dict(
    marker='o', 
    markerfacecolor='lightgray', 
    markersize=2,
    linestyle='none', 
    markeredgecolor='lightgray'
    )

#region: in_and_out_sample_comparisons
def in_and_out_sample_comparisons(
        results_analyzer, 
        plot_settings, 
        function_for_metric, 
        xlim=(0., 1.)
        ):
    '''
    Generate in-sample performance comparisons and out-of-sample prediction 
    scatterplots.

    This function only handles model groupings containing both effect types
    with and without feature selection (i.e., four model keys).
    
    Parameters
    ----------
    xlim : tuple, optional
        Tuple specifying the x-axis limits. Default is (0., 1.).
    '''
    model_key_names = results_analyzer.read_model_key_names()
    grouped_keys_outer = results_analyzer.group_model_keys(
        ['target_effect', 'select_features']
        )

    for grouping_key_outer, model_keys in grouped_keys_outer:

        # Check if the grouping has 2 effect types + 2 feature selection types
        if len(model_keys) != 4:
            continue  # next iteration

        grouped_keys_inner = results_analyzer.group_model_keys(
            'select_features',
            model_keys=model_keys
            )
        
        in_sample_performance_comparisons(
            results_analyzer, 
            grouped_keys_inner, 
            model_key_names,
            grouping_key_outer,
            function_for_metric,
            plot_settings,
            xlim=xlim
        )

        out_of_sample_prediction_scatterplots(
            results_analyzer, 
            grouped_keys_inner,
            grouping_key_outer, 
            model_key_names, 
            function_for_metric,
            plot_settings
    )
#endregion

#region: in_sample_performance_comparisons
def in_sample_performance_comparisons(
        results_analyzer, 
        grouped_keys_inner, 
        model_key_names,
        grouping_key_outer,
        function_for_metric,
        plot_settings,
        xlim=(0., 1.)
    ):
    '''
    Generate in-sample performance comparisons plots using scatterplots and 
    boxplots.
    '''
    # Initialize a Figure for the subplot.
    fig = plt.figure(figsize=(7, 5))

    gs1 = gridspec.GridSpec(2, 2)
    gs2 = gridspec.GridSpec(6, 1)

    _in_sample_performance_scatterplots(
        fig, 
        gs1, 
        results_analyzer, 
        grouped_keys_inner, 
        model_key_names,
        function_for_metric,
        plot_settings
        )
    
    _in_sample_performance_boxplots(
        fig, 
        gs2, 
        results_analyzer, 
        grouped_keys_inner, 
        plot_settings,
        xlim=xlim
        )

    # adjust the 'right' value for gs1 and 'left' value for gs2
    gs1.tight_layout(fig, rect=[0, 0, 0.7, 1]) 
    gs2.tight_layout(fig, rect=[0.7, 0, 1, 1], h_pad=0.5)

    utilities.save_figure(
        fig, 
        in_sample_performance_comparisons, 
        grouping_key_outer
        )
#endregion

#region: _in_sample_performance_scatterplots
def _in_sample_performance_scatterplots(
        fig, 
        gs1, 
        results_analyzer, 
        grouped_keys_inner, 
        model_key_names, 
        function_for_metric,
        plot_settings
    ):
    '''
    Generate scatterplots of observed vs predicted for the in-sample 
    performance comparisons.
    '''

    title = '(A) In-Sample Performance'

    all_axs = []
    # Initialize the limits.
    xmin, xmax = np.inf, -np.inf

    for i, (_, model_keys) in enumerate(grouped_keys_inner):

        for j, model_key in enumerate(model_keys):

            key_for = dict(zip(model_key_names, model_key))

            ax = fig.add_subplot(gs1[i, j])
            all_axs.append(ax)

            y_pred, _, y_true = results_analyzer.get_in_sample_prediction(model_key)

            ## Set labels depending on the Axes.
            xlabel, ylabel = '', ''
            if i == len(grouped_keys_inner) - 1:
                select_features = plot_settings.label_for_select_features[key_for['select_features']]
                xlabel = f'Observed {plot_settings.prediction_label}\n{select_features}'
            if j == 0:
                effect = plot_settings.label_for_effect[key_for['target_effect']]
                ylabel = f'{effect}\nPredicted {plot_settings.prediction_label}'
                if i == 0:
                    ax.set_title(title, loc='left', size='small', style='italic')
            ax.set_xlabel(xlabel, size='small')
            ax.set_ylabel(ylabel, size='small')
            
            utilities.generate_scatterplot(
                ax, 
                y_true, 
                y_pred,
                function_for_metric, 
                plot_settings.label_for_metric,
                color='black'
            )

            # Update the limits for the one-one line.
            xmin = min(xmin, *ax.get_xlim())
            xmax = max(xmax, *ax.get_xlim())

            ax.tick_params(axis='both', labelsize='small')

    # Use the same scale.
    for ax in all_axs:
        utilities.plot_one_one_line(ax, xmin, xmax)
#endregion

#region: _in_sample_performance_boxplots
def _in_sample_performance_boxplots(
        fig, 
        gs2, 
        results_analyzer, 
        grouped_keys_inner, 
        plot_settings,
        xlim=(0., 1.)
        ):
    '''
    Create boxplots for in-sample performances across different models and 
    metrics.
    '''

    title = '(B) Out-of-Sample Performance'

    # Create a counter for the current row
    index = 0
    for i, (_, model_keys) in enumerate(grouped_keys_inner):

        ## Prepare the data
        performances_wide = results_analyzer.combine_results(
            'performances', 
            model_keys=model_keys
            )
        # Filter the data.
        where_subset_metrics = (
            performances_wide.columns
            .get_level_values('metric')
            .isin(plot_settings.label_for_metric)
        )
        performances_wide = performances_wide.loc[:, where_subset_metrics]
        # Rename the columns for visualization.
        label_for_select_features = {
            k : v.split(' ')[0] for k, v in plot_settings.label_for_select_features.items()}
        label_for_column = {
            **plot_settings.label_for_metric, 
            **label_for_select_features
        }
        performances_wide = performances_wide.rename(label_for_column, axis=1)

        metrics = performances_wide.columns.unique(level='metric')

        # The new keys will be used to get the data.
        model_keys_renamed = [
            tuple(label_for_column.get(k, k) for k in model_key) 
            for model_key in model_keys
        ]
        
        reverse_metric = plot_settings.label_for_metric.get(
            'r2_score', 'r2_score'
        )

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
                y='select_features', 
                x='value', 
                ax=ax,
                flierprops=_flierprops,
                orient='h' ,
                linewidth=1.
            )

            utilities.set_axis_limit(
                ax, 
                metric,
                limit_values=xlim,
                reverse_metric=reverse_metric
            )

            # Set labels. 
            ax.set_xlabel(metric, size='small') 
            ax.set_ylabel('')
            ax.tick_params(axis='both', labelsize='small')
            if i == j == 0:
                ax.set_title(title, size='small', loc='right', style='italic')

            # Increase the counter
            index += 1
#endregion

#region: out_of_sample_prediction_scatterplots
def out_of_sample_prediction_scatterplots(
        results_analyzer, 
        grouped_keys_inner,
        grouping_key_outer, 
        model_key_names, 
        function_for_metric,
        plot_settings
    ):
    '''
    Generate and plot scatterplots for out-of-sample predictions.
    '''
    fig, axs = plt.subplots(
        ncols=len(grouped_keys_inner),
        figsize=(7, 3.5),
        sharey=True
    )

    # Initialize the limits.
    xmin, xmax = np.inf, -np.inf

    for i, (_, model_keys) in enumerate(grouped_keys_inner):
            
        ## Get the predictions for all data.

        key_without_selection = next(k for k in model_keys if 'false' in k)
        key_with_selection = next(k for k in model_keys if 'true' in k)

        y_pred_without, *_ = results_analyzer.predict_out_of_sample(key_without_selection)
        y_pred_with, *_ = results_analyzer.predict_out_of_sample(key_with_selection)

        ## Define figure labels.
        
        # Set the title as the common effect.
        effects = []
        for model_key in model_keys:
            key_for = dict(zip(model_key_names, model_key))
            effects.append(key_for['target_effect'])
        if not all(effect == effects[0] for effect in effects):
            raise ValueError(f'Inconsistent target effects: {effects}')
        title = plot_settings.label_for_effect[effects[0]]
        
        def create_label(select_features):
            return f'{plot_settings.prediction_label} {plot_settings.label_for_select_features[select_features]}'
        
        # The labeled samples will be plotted in a separate color.
        _, y_true = results_analyzer.load_features_and_target(**key_for)
        labeled_samples = list(y_true.index)

        utilities.generate_scatterplot(
            axs[i], 
            y_pred_without, 
            y_pred_with,
            function_for_metric, 
            plot_settings.label_for_metric,
            highlight_indices=labeled_samples,
            main_size=plot_settings.marker_size_for_sample_type['out'],
            highlight_size=plot_settings.marker_size_for_sample_type['in'],
            main_label=plot_settings.label_for_sample_type['out'],
            highlight_label=plot_settings.label_for_sample_type['in'],
            color=plot_settings.color_for_sample_type['out'], 
            highlight_color=plot_settings.color_for_sample_type['in'],
            title=title, 
            xlabel=create_label('false'), 
            ylabel=create_label('true') if i == 0 else ''
        )

        # Update the limits for the one-one line.
        xmin = min(xmin, *axs[i].get_xlim())
        xmax = max(xmax, *axs[i].get_xlim())

    # Use the same scale.
    for ax in axs.flatten():
        utilities.plot_one_one_line(ax, xmin, xmax)

    fig.tight_layout()

    utilities.save_figure(
        fig, 
        out_of_sample_prediction_scatterplots, 
        grouping_key_outer
        )
#endregion