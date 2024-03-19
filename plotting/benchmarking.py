'''
Plotting module for the model performance benchmarking.

See Also
--------
plot.py
    The main plotting module where this sub-module is implemented as part of 
    the main package.
'''

import matplotlib.pyplot as plt
import numpy as np

from . import utilities

#region: benchmarking_scatterplots
def benchmarking_scatterplots(
        results_analyzer,
        function_for_metric,
        plot_settings,
        figsize=(6, 9)
        ):
    '''
    Generate scatterplots for each unique combination of the evaluation and
    comparison datasets. One subplot is created for each model key grouped by
    target effect.

    Parameters
    ----------
    results_analyzer : instance of results_analysis.ResultsAnalyzer
        Manages the analysis of results. 
    function_for_metric : dict
        Mapping of performance metrics to their corresponding functions.
    plot_settings : SimpleNamespace
        Configuration settings for plotting.
    figsize : tuple, optional
        Figure size. If None, a default size is used.

    Returns
    -------
    None
        The figures are saved to a dedicated directory derived from the 
        function name.
    '''
    y_auth_df = results_analyzer.load_authoritative_pods()
    y_toxcast = results_analyzer.load_oral_equivalent_doses()

    model_key_names = results_analyzer.read_model_key_names()
    grouped_keys = results_analyzer.group_model_keys('target_effect')

    for grouping_key, model_keys in grouped_keys:
        num_subplots = len(model_keys)

        fig, ax_objs = plt.subplots(3, num_subplots, figsize=figsize)

        # Initialize the limits.
        xmin, xmax = np.inf, -np.inf

        for i, model_key in enumerate(model_keys):
            
            y_pred, _, y_true = results_analyzer.get_in_sample_prediction(model_key)

            key_for = dict(zip(model_key_names, model_key))
            y_comparison = y_auth_df[key_for['target_effect']].dropna()
            y_evaluation_dict = {
                plot_settings.surrogate_label : y_true, 
                plot_settings.qsar_label : y_pred,
                plot_settings.toxcast_label : y_toxcast,
            }

            for j, (label, y_evaluation) in enumerate(
                    y_evaluation_dict.items()):

                ax = ax_objs[j, i]

                color = plot_settings.color_for_effect[key_for['target_effect']]

                ## Set labels depending on the Axes.
                title, xlabel, ylabel = '', '', ''
                if j == 0:  # first row
                    title = plot_settings.label_for_effect[key_for['target_effect']]
                if j == len(y_evaluation_dict)-1:  # last row
                    xlabel = f'Authoritative {plot_settings.prediction_label}'
                if i == 0:  # first column
                    ylabel = f'{label} {plot_settings.prediction_label}'
                
                utilities.generate_scatterplot(
                    ax, 
                    y_comparison, 
                    y_evaluation, 
                    function_for_metric, 
                    plot_settings.label_for_metric,
                    color=color, 
                    title=title, 
                    xlabel=xlabel, 
                    ylabel=ylabel
                    )

                # Update the limits for the one-one line.
                xmin = min(xmin, *ax.get_xlim())
                xmax = max(xmax, *ax.get_xlim())

            # Use the same scale.
            for ax in ax_objs.flatten():
                utilities.plot_one_one_line(ax, xmin, xmax, color='#808080')

        fig.tight_layout()
        
        utilities.save_figure(
            fig, 
            benchmarking_scatterplots, 
            grouping_key
            )
#endregion
