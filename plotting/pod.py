'''
'''

import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
import seaborn as sns
import itertools

from . import utilities

#region: cumulative_pod_distributions
def cumulative_pod_distributions(results_analyzer, plot_settings):
    '''
    '''
    colors = sns.color_palette('colorblind')
    linestyles = [
        style for style in mlines.lineStyles.keys() 
        if isinstance(style, str)
        ]
    grouped_keys = results_analyzer.group_model_keys('target_effect')

    for grouping_key, model_keys in grouped_keys:

        fig, axs = plt.subplots(
            2, 
            len(model_keys), 
            figsize=(len(model_keys)*4, 8)
            )
        
        global_xlim = [float('inf'), float('-inf')]  # initialize

        for i, model_key in enumerate(model_keys):

            y_for_label = results_analyzer.get_pod_comparison_data(model_key)
            
            effect_index = (
                results_analyzer.read_model_key_names()
                .index('target_effect')
            )
            effect = model_key[effect_index]

            line_cycle = itertools.cycle(linestyles)

            # Compute intersection of samples
            common_samples = (
                y_for_label['Regulatory'].index
                .intersection(y_for_label['ToxValDB'].index)
                .intersection(y_for_label['QSAR'].index)
            )
            
            # Plot CDFs for intersection of samples in the first row
            for j, (label, data_series) in enumerate(y_for_label.items()):
                _plot_cdf(
                    axs[0, i],
                    data_series.loc[common_samples], 
                    results_analyzer, 
                    colors[j], 
                    next(line_cycle), 
                    label, 
                    global_xlim
                )

            # Reset linestyle cycle for the next row
            line_cycle = itertools.cycle(linestyles)

            # Plot CDFs as in original in the second row
            for j, (label, data_series) in enumerate(y_for_label.items()):
                _plot_cdf(
                    axs[1, i], 
                    data_series, 
                    results_analyzer,
                    colors[j], 
                    next(line_cycle), 
                    label, 
                    global_xlim
                )

            # Set labels and other properties
            axs[0, i].set_title(plot_settings.label_for_effect[effect])
            for ax_row in axs[:, i]:
                ax_row.set_xlabel("$log_{10}POD$")
                ax_row.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax_row.set_xlim(global_xlim)

            if i == 0: 
                axs[0, i].set_ylabel('Proportion of Chemicals (Intersection)')
                axs[1, i].set_ylabel('Proportion of Chemicals')

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.1)

        legend_ax = axs[-1][-1]
        handles, labels = legend_ax.get_legend_handles_labels()
        fig.legend(
            handles, 
            labels, 
            loc='lower center', 
            fontsize='small',
            ncol=len(labels), 
            bbox_to_anchor=(0.5, -0.01)
        )

        utilities.save_figure(
            fig, 
            cumulative_pod_distributions, 
            grouping_key
        )
#endregion

#region: _plot_cdf
def _plot_cdf(
        ax, 
        data_series, 
        results_analyzer, 
        color, 
        linestyle, 
        label, 
        global_xlim
        ):
    '''
    Helper function to plot CDF and update global x limits.
    '''
    sorted_values, cumulative_proportions = (
        results_analyzer.generate_cdf_data(data_series, normalize=True))
    
    ax.plot(
        sorted_values, 
        cumulative_proportions,
        color=color,
        linestyle=linestyle,
        label=label
    )
    
    # Update global x limits
    global_xlim[0] = min(global_xlim[0], sorted_values.min())
    global_xlim[1] = max(global_xlim[1], sorted_values.max())
#endregion